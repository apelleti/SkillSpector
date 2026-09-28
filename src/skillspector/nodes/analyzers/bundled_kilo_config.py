# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze bundled Kilo configuration, agent, plugin, and script surfaces.

Kilo (Kilo Code / Kilo CLI) loads project configuration from ``kilo.json(c)``
and the legacy ``opencode.json(c)``, at the project root and inside its
``.kilo/`` and legacy ``.kilocode/`` directories. From those directories it
also loads agent and mode definitions (whose frontmatter may carry
permissions) and imports plugin modules on startup; the Kilo Agent Manager
runs ``.kilo/setup-script`` when it creates a worktree. A skill bundle that
ships these files can, once its directory becomes (or is copied into) a Kilo
project, pre-approve tools, start MCP server processes, run code, or pull
further skills and instructions from outside the reviewed bundle.

Behavior was verified against Kilo CLI 7.8.1 (``kilo debug config``,
``kilo debug agent``, ``kilo debug skill``).

Rules:

- KC1: permission grants that lift a prompt of Kilo's built-in rules
  (globally, per tool, or through a wildcard pattern; also inside
  ``agent``/``mode`` entries and agent/mode frontmatter).
- KC2: MCP servers (``local`` servers run a command; ``remote`` servers
  receive tool traffic).
- KC3: code execution (``plugin`` entries, plugin modules, and Agent Manager
  setup/run scripts).
- KC4: skill and instruction sources outside the bundle (``skills.urls``,
  ``skills.paths`` and ``instructions`` pointing outside the bundle).
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import yaml

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .static_runner import MAX_FILE_CHARS, analyzer_finding_to_finding

ANALYZER_ID = "bundled_kilo_config"
logger = get_logger(__name__)

_CONFIG_PATHS: Final = frozenset(
    {
        "kilo.json",
        "kilo.jsonc",
        ".kilo/kilo.json",
        ".kilo/kilo.jsonc",
        ".kilocode/kilo.json",
        ".kilocode/kilo.jsonc",
        # Kilo still reads the OpenCode files it was forked from, both at the
        # project root and inside its config directories.
        "opencode.json",
        "opencode.jsonc",
        ".kilo/opencode.json",
        ".kilo/opencode.jsonc",
        ".kilocode/opencode.json",
        ".kilocode/opencode.jsonc",
    }
)
# Kilo imports these modules as soon as it starts in the project.
_PLUGIN_PATH: Final = re.compile(r"\A\.kilo(?:code)?/plugins?/[^/]+\.(?:ts|js)\Z")
# Agent and mode definitions whose frontmatter may carry a ``permission`` map.
_AGENT_FILE_PATH: Final = re.compile(r"\A\.kilo(?:code)?/(?:agents?/.+|modes?/[^/]+)\.md\Z")
# Agent Manager (Kilo VS Code extension) runs the setup script automatically
# when it creates a worktree, and the run script from the Run button.
_AGENT_MANAGER_SCRIPT_PATH: Final = re.compile(
    r"\A\.kilo/(?P<kind>setup|run)-script(?:\.(?:sh|ps1|cmd|bat))?\Z"
)
_MAX_FRONTMATTER_CHARS: Final = 256 * 1024

_MAX_DECLARATIONS: Final = 2_048
_MAX_EVIDENCE_ITEMS: Final = 32
_MAX_EVIDENCE_CHARS: Final = 200

# Kilo appends project rules after its built-in ones and the last matching
# rule wins. The built-in ``code`` agent (Kilo CLI 7.8.1) already allows every
# tool except these, so an ``allow`` elsewhere removes no prompt and is not
# reported. ``read`` only prompts for ``.env`` files.
_PROMPTED_TOOL_SEVERITY: Final = {
    "bash": Severity.HIGH,
    "external_directory": Severity.HIGH,
    "read": Severity.HIGH,
    "doom_loop": Severity.MEDIUM,
}
_WILDCARD_PATTERNS: Final = frozenset({"*", "**"})
_SEVERITY_RANK: Final = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class _DuplicateKeyError(ValueError):
    """Raised when a document repeats an object key."""


class _DeclarationLimitError(ValueError):
    """Raised when a document declares more records than the analyzer models."""


@dataclass(frozen=True, slots=True)
class _Declaration:
    rule_id: str
    severity: Severity
    kind: str
    subject: str


# ---------------------------------------------------------------------------
# Bounded JSONC parsing
# ---------------------------------------------------------------------------


def _strip_jsonc(text: str) -> str:
    """Remove JSONC comments and trailing commas outside of string literals."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated block comment")
            out.append(" ")
            i = end + 2
            continue
        if char == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            # A comment between the comma and the closing bracket is handled by
            # the next iterations; only drop commas directly before a closer.
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(char)
        i += 1
    if in_string:
        raise ValueError("unterminated string")
    return "".join(out)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(value)


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(value)
    return parsed


def _parse_document(content: str) -> object:
    stripped = _strip_jsonc(content)
    # Commas left before a closer by an intervening comment are removed in a
    # second pass over the comment-free text.
    stripped = _strip_jsonc(stripped)
    return json.loads(
        stripped,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite_constant,
        parse_float=_parse_finite_float,
    )


# ---------------------------------------------------------------------------
# Declaration extraction
# ---------------------------------------------------------------------------


def _bounded(value: str) -> str:
    return value[:_MAX_EVIDENCE_CHARS]


def _tool_severity(tool: str) -> Severity | None:
    """Severity of approving ``tool``, which may itself be a glob over tool names."""
    if tool == "*":
        return Severity.CRITICAL
    matched = [
        severity
        for name, severity in _PROMPTED_TOOL_SEVERITY.items()
        if fnmatch.fnmatchcase(name, tool)
    ]
    if not matched:
        return None
    return max(matched, key=_SEVERITY_RANK.__getitem__)


def _pattern_declaration(tool: str, subject: str, pattern: str) -> _Declaration | None:
    """Classify one ``allow`` pattern of a prompted tool."""
    severity = _tool_severity(tool)
    if severity is None:
        return None
    if pattern in _WILDCARD_PATTERNS:
        return _Declaration("KC1", severity, "tool_allow", subject)
    if tool == "read":
        # Only the built-in ``.env`` prompt can be lifted for ``read``.
        if ".env" in pattern or fnmatch.fnmatchcase(".env", pattern):
            return _Declaration(
                "KC1", severity, "secret_read_allow", f"{subject}:{_bounded(pattern)}"
            )
        return None
    if "*" in pattern:
        return _Declaration(
            "KC1", Severity.MEDIUM, "pattern_allow", f"{subject}:{_bounded(pattern)}"
        )
    return None


def _permission_declarations(permission: object, scope: str) -> list[_Declaration]:
    """Return KC1 declarations for one ``permission`` value."""
    if permission == "allow":
        return [_Declaration("KC1", Severity.CRITICAL, "global_allow", scope or "*")]
    if not isinstance(permission, dict):
        return []
    declarations: list[_Declaration] = []
    for tool, value in permission.items():
        if not isinstance(tool, str):
            continue
        severity = _tool_severity(tool)
        if severity is None:
            continue
        subject = f"{scope}.{tool}" if scope else tool
        if value == "allow":
            declarations.append(_Declaration("KC1", severity, "tool_allow", subject))
        elif isinstance(value, dict):
            for pattern, action in value.items():
                if action == "allow" and isinstance(pattern, str):
                    declaration = _pattern_declaration(tool, subject, pattern)
                    if declaration is not None:
                        declarations.append(declaration)
    return declarations


def _agent_permission_declarations(document: dict[str, object]) -> list[_Declaration]:
    declarations: list[_Declaration] = []
    for section in ("agent", "mode"):
        agents = document.get(section)
        if not isinstance(agents, dict):
            continue
        for name, agent in agents.items():
            if isinstance(agent, dict) and "permission" in agent:
                declarations.extend(
                    _permission_declarations(agent["permission"], f"{section}.{name}")
                )
    return declarations


def _mcp_declarations(document: dict[str, object]) -> list[_Declaration]:
    servers = document.get("mcp")
    if not isinstance(servers, dict):
        return []
    declarations: list[_Declaration] = []
    for name, server in servers.items():
        if not isinstance(server, dict) or server.get("enabled") is False:
            continue
        server_type = server.get("type")
        command = server.get("command")
        url = server.get("url")
        if server_type == "local" or (server_type is None and command is not None):
            program = ""
            if isinstance(command, list) and command and isinstance(command[0], str):
                program = command[0]
            elif isinstance(command, str):
                program = command.split(" ", 1)[0]
            declarations.append(
                _Declaration("KC2", Severity.HIGH, "local_server", f"{name}:{_bounded(program)}")
            )
        elif server_type == "remote" or (server_type is None and url is not None):
            target = url if isinstance(url, str) else ""
            declarations.append(
                _Declaration("KC2", Severity.MEDIUM, "remote_server", f"{name}:{_bounded(target)}")
            )
    return declarations


def _plugin_declarations(document: dict[str, object]) -> list[_Declaration]:
    plugins = document.get("plugin")
    if isinstance(plugins, str):
        plugins = [plugins]
    if not isinstance(plugins, list):
        return []
    return [
        _Declaration("KC3", Severity.HIGH, "config_plugin", _bounded(plugin))
        for plugin in plugins
        if isinstance(plugin, str) and plugin.strip()
    ]


def _is_remote_reference(value: str) -> bool:
    return bool(re.match(r"(?i)\A[a-z][a-z0-9+.-]*://", value)) and not value.lower().startswith(
        "file://"
    )


def _is_outside_bundle(value: str) -> bool:
    """Whether a local path reference escapes the bundle directory."""
    normalized = value.replace("\\", "/")
    if normalized.lower().startswith("file://"):
        return True
    if normalized.startswith(("/", "~")) or re.match(r"\A[A-Za-z]:/", normalized):
        return True
    if normalized.startswith("$") or "${" in normalized:
        return True
    return ".." in normalized.split("/")


def _source_declarations(document: dict[str, object]) -> list[_Declaration]:
    declarations: list[_Declaration] = []
    skills = document.get("skills")
    if isinstance(skills, dict):
        urls = skills.get("urls")
        if isinstance(urls, list):
            declarations.extend(
                _Declaration("KC4", Severity.HIGH, "remote_skills", _bounded(url))
                for url in urls
                if isinstance(url, str) and url.strip()
            )
        paths = skills.get("paths")
        if isinstance(paths, list):
            # Kilo CLI 7.8.1 ignores project-level paths outside the project
            # (it honors them only from the global config), hence LOW.
            declarations.extend(
                _Declaration("KC4", Severity.LOW, "external_skill_path", _bounded(path))
                for path in paths
                if isinstance(path, str) and _is_outside_bundle(path)
            )
    instructions = document.get("instructions")
    if isinstance(instructions, list):
        for entry in instructions:
            if not isinstance(entry, str):
                continue
            if _is_remote_reference(entry):
                declarations.append(
                    _Declaration("KC4", Severity.MEDIUM, "remote_instructions", _bounded(entry))
                )
            elif _is_outside_bundle(entry):
                declarations.append(
                    _Declaration("KC4", Severity.MEDIUM, "external_instructions", _bounded(entry))
                )
    return declarations


def _config_declarations(document: object) -> list[_Declaration]:
    if not isinstance(document, dict):
        return []
    declarations: list[_Declaration] = []
    if "permission" in document:
        declarations.extend(_permission_declarations(document["permission"], ""))
    declarations.extend(_agent_permission_declarations(document))
    declarations.extend(_mcp_declarations(document))
    declarations.extend(_plugin_declarations(document))
    declarations.extend(_source_declarations(document))
    if len(declarations) > _MAX_DECLARATIONS:
        raise _DeclarationLimitError(len(declarations))
    return declarations


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

_RULE_TEXT: Final = {
    "KC1": (
        "Bundled Kilo configuration pre-approves tool permissions.",
        "Remove blanket 'allow' grants; leave tools at 'ask' and allow only narrow "
        "command patterns the skill needs.",
        "Permissions",
    ),
    "KC2": (
        "Bundled Kilo configuration declares MCP servers.",
        "Remove bundled MCP servers or document them so users install them explicitly; "
        "never ship local server commands that start with the project.",
        "MCP",
    ),
    "KC3": (
        "Bundled Kilo plugins or Agent Manager scripts run code automatically.",
        "Remove bundled plugins and setup/run scripts, or ship them separately so users "
        "review and install them on purpose.",
        "Plugins",
    ),
    "KC4": (
        "Bundled Kilo configuration loads skills or instructions from outside the bundle.",
        "Ship every skill and instruction inside the bundle; remove remote skill URLs and "
        "paths that escape the bundle directory.",
        "External Sources",
    ),
}


def _line_of_key(content: str, keys: tuple[str, ...]) -> int:
    for key in keys:
        # JSON keys are quoted; YAML frontmatter keys start a line.
        match = re.search(rf'"{re.escape(key)}"\s*:|^{re.escape(key)}:', content, re.MULTILINE)
        if match is not None:
            return content.count("\n", 0, match.start()) + 1
    return 1


_RULE_KEYS: Final = {
    "KC1": ("permission",),
    "KC2": ("mcp",),
    "KC3": ("plugin",),
    "KC4": ("skills", "instructions"),
}


def _finding(path: str, rule_id: str, declarations: list[_Declaration], start_line: int) -> Finding:
    message, remediation, tag = _RULE_TEXT[rule_id]
    severity = max((d.severity for d in declarations), key=_SEVERITY_RANK.__getitem__)
    analyzer_finding = AnalyzerFinding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        location=Location(path, start_line),
        confidence=0.95,
        remediation=remediation,
        tags=["Bundled Kilo Configuration", tag],
        matched_text=f"document:{path}",
        evidence={
            "activation_reason": "requires_kilo_project_activation",
            "activation_state": "conditional",
            "declaration_count": len(declarations),
            "declaration_kinds": sorted({d.kind for d in declarations})[:_MAX_EVIDENCE_ITEMS],
            "subjects": sorted({d.subject for d in declarations})[:_MAX_EVIDENCE_ITEMS],
        },
    )
    return analyzer_finding_to_finding(analyzer_finding)


def _frontmatter_declarations(path: str, content: str) -> list[_Declaration]:
    """Return KC1 declarations from an agent or mode file's YAML frontmatter."""
    if not content.startswith("---"):
        return []
    end = re.search(r"^---[ \t]*$", content[3 : 3 + _MAX_FRONTMATTER_CHARS + 4], re.MULTILINE)
    if end is None:
        if len(content) > _MAX_FRONTMATTER_CHARS:
            raise ValueError("frontmatter exceeds its size limit")
        raise ValueError("unterminated frontmatter")
    frontmatter = yaml.safe_load(content[3 : 3 + end.start()])
    if not isinstance(frontmatter, dict) or "permission" not in frontmatter:
        return []
    name = path.rsplit("/", 1)[-1].removesuffix(".md")
    declarations = _permission_declarations(frontmatter["permission"], f"agent:{name}")
    if len(declarations) > _MAX_DECLARATIONS:
        raise _DeclarationLimitError(len(declarations))
    return declarations


def _analyze_config(path: str, content: str) -> tuple[list[Finding], InspectionLedgerEvent]:
    return _analyze_document(path, content, lambda: _config_declarations(_parse_document(content)))


def _analyze_agent_file(path: str, content: str) -> tuple[list[Finding], InspectionLedgerEvent]:
    return _analyze_document(path, content, lambda: _frontmatter_declarations(path, content))


def _analyze_document(
    path: str, content: str, extract: Callable[[], list[_Declaration]]
) -> tuple[list[Finding], InspectionLedgerEvent]:
    if len(content) > MAX_FILE_CHARS:
        return [], ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            phase="static",
            analyzer_id=ANALYZER_ID,
            path=path,
            reason=LedgerReason.SIZE_LIMIT,
            observed_characters=len(content),
            limit_characters=MAX_FILE_CHARS,
            observed_bytes=len(content.encode("utf-8", errors="replace")),
        )
    try:
        declarations = extract()
    except _DeclarationLimitError as exc:
        return [], ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            phase="static",
            analyzer_id=ANALYZER_ID,
            path=path,
            reason=LedgerReason.OPAQUE_CONTENT,
            observed_records=int(exc.args[0]),
            limit_records=_MAX_DECLARATIONS,
        )
    except (RecursionError, ValueError, yaml.YAMLError):
        return [], ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            phase="static",
            analyzer_id=ANALYZER_ID,
            path=path,
            reason=LedgerReason.OPAQUE_CONTENT,
        )

    by_rule: dict[str, list[_Declaration]] = {}
    for declaration in declarations:
        by_rule.setdefault(declaration.rule_id, []).append(declaration)
    findings = [
        _finding(path, rule_id, by_rule[rule_id], _line_of_key(content, _RULE_KEYS[rule_id]))
        for rule_id in sorted(by_rule)
    ]
    return findings, ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase="static",
        analyzer_id=ANALYZER_ID,
        path=path,
        emitted_finding_ids=[finding.finding_id for finding in findings],
    )


def _analyze_code_file(
    path: str, severity: Severity, kind: str
) -> tuple[list[Finding], InspectionLedgerEvent]:
    finding = _finding(path, "KC3", [_Declaration("KC3", severity, kind, path)], 1)
    return [finding], ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase="static",
        analyzer_id=ANALYZER_ID,
        path=path,
        emitted_finding_ids=[finding.finding_id],
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Inspect bundled Kilo configuration, agent files, plugins, and scripts."""
    components = state.get("components") or []
    file_cache = state.get("local_file_cache") or state.get("file_cache") or {}
    decodable = {
        record.get("path"): record.get("decodable", True)
        for record in (state.get("artifact_inventory") or [])
    }
    findings: list[Finding] = []
    ledger_events: list[InspectionLedgerEvent] = []

    paths = sorted({path for path in components if isinstance(path, str)})
    documents: list[tuple[str, Callable[[str, str], tuple[list[Finding], InspectionLedgerEvent]]]]
    documents = []
    for path in paths:
        script = _AGENT_MANAGER_SCRIPT_PATH.match(path)
        if _PLUGIN_PATH.match(path):
            path_findings, event = _analyze_code_file(path, Severity.HIGH, "bundled_plugin")
        elif script is not None:
            # The setup script runs unprompted; the run script needs a click.
            automatic = script.group("kind") == "setup"
            path_findings, event = _analyze_code_file(
                path,
                Severity.HIGH if automatic else Severity.MEDIUM,
                "setup_script" if automatic else "run_script",
            )
        else:
            if path in _CONFIG_PATHS:
                documents.append((path, _analyze_config))
            elif _AGENT_FILE_PATH.match(path):
                documents.append((path, _analyze_agent_file))
            continue
        findings.extend(path_findings)
        ledger_events.append(event)

    for path, analyze in documents:
        if decodable.get(path) is False:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    phase="static",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.OPAQUE_CONTENT,
                )
            )
            continue
        content = file_cache.get(path)
        if content is None:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.MISSING_FILE_CACHE,
                )
            )
            continue
        try:
            path_findings, event = analyze(path, content)
        except Exception as exc:
            logger.exception("%s failed for %s", ANALYZER_ID, path)
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    error_class=type(exc).__name__,
                )
            )
            continue
        findings.extend(path_findings)
        ledger_events.append(event)

    status = analyzer_status_for_events(ANALYZER_ID, ledger_events)
    return {
        "findings": findings,
        "inspection_ledger": ledger_events,
        "analyzer_status_events": [status],
    }
