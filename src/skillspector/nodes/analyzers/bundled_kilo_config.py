# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze bundled Kilo configuration and plugin surfaces.

Kilo (Kilo Code / Kilo CLI) loads project configuration from ``kilo.json``,
``kilo.jsonc``, ``.kilo/kilo.json(c)``, the legacy ``.kilocode/`` directory and
the legacy ``opencode.json(c)`` project files. It also auto-loads plugin
modules from ``.kilo/plugin(s)/``. A skill bundle that ships these files can,
once its directory becomes (or is copied into) a Kilo project, pre-approve
tools, start MCP server processes, load plugin code, or pull further skills
and instructions from outside the reviewed bundle.

Rules:

- KC1: permission grants (``permission`` set to ``allow`` globally, per tool,
  or through a wildcard pattern; also inside ``agent``/``mode`` entries).
- KC2: MCP servers (``local`` servers run a command; ``remote`` servers
  receive tool traffic).
- KC3: plugins (``plugin`` entries and bundled plugin modules).
- KC4: skill and instruction sources outside the bundle (``skills.urls``,
  ``skills.paths`` and ``instructions`` pointing outside the bundle).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Final

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
        # Kilo still reads the OpenCode project file it was forked from.
        "opencode.json",
        "opencode.jsonc",
    }
)
_PLUGIN_PATH: Final = re.compile(r"\A\.kilo(?:code)?/plugins?/[^/]+\.(?:ts|js|mjs|cjs)\Z")

_MAX_DECLARATIONS: Final = 2_048
_MAX_EVIDENCE_ITEMS: Final = 32
_MAX_EVIDENCE_CHARS: Final = 200

# Tools whose blanket approval lets the agent run commands, change files, or
# leave the project without a prompt.
_HIGH_RISK_TOOLS: Final = frozenset(
    {"bash", "edit", "write", "patch", "multiedit", "external_directory"}
)
# Tools whose approval is harmless on its own; they are not reported.
_LOW_RISK_TOOLS: Final = frozenset(
    {"read", "glob", "grep", "list", "lsp", "todoread", "todowrite", "question"}
)
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
    if tool == "*":
        return Severity.CRITICAL
    if tool in _HIGH_RISK_TOOLS:
        return Severity.HIGH
    if tool in _LOW_RISK_TOOLS:
        return None
    # webfetch, websearch, task, skill, MCP tool names, and future tools.
    return Severity.MEDIUM


def _permission_declarations(permission: object, scope: str) -> list[_Declaration]:
    """Return KC1 declarations for one ``permission`` value."""
    if permission == "allow":
        return [_Declaration("KC1", Severity.CRITICAL, "global_allow", scope)]
    if not isinstance(permission, dict):
        return []
    declarations: list[_Declaration] = []
    for tool, value in permission.items():
        severity = _tool_severity(tool)
        if severity is None:
            continue
        subject = f"{scope}.{tool}" if scope else tool
        if value == "allow":
            declarations.append(_Declaration("KC1", severity, "tool_allow", subject))
        elif isinstance(value, dict):
            for pattern, action in value.items():
                if action != "allow":
                    continue
                if pattern in _WILDCARD_PATTERNS:
                    declarations.append(_Declaration("KC1", severity, "tool_allow", subject))
                elif "*" in pattern and severity in (Severity.HIGH, Severity.CRITICAL):
                    declarations.append(
                        _Declaration(
                            "KC1",
                            Severity.MEDIUM,
                            "pattern_allow",
                            f"{subject}:{_bounded(pattern)}",
                        )
                    )
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
            declarations.extend(
                _Declaration("KC4", Severity.MEDIUM, "external_skill_path", _bounded(path))
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
        "Bundled Kilo plugins load code into the agent runtime.",
        "Remove bundled plugins or ship them separately so users review and install them "
        "on purpose.",
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
        match = re.search(rf'"{re.escape(key)}"\s*:', content)
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


def _analyze_config(path: str, content: str) -> tuple[list[Finding], InspectionLedgerEvent]:
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
        declarations = _config_declarations(_parse_document(content))
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
    except (RecursionError, ValueError):
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


def _analyze_plugin_file(path: str) -> tuple[list[Finding], InspectionLedgerEvent]:
    finding = _finding(path, "KC3", [_Declaration("KC3", Severity.HIGH, "bundled_plugin", path)], 1)
    return [finding], ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase="static",
        analyzer_id=ANALYZER_ID,
        path=path,
        emitted_finding_ids=[finding.finding_id],
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Inspect bundled Kilo configuration files and plugin modules."""
    components = state.get("components") or []
    file_cache = state.get("local_file_cache") or state.get("file_cache") or {}
    decodable = {
        record.get("path"): record.get("decodable", True)
        for record in (state.get("artifact_inventory") or [])
    }
    findings: list[Finding] = []
    ledger_events: list[InspectionLedgerEvent] = []

    config_paths = sorted(set(components).intersection(_CONFIG_PATHS))
    plugin_paths = sorted(
        {path for path in components if isinstance(path, str) and _PLUGIN_PATH.match(path)}
    )

    for path in plugin_paths:
        path_findings, event = _analyze_plugin_file(path)
        findings.extend(path_findings)
        ledger_events.append(event)

    for path in config_paths:
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
            path_findings, event = _analyze_config(path, content)
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
