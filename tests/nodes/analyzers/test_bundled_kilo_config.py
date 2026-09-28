# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-behavior tests for bundled Kilo configuration surfaces."""

from __future__ import annotations

import json

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.models import Finding
from skillspector.nodes.analyzers import bundled_kilo_config
from skillspector.nodes.report import _compute_risk_score


def _run(documents: dict[str, object | str]) -> dict:
    cache = {
        path: value if isinstance(value, str) else json.dumps(value)
        for path, value in documents.items()
    }
    return bundled_kilo_config.node({"components": list(cache), "local_file_cache": cache})


def _by_rule(result: dict) -> dict[str, Finding]:
    return {finding.rule_id: finding for finding in result["findings"]}


@pytest.mark.parametrize(
    ("permission", "severity", "subjects"),
    [
        ("allow", "CRITICAL", [""]),
        ({"*": "allow"}, "CRITICAL", ["*"]),
        ({"bash": "allow"}, "HIGH", ["bash"]),
        ({"edit": {"*": "allow"}}, "HIGH", ["edit"]),
        ({"external_directory": "allow"}, "HIGH", ["external_directory"]),
        ({"bash": {"git *": "allow", "rm -rf /": "deny"}}, "MEDIUM", ["bash:git *"]),
        ({"webfetch": "allow"}, "MEDIUM", ["webfetch"]),
    ],
)
def test_kc1_permission_grants(permission, severity, subjects):
    result = _run({"kilo.json": {"permission": permission}})
    finding = _by_rule(result)["KC1"]
    assert finding.severity == severity
    assert finding.evidence["subjects"] == subjects
    assert finding.evidence["activation_state"] == "conditional"


@pytest.mark.parametrize(
    "permission",
    [
        {"bash": "ask", "edit": "deny"},
        {"read": "allow", "grep": "allow", "glob": "allow"},
        {"bash": {"git status": "allow", "*": "ask"}},
        "ask",
    ],
)
def test_kc1_narrow_or_prompting_permissions_are_not_reported(permission):
    result = _run({"kilo.json": {"permission": permission}})
    assert "KC1" not in _by_rule(result)


def test_kc1_agent_and_mode_permissions():
    result = _run(
        {
            ".kilo/kilo.jsonc": {
                "agent": {"helper": {"permission": {"bash": "allow"}}},
                "mode": {"legacy": {"permission": "allow"}},
            }
        }
    )
    finding = _by_rule(result)["KC1"]
    assert finding.severity == "CRITICAL"
    assert finding.evidence["subjects"] == ["agent.helper.bash", "mode.legacy"]


def test_kc2_mcp_servers():
    result = _run(
        {
            "kilo.jsonc": {
                "mcp": {
                    "runner": {"type": "local", "command": ["npx", "-y", "evil-mcp"]},
                    "remote": {"type": "remote", "url": "https://mcp.example.com/sse"},
                    "off": {"type": "local", "command": ["x"], "enabled": False},
                }
            }
        }
    )
    finding = _by_rule(result)["KC2"]
    assert finding.severity == "HIGH"
    assert finding.evidence["subjects"] == ["remote:https://mcp.example.com/sse", "runner:npx"]
    assert finding.evidence["declaration_kinds"] == ["local_server", "remote_server"]


def test_kc2_remote_only_is_medium():
    result = _run({"kilo.json": {"mcp": {"r": {"type": "remote", "url": "https://x.example"}}}})
    assert _by_rule(result)["KC2"].severity == "MEDIUM"


def test_kc3_config_plugins_and_bundled_plugin_modules():
    result = _run(
        {
            "opencode.json": {"plugin": ["kilo-evil-plugin@1.0.0"]},
            ".kilo/plugin/hook.ts": "export default async () => ({})",
            ".kilocode/plugins/legacy.js": "module.exports = {}",
            ".kilo/plugin/README.md": "not a plugin module",
        }
    )
    kc3 = [finding for finding in result["findings"] if finding.rule_id == "KC3"]
    assert sorted(finding.file for finding in kc3) == [
        ".kilo/plugin/hook.ts",
        ".kilocode/plugins/legacy.js",
        "opencode.json",
    ]
    assert {finding.severity for finding in kc3} == {"HIGH"}


def test_kc4_external_skill_and_instruction_sources():
    result = _run(
        {
            "kilo.json": {
                "skills": {
                    "urls": ["https://skills.example.com/.well-known/skills/"],
                    "paths": ["./skills", "~/shared-skills", "../other"],
                },
                "instructions": ["AGENTS.md", "https://example.com/rules.md", "/etc/rules.md"],
            }
        }
    )
    finding = _by_rule(result)["KC4"]
    assert finding.severity == "HIGH"
    assert finding.evidence["subjects"] == [
        "../other",
        "/etc/rules.md",
        "https://example.com/rules.md",
        "https://skills.example.com/.well-known/skills/",
        "~/shared-skills",
    ]


def test_kc4_bundle_local_sources_are_not_reported():
    result = _run(
        {"kilo.json": {"skills": {"paths": ["skills", "./more"]}, "instructions": ["a.md"]}}
    )
    assert result["findings"] == []


def test_jsonc_comments_and_trailing_commas_are_parsed():
    content = """{
      // grant everything
      "permission": {
        "bash": "allow", /* inline */
        "edit": "ask",
      },
      "instructions": ["https://example.com/x // not a comment",],
    }"""
    result = _run({"kilo.jsonc": content})
    rules = _by_rule(result)
    assert rules["KC1"].severity == "HIGH"
    assert rules["KC1"].start_line == 3
    assert rules["KC4"].evidence["subjects"] == ["https://example.com/x // not a comment"]
    assert result["inspection_ledger"][0]["outcome"] == LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "content",
    ['{"permission": "allow", "permission": "ask"}', '{"permission": ', '{"a": "/* x}'],
)
def test_malformed_or_ambiguous_documents_are_partial(content):
    result = _run({"kilo.json": content})
    assert result["findings"] == []
    event = result["inspection_ledger"][0]
    assert event["outcome"] == LedgerOutcome.PARTIAL
    assert event["reason_code"] == LedgerReason.OPAQUE_CONTENT


def test_unrelated_paths_are_ignored():
    result = _run(
        {
            "docs/kilo.json": {"permission": "allow"},
            "skills/kilo.json": {"permission": "allow"},
            ".opencode/opencode.json": {"permission": "allow"},
        }
    )
    assert result["findings"] == []
    assert result["inspection_ledger"] == []


def test_missing_cache_entry_fails_closed():
    result = bundled_kilo_config.node({"components": ["kilo.json"], "local_file_cache": {}})
    assert result["inspection_ledger"][0]["outcome"] == LedgerOutcome.FAILED


def test_critical_kc1_sets_blocking_score_floor():
    result = _run({"kilo.json": {"permission": "allow"}})
    score, _severity, _recommendation = _compute_risk_score(result["findings"], False)
    assert score >= 51
