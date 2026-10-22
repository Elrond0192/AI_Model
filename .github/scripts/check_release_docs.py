#!/usr/bin/env python3
"""Release documentation contract and pull-request freshness gate.

Without --base/--head, validates repository documentation consistency locally.
With --base/--head, also requires relevant docs/changelog changes when runtime,
API/UI, database or dependency files change in a pull request.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    target = ROOT / path
    if not target.is_file():
        raise AssertionError(f"Required documentation file is missing: {path}")
    return target.read_text(encoding="utf-8")


def changed_files(base: str, head: str) -> set[str]:
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=ROOT,
        text=True,
    )
    return {line.strip() for line in output.splitlines() if line.strip()}


def check_current_contract() -> list[str]:
    errors: list[str] = []
    scheduler = read("docs/SCHEDULER.md")
    readme = read("README.md")
    runbook = read("RUNBOOK.md")
    changelog = read("CHANGELOG.md")
    contributor = read("CONTRIBUTING.md")
    causal = read("docs/CAUSAL_PLAYER_ATTRIBUTION_ROADMAP.md")
    player_intelligence = read("docs/PLAYER_INTELLIGENCE.md")
    simulation_guide = read("docs/SIMULATION_ENGINE.md")
    roadmap = read("roadmap_intelligence.md")
    sql_contract = read("basketball_ai/data/ai_scenario_serving.sql")
    ui = read("basketball_ai/admin_web/static/index.html")

    required_scheduler_terms = {
        "interval mode": "Every X hours",
        "daily wall-clock mode": "Daily at a time",
        "Rome timezone": "Europe/Rome",
        "inclusive start/end bounds": "Both boundaries are inclusive",
        "activation switch": "enable/disable",
        "seasonal refresh task": "refresh_context_seasonal",
        "model training boundary": "does not imply",
        "causal refresh boundary": "not keys in the recurring scheduler",
    }
    for label, phrase in required_scheduler_terms.items():
        if phrase.lower() not in scheduler.lower():
            errors.append(f"docs/SCHEDULER.md is missing {label}: {phrase!r}")

    for path, content in (("README.md", readme), ("RUNBOOK.md", runbook)):
        if "docs/SCHEDULER.md" not in content:
            errors.append(f"{path} must link to docs/SCHEDULER.md")

    for label, content in (
        ("CONTRIBUTING.md", contributor),
        ("CHANGELOG.md", changelog),
    ):
        if "documentation" not in content.lower() and "docs" not in content.lower():
            errors.append(f"{label} must state the documentation release policy/update")

    causal_required = [
        "bulk-refresh",
        "Model Releases",
        "not a chat side effect",
        "remains partial",
        "out-of-time",
    ]
    for phrase in causal_required:
        if phrase.lower() not in causal.lower():
            errors.append(
                "docs/CAUSAL_PLAYER_ATTRIBUTION_ROADMAP.md is missing current status "
                f"statement: {phrase!r}"
            )

    stale_phrases = [
        "not yet persisted or refresh-audited",
        "the panel is currently built in memory and is not yet persisted",
        "SimulationCausalPanel\u0060 è solo schema e rimane vuota",
        "SimulationCausalPanel\u0060 deve restare vuota",
        "Teammate-specific causal attribution remains unavailable until a defensible "
        "treatment/exposure and outcome design is implemented",
    ]
    for phrase in stale_phrases:
        if phrase.lower() in causal.lower() or phrase.lower() in roadmap.lower():
            errors.append(f"Obsolete capability statement remains in roadmap: {phrase}")

    if "Keep both capabilities `partial`" in player_intelligence:
        errors.append(
            "docs/PLAYER_INTELLIGENCE.md must distinguish descriptive availability "
            "from the partial causal capability"
        )
    if "SimulationCausalPanel is refreshed separately by the explicit guarded Model Releases cohort-refresh action" not in sql_contract:
        errors.append(
            "SQL diagnostics must state that causal panels are refreshed separately "
            "from the Simulation serving refresh"
        )
    if "manual" not in simulation_guide.lower() or "does not run from chat" not in simulation_guide:
        errors.append(
            "docs/SIMULATION_ENGINE.md must distinguish manual causal-panel refresh "
            "from the recurring Simulation refresh"
        )

    if "Operazioni dati della PR #76" in ui:
        errors.append("Remove the obsolete PR #76 implementation-label copy from the UI")

    if "Recurring scheduler controls" not in changelog and "Persistent scheduler controls" not in changelog:
        errors.append("CHANGELOG.md must include the scheduler release entry")
    if "causal-panel" not in changelog.lower():
        errors.append("CHANGELOG.md must include the causal-panel refresh release entry")
    if "same release change" not in contributor.lower():
        errors.append("CONTRIBUTING.md must require docs/changelog updates in the release change")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Base commit SHA for PR documentation diff check")
    parser.add_argument("--head", help="Head commit SHA for PR documentation diff check")
    args = parser.parse_args()

    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be supplied together")

    errors = check_current_contract()
    if args.base and args.head:
        files = changed_files(args.base, args.head)
        runtime_prefixes = ("basketball_ai/",)
        runtime_files = {
            path for path in files
            if path.startswith(runtime_prefixes)
            or path in {"main.py", "Dockerfile", "docker-compose.yml", "pyproject.toml", "requirements.txt"}
        }
        doc_files = {
            "README.md", "RUNBOOK.md", "CHANGELOG.md", "CONTRIBUTING.md",
            "roadmap_intelligence.md",
        }
        has_documentation_update = any(
            path in doc_files or path.startswith("docs/") or path.startswith("roadmap")
            for path in files
        )
        # Tests alone do not represent a product/runtime contract change.
        has_runtime_change = any(
            not path.startswith("basketball_ai/__pycache__/")
            and not path.startswith("basketball_ai/tests/")
            for path in runtime_files
        )
        if has_runtime_change and not has_documentation_update:
            errors.append(
                "Runtime/API/UI/database changes require a relevant documentation "
                "or CHANGELOG.md update in the same pull request."
            )

    if errors:
        print("Documentation contract check failed:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("Documentation contract check passed.")
    if args.base and args.head:
        print("Pull-request documentation freshness gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
