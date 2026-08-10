from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_complete_rho_evolution_archive_is_present() -> None:
    archive = ROOT / "docs" / "rho_evolution"
    expected = {
        "README.md",
        *{f"{index:02d}-{name}.md" for index, name in [
            (1, "overview"),
            (2, "data-model"),
            (3, "control-flow"),
            (4, "coreset-selection"),
            (5, "diagnosis"),
            (6, "candidate-generation"),
            (7, "pairwise-judging"),
            (8, "acceptance-and-promotion"),
            (9, "artifacts-and-versioning"),
            (10, "runner-and-batch-integration"),
            (11, "configuration-reference"),
            (12, "tracing-and-debugging"),
            (13, "rho-gepa-population-evolution"),
            (14, "agent-integration-and-history-rag"),
            (15, "rho-gepa-architecture-and-debugging"),
            (16, "rho-gepa-execution-atlas"),
            (17, "rho-gepa-prompts-and-data-contracts"),
            (18, "rho-parallel-gepa-target-architecture"),
            (19, "rho-parallel-gepa-research-hypotheses"),
        ]},
        "selection_algo_explaination.md",
    }
    assert {path.name for path in archive.glob("*.md")} == expected


def test_reference_baseline_is_complete_and_explicitly_non_importable() -> None:
    reference = ROOT / "reference" / "gaia_evolution_core"
    assert {path.name for path in reference.glob("*.py")} == {
        "__init__.py", "contracts.py", "history.py", "operators.py", "population.py"
    }
    readme = (reference / "README.md").read_text(encoding="utf-8")
    assert "read-only historical baseline" in readme
    assert "must never be imported" in readme


def test_continuation_brief_names_vision_decisions_and_cuga_boundary() -> None:
    brief = (ROOT / "docs" / "vision-and-decision-record.md").read_text(encoding="utf-8")
    for phrase in (
        "evolves externally configurable agent harnesses, not model weights",
        "persistent GEPA pool",
        "base plus every RHO candidate",
        "CUGA SDK",
        "do not invent CUGA APIs",
        "minimal",
    ):
        assert phrase in brief


def test_active_package_has_no_legacy_or_adapter_runtime_imports() -> None:
    forbidden = re.compile(r"^\s*(?:from|import)\s+(?:cuga|agent\.|dataset\.|reference\.|gaia)", re.MULTILINE)
    violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src" / "agent_evolve").rglob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert violations == []


def test_copied_documentation_has_no_broken_relative_markdown_links() -> None:
    broken: list[str] = []
    for path in (ROOT / "docs" / "rho_evolution").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", text):
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert broken == []
