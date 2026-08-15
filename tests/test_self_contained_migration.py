from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]

REQUIRED_TRACKED_PATHS = {
    ".gitignore",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "AGENTS.md",
    "docs/START_HERE.md",
    "docs/architecture/target-rho-parallel-gepa.md",
    "docs/research/hypotheses-and-validation.md",
    "docs/migration/gaia-baseline-and-gap-audit.md",
    "docs/migration/cuga-adaptation-guide.md",
    "docs/migration/cuga-sdk-integration-notes.md",
    "docs/migration/self-contained-migration-inventory.md",
    "docs/plans/rho-parallel-gepa-completion.md",
    "docs/vision-and-decision-record.md",
    "docs/rho_evolution/README.md",
    "docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md",
    "docs/rho_evolution/19-rho-parallel-gepa-research-hypotheses.md",
    "reference/gaia_evolution_core/README.md",
    "tests/test_contracts.py",
}

REQUIRED_ACTIVE_PACKAGE_PATHS = {
    "src/agent_evolve/__init__.py",
    "src/agent_evolve/core/__init__.py",
    "src/agent_evolve/core/contracts.py",
    "src/agent_evolve/adapters/__init__.py",
    "src/agent_evolve/adapters/base.py",
}


def test_active_package_configuration_and_onboarding_are_tracked() -> None:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    required = REQUIRED_TRACKED_PATHS | REQUIRED_ACTIVE_PACKAGE_PATHS
    assert required <= tracked, sorted(required - tracked)

    package = ROOT / "src" / "agent_evolve"
    assert package.is_dir()
    assert {path.relative_to(ROOT).as_posix() for path in package.rglob("*.py")} >= REQUIRED_ACTIVE_PACKAGE_PATHS


def test_repository_ignores_macos_finder_metadata() -> None:
    assert ".DS_Store" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_historical_archive_quick_start_is_explicitly_non_runnable() -> None:
    archive_readme = (ROOT / "docs" / "rho_evolution" / "README.md").read_text(encoding="utf-8")
    assert "not runnable in AgentEvolve" in archive_readme
    assert "unavailable in AgentEvolve" in archive_readme
    assert "../START_HERE.md" in archive_readme
    assert "CUGA-neutral" in archive_readme


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
    """No legacy Gaia/dataset imports, and CUGA only inside adapter boundaries.

    ``core/`` remains agent-neutral: it must never import CUGA. ``adapters/``
    is the CUGA boundary by design -- the editor (and later the judge and seed
    generator) are CUGA-backed agents, so an adapter module importing the SDK
    is the intended architecture, not a violation. Those imports stay deferred
    inside functions, which ``test_editor_offline_decoupling`` proves by
    importing every editing module with the SDK blocked.
    """
    forbidden = re.compile(r"^\s*(?:from|import)\s+(?:cuga|agent\.|dataset\.|reference\.|gaia)", re.MULTILINE)
    violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src" / "agent_evolve").rglob("*.py")
        if "cuga_wrapper" not in path.parts
        if "adapters" not in path.parts
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert violations == []


def test_core_never_imports_cuga_or_adapters() -> None:
    """The agent-neutral boundary that actually matters (AGENTS.md).

    Narrowing the scan above to exclude ``adapters/`` would be a silent
    weakening if nothing still guarded ``core/``, so this asserts the real
    invariant directly: no CUGA, and no adapter module, reachable from core.
    """
    forbidden = re.compile(
        r"^\s*(?:from|import)\s+(?:cuga|agent_evolve\.adapters)", re.MULTILINE
    )
    violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src" / "agent_evolve" / "core").rglob("*.py")
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
