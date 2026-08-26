"""?04 — the declared cuga constraint must match the installed reality.

The ledger carried an open discrepancy for a whole session because nothing
mechanical tied ``pyproject.toml`` to the venv: the note said one version was
installed, the pin said another, and only a human reading both could tell.
This test makes drift loud.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _declared_cuga_constraint() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    return next(d for d in deps if d.strip().startswith("cuga"))


def test_installed_cuga_satisfies_the_declared_constraint() -> None:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    constraint = _declared_cuga_constraint()
    spec = SpecifierSet(constraint.split(";", 1)[0].replace("cuga", "").strip())
    installed = Version(importlib.metadata.version("cuga"))
    assert installed in spec, (
        f"installed cuga {installed} does not satisfy declared '{constraint}'; "
        f"the pin and the environment have drifted (?04 class of bug)"
)


def test_pin_has_an_upper_bound() -> None:
    """An open-ended cuga pin let 0.2.20-vs-0.3.x drift go unnoticed once."""
    constraint = _declared_cuga_constraint()
    assert "<" in constraint, (
        "cuga constraint must carry an upper bound so untested upgrades "
        "cannot arrive silently"
    )
