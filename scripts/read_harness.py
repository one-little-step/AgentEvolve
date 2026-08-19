#!/usr/bin/env python3
"""Read an exported harness JSON in human terms.

Why this exists: an exported harness is a flat JSON blob whose surfaces are
*implicit in the key names* -- ``instructions`` is a scalar, while ``skills``,
``policies`` and ``memory`` are name->body maps. Reading one by eye means knowing
that mapping and manually diffing against a parent to see what actually changed.
This prints the surfaces, the provenance, and (with ``--base``) the diff.

Usage::

    uv run python scripts/read_harness.py data/live_harnesses/rho_genetic/champion.json
    uv run python scripts/read_harness.py <file> --base data/live_harnesses/rho_genetic/candidate-base.json
    uv run python scripts/read_harness.py <dir>            # summarise a whole export dir
    uv run python scripts/read_harness.py <dir> --lineage  # evolution overview
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

# Mirrors agent_evolve.adapters.cuga_adapter: _SCALAR_ARTIFACTS / _GROUP_PREFIXES.
# Duplicated deliberately -- this script must stay runnable without importing the
# package (and therefore without CUGA installed).
SCALAR_SURFACES = ("instructions",)
GROUP_SURFACES = ("skills", "policies", "memory")
META_KEYS = ("version", "export_format", "provenance")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def surfaces(doc: dict) -> dict[str, str]:
    """Flatten a harness doc back to ``{artifact_id: content}``.

    The inverse of the export mapping: a scalar key becomes its own artifact id,
    and a group key becomes ``<group>/<member>``.
    """
    out: dict[str, str] = {}
    for key in SCALAR_SURFACES:
        value = doc.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    for group in GROUP_SURFACES:
        members = doc.get(group) or {}
        if isinstance(members, dict):
            for name, body in members.items():
                out[f"{group}/{name}"] = body
    return out


def describe(path: Path, base_path: Path | None, show_full: bool) -> None:
    doc = load(path)
    arts = surfaces(doc)
    prov = doc.get("provenance") or {}

    print("=" * 78)
    print(f"FILE     {path}")
    print(f"version  {doc.get('version')}")
    print(f"format   {doc.get('export_format')}")
    print("=" * 78)

    # ---- provenance -------------------------------------------------------
    print("\nPROVENANCE (where this harness came from, and how well it scored)")
    if not prov:
        print("  (none recorded)")
    else:
        rows = [
            ("candidate_id", "its identity in the pool"),
            ("candidate_version", "the version rollouts ran under"),
            ("source_base_version", "the base it was derived from"),
            ("parent_ids", "immediate parent(s) it was edited from"),
            ("ancestor_ids", "full lineage back to the base"),
            ("origin_attempt_ids", "the attempt(s) that produced it"),
            ("attempt_ids", "attempts recorded against it"),
            ("is_base", "True if this IS the unmodified base"),
            ("is_champion", "True if it won champion selection"),
            ("mean_score", "mean score over scored cells"),
            ("scored_cells", "how many (task, mechanism) cells were measured"),
            ("grader_name", "which grader produced those scores"),
            ("task_ids", "tasks in the run it was measured in"),
        ]
        for key, meaning in rows:
            if key in prov:
                value = prov[key]
                if isinstance(value, list):
                    value = f"[{len(value)}] {', '.join(map(str, value[:6]))}" + (
                        " ..." if len(value) > 6 else ""
                    )
                print(f"  {key:<22} {value}")
                print(f"  {'':<22}   ^ {meaning}")
        if "unexported_artifacts" in prov:
            print("\n  ** unexported_artifacts present **")
            print("     Artifact ids with no CUGA harness slot. They were kept here")
            print("     rather than dropped, but the agent will NOT load them:")
            for key in prov["unexported_artifacts"]:
                print(f"       - {key}")

    # ---- caveat on scoring ------------------------------------------------
    cells = prov.get("scored_cells")
    if isinstance(cells, int) and 0 < cells <= 3:
        print(
            f"\n  CAVEAT: mean_score rests on {cells} scored cell(s). "
            "That is far too few\n          to distinguish signal from the "
            "measured 16.67 pp noise floor."
        )

    # ---- surfaces ---------------------------------------------------------
    print(f"\nSURFACES ({len(arts)} artifact(s))")
    for slot in SCALAR_SURFACES:
        mark = "SET" if slot in arts else "absent"
        size = f"{len(arts[slot])} chars" if slot in arts else ""
        print(f"  {slot:<24} {mark:<8} {size}")
    for group in GROUP_SURFACES:
        members = doc.get(group) or {}
        if members:
            print(f"  {group:<24} {len(members)} member(s)")
            for name, body in members.items():
                print(f"      {group}/{name:<18} {len(body)} chars")
        else:
            print(f"  {group:<24} absent")

    created = [a for a in arts if a.startswith("skills/generated-")]
    print(
        f"\n  created-by-evolution artifacts: "
        f"{', '.join(created) if created else 'NONE (only pre-existing surfaces edited)'}"
    )

    # ---- bodies / diff ----------------------------------------------------
    if base_path is not None:
        base_arts = surfaces(load(base_path))
        print(f"\nDIFF vs {base_path.name}")
        all_ids = sorted(set(arts) | set(base_arts))
        for artifact_id in all_ids:
            before = base_arts.get(artifact_id, "")
            after = arts.get(artifact_id, "")
            if before == after:
                print(f"\n  [{artifact_id}] UNCHANGED")
                continue
            tag = "CREATED" if not before else ("DELETED" if not after else "EDITED")
            print(f"\n  [{artifact_id}] {tag}  {len(before)} -> {len(after)} chars")
            for line in difflib.unified_diff(
                before.splitlines(), after.splitlines(),
                fromfile="base", tofile="candidate", lineterm="", n=1,
            ):
                if not line.startswith(("---", "+++")):
                    print(f"    {line}")
    elif show_full:
        for artifact_id, body in arts.items():
            print(f"\n--- {artifact_id} " + "-" * (70 - len(artifact_id)))
            print(body)


def lineage(directory: Path) -> None:
    """Summarise every harness in an export directory, base first."""
    files = sorted(directory.glob("*.json"))
    if not files:
        print(f"no harness json in {directory}")
        return
    base_text = ""
    for path in files:
        doc = load(path)
        if (doc.get("provenance") or {}).get("is_base"):
            base_text = doc.get("instructions") or ""
    print(f"{'FILE':<42}{'chars':>7}{'vs base':>9}{'score':>7}{'cells':>7}  flags")
    print("-" * 88)
    for path in files:
        doc = load(path)
        prov = doc.get("provenance") or {}
        arts = surfaces(doc)
        text = doc.get("instructions") or ""
        flags = []
        if prov.get("is_base"):
            flags.append("BASE")
        if prov.get("is_champion"):
            flags.append("CHAMPION")
        created = [a for a in arts if a.startswith("skills/generated-")]
        if created:
            flags.append(f"+{len(created)} new skill")
        score = prov.get("mean_score")
        print(
            f"{path.name:<42}{len(text):>7}{len(text) - len(base_text):>+9}"
            f"{(f'{score:.2f}' if isinstance(score, (int, float)) else '-'):>7}"
            f"{str(prov.get('scored_cells', '-')):>7}  {' '.join(flags)}"
        )
    print(
        "\nnote: 'vs base' is instructions length only. A same-length rewrite "
        "shows +0;\n      use --base for the real diff."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read an exported harness JSON in human terms."
    )
    parser.add_argument("path", type=Path, help="harness json file, or an export directory")
    parser.add_argument("--base", type=Path, default=None,
                        help="diff against this harness (usually candidate-base.json)")
    parser.add_argument("--full", action="store_true",
                        help="print every artifact body in full")
    parser.add_argument("--lineage", action="store_true",
                        help="directory mode: one summary row per harness")
    args = parser.parse_args()

    if args.path.is_dir():
        if args.lineage or args.base is None:
            lineage(args.path)
            return 0
        for path in sorted(args.path.glob("*.json")):
            describe(path, args.base, args.full)
        return 0
    describe(args.path, args.base, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
