"""Diagnose which directory CUGA actually scans for skills and policies.

Hypothesis: skill discovery derives its root from ``cuga_folder``, NOT from
``skills_folder``. If true, a candidate carrying ONLY skills (no policies)
gets ``cuga_folder=None`` from ``_construct_agent`` and therefore silently
falls back to ``<cwd>/.cuga/skills`` -- the stale project directory -- instead
of the candidate workspace.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

WS = Path("data/workspaces/e2e-candidate-A").resolve()


def main() -> None:
    from cuga.backend.skills.loader import discover_skills, get_skill_root

    print("cwd:", os.getcwd())
    print("candidate workspace:", WS)
    print("SKILLS_ROOT env:", os.environ.get("SKILLS_ROOT"))

    print("\n=== skill root resolution ===")
    root_none = get_skill_root(None)
    root_ws = get_skill_root(str(WS))
    print("cuga_folder=None       ->", root_none)
    print("cuga_folder=<workspace> ->", root_ws)

    print("\n=== what each root actually yields ===")
    for label, cuga_folder in (("cuga_folder=None", None), ("cuga_folder=workspace", str(WS))):
        entries = discover_skills(cuga_folder)
        print(f"{label}: {len(entries)} skill(s)")
        for entry in entries:
            print(f"    name={entry.name!r} source={entry.source}")

    print("\n=== stale project state that None falls back to ===")
    stale_skills = Path(os.getcwd()) / ".cuga" / "skills"
    stale_playbooks = Path(os.getcwd()) / ".cuga" / "playbooks"
    for path in (stale_skills, stale_playbooks):
        if path.is_dir():
            found = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]
            print(f"  {path}: {found}")

    print("\n=== VERDICT ===")
    leaks_to_stale = root_none == stale_skills
    workspace_honored = root_ws == WS / "skills"
    print("cuga_folder=None resolves to stale project .cuga/skills:", leaks_to_stale)
    print("cuga_folder=workspace resolves to candidate skills:", workspace_honored)
    print(
        "CONFIRMED BUG:" if (leaks_to_stale and workspace_honored) else "hypothesis not confirmed:",
        "skill discovery keys off cuga_folder, so a skills-only candidate "
        "silently loads the stale global directory",
    )


if __name__ == "__main__":
    main()
