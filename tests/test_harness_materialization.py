from __future__ import annotations

import yaml

import types

from agent_evolve.cuga_wrapper import CugaSdkRuntime, _construct_agent, materialize_harness


def test_materialize_harness_writes_skill_with_frontmatter(tmp_path):
    ws = tmp_path / "ws"

    result = materialize_harness({"skills": {"retrieval": "Use the catalog."}}, ws)

    skill_file = ws / "skills" / "retrieval" / "SKILL.md"
    assert result == str(ws)
    text = skill_file.read_text(encoding="utf-8")
    assert "name: retrieval" in text
    # Quoted scalar: an unquoted description starting with "#" is a YAML
    # comment, and one containing ":" is invalid YAML. Either drops the
    # skill silently while the run still succeeds.
    assert 'description: "Use the catalog."' in text
    assert yaml.safe_load(text.split("---")[1])["description"] == "Use the catalog."
    assert "Use the catalog." in text


def test_materialize_harness_writes_policy_playbook(tmp_path):
    ws = tmp_path / "ws"

    materialize_harness({"policies": {"concise": "Answer in one sentence."}}, ws)

    policy_file = ws / "playbooks" / "concise.md"
    text = policy_file.read_text(encoding="utf-8")
    assert "name: concise" in text
    assert "id: playbook_concise" in text
    assert "always: true" in text
    assert "Answer in one sentence." in text


def test_materialize_harness_writes_memory_doc(tmp_path):
    ws = tmp_path / "ws"

    materialize_harness({"memory": {"city": "Paris"}}, ws)

    assert "Paris" in (ws / "memory" / "city.md").read_text(encoding="utf-8")


def test_materialize_harness_returns_none_when_no_editable_artifacts(tmp_path):
    assert materialize_harness({"input": "hello"}, tmp_path / "ws") is None


def test_materialize_harness_sanitizes_names(tmp_path):
    ws = tmp_path / "ws"

    materialize_harness(
        {"skills": {"a/b..c": "body"}, "policies": {"x/y": "content"}, "memory": {"k/v": "v"}},
        ws,
    )

    assert (ws / "skills" / "a_b_c" / "SKILL.md").exists()
    assert (ws / "playbooks" / "x_y.md").exists()
    assert (ws / "memory" / "k_v.md").exists()


def test_construct_agent_passes_skills_and_policies_surfaces(monkeypatch, tmp_path):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    cuga = types.ModuleType("cuga")
    cuga.CugaAgent = FakeAgent
    monkeypatch.setitem(__import__("sys").modules, "cuga", cuga)

    ws = tmp_path / "ws"
    _construct_agent({"skills": {"s": "b"}, "policies": {"p": "c"}}, ["t"], "d", str(ws))

    assert captured["enable_skills"] is True
    assert captured["skills_folder"] == str(ws)
    assert captured["cuga_folder"] == str(ws)
    assert captured["auto_load_policies"] is True


def test_construct_agent_disables_skills_and_policies_when_absent(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    cuga = types.ModuleType("cuga")
    cuga.CugaAgent = FakeAgent
    monkeypatch.setitem(__import__("sys").modules, "cuga", cuga)

    _construct_agent({}, ["t"], "d")

    assert captured["enable_skills"] is False
    assert captured["skills_folder"] is None
    assert captured["cuga_folder"] is None
    assert captured["auto_load_policies"] is False


def test_cuga_sdk_runtime_ingests_memory_before_invoke(tmp_path):
    ingested = []

    class FakeResult:
        answer = "ok"
        error = None
        tool_calls = []

    class FakeKnowledge:
        async def ingest(self, file_path):
            ingested.append(file_path)

    class FakeAgent:
        knowledge = FakeKnowledge()

        async def invoke(self, message, *, track_tool_calls):
            return FakeResult()

        async def aclose(self):
            return None

    runtime = CugaSdkRuntime(
        agent_factory=lambda config, workspace_dir=None: FakeAgent(),
        workspace_root=tmp_path,
    )

    runtime.run_task("task-1", {"memory": {"city": "Paris"}, "input": "where"})

    assert len(ingested) == 1
    assert ingested[0].endswith("memory/city.md")


def test_skill_description_survives_markdown_heading_and_colon(tmp_path) -> None:
    """Derived descriptions must load as YAML regardless of body text.

    A body starting with "# Heading" produced `description: None` (unquoted "#"
    opens a YAML comment) and CUGA's loader then rejects the skill for a missing
    description -- the file exists on disk but never reaches the model.
    """
    from agent_evolve.cuga_wrapper import materialize_harness

    ws = tmp_path / "ws"
    materialize_harness(
        {
            "skills": {
                "heading": "# Refining an artifact\n\nBody.\n",
                "colon": "Rule: always verify before reporting.\n\nBody.\n",
            }
        },
        ws,
    )
    for name in ("heading", "colon"):
        text = (ws / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        meta = yaml.safe_load(text.split("---")[1])
        assert meta["name"] == name
        assert meta["description"], f"{name} lost its description"
        assert "#" not in str(meta["description"])[:1]
