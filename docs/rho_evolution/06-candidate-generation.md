# 06 - Candidate Generation

Candidate generation turns diagnoses into proposed wisdom edits. Each candidate
starts as a copy of the parent bundle and is modified through a gated edit
registry.

## Location

- Prompt: agent/gaia_lg_react/evolution/prompts.py, optimize_candidate
- Editor: agent/gaia_lg_react/evolution/edit_tools.py

## Flow

1. Materialize parent bundle into candidate directory.
2. Create WisdomEditRegistry with enabled=True.
3. Call optimize_candidate with diagnoses, parent bundle, and registry.
4. The LLM proposes a JSON object containing an edits array.
5. Each edit is applied through the registry.
6. Registry closes and flushes edit_log.jsonl.
7. The resulting bundle is loaded and validated.

## Allowed edit operations

- append_section(filename, heading, content)
- replace_section(filename, heading, content)
- delete_section(filename, heading)

Only these files may be edited:

- intent_planner.md
- reAct.md
- critic.md
- consolidator.md
- scratchpad.md
- synthesis.md

## Safety gates

- The registry is created with enabled=True only inside evolution.
- Normal agent runs never enable editing.
- Absolute paths, path traversal, symlinks, and unknown filenames are rejected.
- Append fails if the heading already exists.
- Replace and delete fail if the heading does not exist.
- Each edit is recorded as a unified diff in edit_log.jsonl.

## Optimizer prompt

The optimizer receives:

- Parent wisdom bundle summary
- JSON array of diagnoses
- Tool description with allowed operations

It must return:

```json
{
  "edits": [
    {
      "operation": "append_section",
      "filename": "synthesis.md",
      "heading": "Answer richness check",
      "content": "Before finalizing, verify the answer has more than one token."
    }
  ]
}
```

## Failure handling

If the optimizer returns invalid JSON, unknown operations, or edits that all
fail, the candidate falls back to the unchanged parent bundle. The raw output
and parse error are preserved.

## Candidate directory layout

```text
dataset/runs/evolution/<round_id>/candidates/candidate_0/
  intent_planner.md
  reAct.md
  synthesis.md
  edit_log.jsonl
  rollouts/
    <task_id>.json
```
