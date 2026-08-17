Yes. I can identify the complete RHO prompt surface and point you to the exact public source locations. I cannot reproduce every lengthy prompt verbatim here, but I can give you a precise inventory, faithful functional reconstructions, and a command that extracts the exact strings from the authors’ MIT-licensed repository.

The paper is**“Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference,” arXiv:2606.05922**. Its public implementation is wbopan/retro-harness, and the repository states that prompts, completions, trajectories, diagnoses, candidate harnesses, and scores are persisted for each run.

1. Important terminology

Not every instruction used by RHO is technically an API-level system message.

The implementation has three categories:

Agent instructions
 Instructions passed to Codex for solving, diagnosing, editing, or evaluating.

LLM judge prompts
 Prompts sent to a separate LLM client for difficulty and fingerprint prediction.

Task and workspace context
 Files such as task/prompt.md, diagnoses/.../diagnosis.md, trajectory directories, and harness files.

So, “all RHO system prompts” should normally include the following core prompt families:

P1. Task difficulty and abstract-fingerprint judge
P2. Ordinary task-solving agent
P3. Group-trajectory diagnosis agent
P4. Harness-optimization agent
P5. Pairwise self-preference evaluator


These correspond to the paper’s task selection, repeated solving, diagnosis, candidate proposal, and candidate evaluation stages.

2. Prompt P1: Difficulty and abstract-fingerprint judge
Exact source
src/rho/selection/difficulty_selector.py


The prompt is stored in:

_JUDGE_INSTRUCTIONS


The implementation asks the judge to produce:

{
  "difficulty": 0.0,
  "abstract_fingerprint": "..."
}


The difficulty must be between 0.0 and 10.0. The fingerprint is intended to describe the abstract structural difficulty and failure pattern rather than merely repeat task-specific names.

Faithful functional reconstruction
Role:
You are a task-difficulty and structural-pattern judge.

Input:
- The original task
- A digest of an observed agent trajectory, when available

Responsibilities:
1. Estimate task difficulty on a 0 to 10 scale.
2. Infer the underlying structural challenge.
3. Produce an abstract fingerprint describing the task’s reasoning shape,
   dependencies, failure mode, and required coordination.
4. Avoid superficial identifiers and project-specific names.
5. Return only a JSON object.

Required output:
{
  "difficulty": <number from 0.0 to 10.0>,
  "abstract_fingerprint": "<abstract structural description>"
}


The visible source begins by asking the model to rate a software-engineering task and write an abstract structural fingerprint, optionally using the observed run. It explicitly requests raw JSON without Markdown fences.

Critical observation

This prompt appears partly software-engineering specific. If AgentEvolve applies RHO to GAIA, CUGA, or other non-code environments, I recommend moving the domain-specific rubric into the adapter:

Generic core:
  difficulty + structural fingerprint schema

Adapter:
  domain-specific meaning of difficulty
  domain-specific examples
  prohibited identifiers


Otherwise, a software-oriented rubric may distort selection in knowledge-work or multi-agent environments.

3. Prompt P2: Ordinary task-solving agent
Exact source
src/rho/orchestrators/solve.py


The solve prompt tells the agent to inspect harness/, inspect the task request, complete the task, modify the task repository directly for repair tasks, and place the requested answer in its final message.

Faithful functional reconstruction
Role:
You are the task-solving agent operating with a supplied harness.

Workspace:
- harness/ contains the available instructions, skills, workflows, and tools.
- task/ contains the task specification.
- task/prompt.md contains the user’s request.
- task/repo/ may contain a repository that must be modified.

Procedure:
1. Familiarize yourself with the harness resources and available tools.
2. Read task/prompt.md carefully.
3. Complete the requested task.
4. For code-repair tasks, edit files under task/repo/.
5. In the final message, follow the output format requested by prompt.md.
6. If no special output format is specified, answer in normal prose.


This is the same solving primitive used for:

Baseline group rollouts
Candidate-harness rollouts
Held-out evaluation solves

The difference is the harness snapshot mounted into harness/, not a substantially different solver prompt. The implementation materializes isolated temporary solve workspaces for these calls.

4. Prompt P3: Group-trajectory diagnosis agent
Exact source
src/rho/orchestrators/diagnose.py


The main prompt components include:

_DIAGNOSE_PREAMBLE


and variants controlled by whether self-validation and self-consistency are enabled.

The source begins by instructing the agent to analyze three solve trajectories for the same task. The workspace includes the original task, harness, and trajectory material.

Faithful functional reconstruction
Role:
You are an offline trajectory analyst.

Input:
- One original task
- The harness used for solving
- Three independent solve trajectories for that task
- Each trajectory’s event history
- Final answer
- Workspace changes or diff

Responsibilities:
1. Read the original task and determine its exact requirements.
2. Analyze each trajectory independently.
3. Judge whether each trajectory likely completed the task successfully.
4. Identify incorrect assumptions, tool mistakes, missing evidence,
   incomplete work, premature stopping, and wasted steps.
5. Compare all trajectories.
6. Identify meaningful disagreements in interpretation, plans, actions,
   edited files, verification, and final answers.
7. Distinguish harmless variation from consequential inconsistency.
8. Infer recurring failure modes.
9. Estimate the severity of the observed harness weakness.
10. Propose a general harness-improvement direction.

Do not:
- Edit the harness.
- Treat one trajectory as ground truth.
- Assume that agreement means correctness.
- Include task-specific expected answers as reusable instructions.

Return structured output describing:
- Per-trajectory assessment
- Likely success or failure
- Quality analysis
- Issues
- Failure-mode analysis
- Inconsistency analysis
- Severity
- Harness-improvement direction


The diagnosis output is later serialized into sections such as:

Failure mode analysis
Inconsistency analysis


and passed to the optimizer. The source supports ablations that disable consistency analysis, validation analysis, or both.

Likely structured fields

From the implementation and tests, the diagnosis object includes concepts equivalent to:

{
  "severity": 0.0,
  "trajectory_analyses": [
    {
      "trajectory": "...",
      "successful": 0,
      "quality_analysis": "...",
      "issues": "..."
    }
  ],
  "failure_mode_analysis": "...",
  "inconsistency_analysis": "...",
  "harness_improvement_direction": "..."
}


The precise active fields can depend on the validation and consistency feature switches.

Advice for AgentEvolve

Do not combine this RHO diagnosis prompt directly with your proposed causal-blame prompt.

Use two stages:

Stage A: observational diagnosis
  What happened?
  What differed?
  What failed?
  What evidence supports that finding?

Stage B: attribution hypothesis
  Which agent, module, or artifact may have contributed?
  What intervention would test that hypothesis?


That prevents an observational diagnosis from being prematurely represented as causal evidence.

5. Prompt P4: Harness-optimization agent
Exact source
src/rho/strategies/diagnose.py


The relevant constants include:

_OPTIMIZE_PREAMBLE
OPTIMIZE_INSTRUCTIONS


The source instructs the agent to analyze per-task diagnoses under diagnoses/ and modify the current harness/ to improve future performance. It defines improved performance in terms of answering tasks more directly and correctly with fewer wasted steps.

Faithful functional reconstruction
Role:
You are an offline harness optimizer.

Workspace:
- harness/ is the current writable harness.
- diagnoses/task_XXXX/diagnosis.md contains one task diagnosis.
- diagnoses/task_XXXX/prompt.md contains the corresponding original task.

Objective:
Improve the harness for future tasks, not merely for the supplied examples.

Procedure:
1. Read all diagnosis files and their corresponding task prompts.
2. Use diagnosis severity as a prioritization signal, not as ground truth.
3. Look for recurring patterns across tasks.
4. Prioritize repeated, high-severity, generalizable weaknesses.
5. Do not make an edit solely because of one isolated low-severity problem,
   unless the same issue motif appears elsewhere.
6. Inspect the current harness before editing.
7. Make surgical improvements.
8. Preserve useful existing behavior.
9. Add or modify instructions, skills, workflows, or tools when justified.
10. Avoid hardcoding task-specific answers, names, repositories, paths,
    expected outputs, or evaluator details.
11. Leave the harness unchanged if the diagnoses do not justify an improvement.

Output mechanism:
Modify the files under harness/.
The resulting filesystem is the candidate harness.


The important part is that the optimizer’s final chat response is not the canonical harness. The candidate is recovered from the modified harness/ filesystem. Each optimizer sample runs inside a fresh workspace, ensuring independently generated candidates.

Prompt variants

The diagnosis strategy supports variants corresponding to the paper’s ablations:

Full:
  self-validation + self-consistency

No consistency:
  self-validation only

No validation:
  self-consistency only

Raw trajectories:
  reduced or absent structured diagnosis


The implementation exposes validation and consistency switches, and the strategy constructs the optimizer context accordingly.

6. Prompt P5: Pairwise self-preference evaluator
Exact source
src/rho/orchestrators/evaluate.py


This stage compares a transition from the old harness to a candidate harness. The evaluator receives baseline and candidate evidence and produces a structured score. The evaluation trajectory is annotated as an evaluate stage, and the parser accepts structured JSON output.

Faithful functional reconstruction
Role:
You are a pairwise trajectory evaluator.

Input:
- The original task
- The before or baseline harness
- The after or candidate harness
- A baseline trajectory produced with the original harness
- A candidate trajectory produced with the candidate harness
- Their final answers, event histories, and relevant workspace changes

Objective:
Determine whether the candidate-harness trajectory is better than the
baseline-harness trajectory for the original task.

Evaluation criteria:
1. Correctness relative to the original request
2. Completeness
3. Compliance with exact task requirements
4. Quality and relevance of the final answer
5. Evidence and verification
6. Efficiency and avoidance of wasted steps
7. Whether the candidate introduced regressions
8. Whether either trajectory stopped prematurely

Return:
- A signed preference score
- Positive if the candidate is better
- Zero if equivalent or indeterminate
- Negative if the candidate is worse
- Structured JSON only


The paper describes the result as a preference score in:

[−10,10][-10,10]

Candidate scores are averaged across the coreset, and a candidate is accepted only when the best average is strictly positive. The public repository confirms that candidate selection uses the agent’s own pairwise preference and that no external labels are used.

Important implementation detail

In your AgentEvolve implementation, preserve these properties:

Same baseline trajectory for all candidates on a task
Blind candidate identifiers
Recorded candidate/baseline presentation order
Consistent sign convention after parsing
Zero or invalid evaluation stored separately from evaluator failure
No expected answer or grader output in the judge context


An evaluator timeout should not be indistinguishable from a genuine tie.

7. Shared harness-description prompt component

The source imports a shared constant:

HARNESS_DESCRIPTION


from:

src/rho/orchestrators/_util.py


This description explains the workspace-level meaning of harness/ to the solve, diagnose, and optimization agents.

Conceptually, it tells an agent:

harness/ contains persistent resources available to the task-solving agent,
including instructions, skills, workflows, and executable tools.

These resources affect future solves and should be interpreted as the
agent’s operating harness.


For AgentEvolve, this shared section should be generated from the adapter’s ArtifactDescriptor inventory rather than hardcoded around files:

Artifact ID
Artifact kind
Read permission
Write permission
Format
Phase binding
Atomic group
Merge policy

8. Prompt-to-pipeline map
Pipeline stage	Prompt	Primary sourceHistorical-task scoring	Difficulty and fingerprint judge	selection/difficulty_selector.py
Baseline group rollout	Solve instructions	orchestrators/solve.py
Candidate rollout	Same solve instructions, different harness	orchestrators/solve.py
Self-validation	Diagnosis prompt	orchestrators/diagnose.py
Self-consistency	Diagnosis prompt	orchestrators/diagnose.py
Candidate generation	Optimization instructions	strategies/diagnose.py
Pairwise ranking	Evaluation instructions	orchestrators/evaluate.py
Shared harness semantics	Harness description	orchestrators/_util.py

The public repository organizes the implementation into selection, strategies, orchestrators, agents, datasets, stores, and the outer RHO loop.

9. How to extract the exact prompt strings locally

Because the repository is public and MIT-licensed, the cleanest way to obtain the exact current text is to clone the authors’ repository and inspect the constants directly. The repository identifies itself as the public implementation of the RHO paper and includes an MIT license.

git clone https://github.com/wbopan/retro-harness.git
cd retro-harness

grep -RInE \
  '_[A-Z_]*(PROMPT|INSTRUCTIONS|PREAMBLE)|SYSTEM_PROMPT|system_prompt|You are|Return only' \
  src/rho codex .claude


For the five core prompts:

sed -n '1,220p' src/rho/selection/difficulty_selector.py
sed -n '1,380p' src/rho/orchestrators/diagnose.py
sed -n '1,180p' src/rho/orchestrators/solve.py
sed -n '1,280p' src/rho/strategies/diagnose.py
sed -n '1,180p' src/rho/orchestrators/evaluate.py
sed -n '1,120p' src/rho/orchestrators/_util.py


To locate multiline string constants automatically:

python - <<'PY'
import ast
from pathlib import Path

roots = [
    Path("src/rho"),
    Path("codex"),
]

for root in roots:
    if not root.exists():
        continue

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue

            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue

            if isinstance(node, ast.Assign):
                names = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
            else:
                names = [node.target.id] if isinstance(node.target, ast.Name) else []

            for name in names:
                upper = name.upper()
                if any(token in upper for token in (
                    "PROMPT",
                    "INSTRUCTION",
                    "PREAMBLE",
                    "SYSTEM",
                    "DESCRIPTION",
                )):
                    print("=" * 80)
                    print(f"{path}:{node.lineno}  {name}")
                    print("=" * 80)
                    print(value.value.strip())
                    print()
PY


This extracts statically declared string constants. Some final prompts are assembled dynamically from:

Shared preambles
Task queries
Harness descriptions
Feature-specific prompt fragments
Output schemas
Trajectory manifests

For the exact runtime-expanded prompts, use a completed run directory. The project states that every run persists prompts and related completions under runs/<timestamp>-<dataset>/.

find runs -type f \( \
  -iname '*prompt*' -o \
  -iname '*instruction*' -o \
  -iname '*request*' \
\) -print

10. What should be copied into AgentEvolve

Do not copy the prompts as one giant prompt. Preserve role separation:

difficulty_judge.md
solve_agent.md
group_analyzer.md
mechanism_judge.md
harness_editor.md
pairwise_evaluator.md


Each template should have:

prompt_id:
prompt_version:
role:
input_schema:
output_schema:
allowed_artifacts:
prohibited_context:
model_id:
temperature:
source_commit:


The most important modification for your RHO-GEPA architecture is to separate:

RHO diagnosis:
  observational evidence and inconsistency

GEPA attribution:
  editable-target hypothesis

Intervention calibration:
  actual causal evidence


That prevents the single diagnosis prompt from simultaneously acting as observer, causal judge, optimizer, and evaluator.

Main extraction rationale

The inventory above separates true model instructions from task files and workspace context, maps every core prompt to the exact implementation stage, and gives you a reproducible way to extract both static prompt constants and fully expanded runtime prompts.