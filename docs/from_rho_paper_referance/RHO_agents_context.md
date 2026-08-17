another things, do rho uses codex_cli everywhere (ie., even with consististancy , validator , pairwise ranker judge, ..... or those are simple llm-calls?) ?

Yes, but not literally everywhere. In the public RHO implementation, the main reasoning stages are run as full Codex CLI agent invocations, while task-difficulty scoring is a conventional structured LLM request and fingerprint embedding is a local embedding operation.

Short answer
RHO component	Execution mechanismTask solver	Codex CLI agent
Repeated group rollouts	Codex CLI agent, one invocation per rollout
Self-validation	Codex CLI agent, inside the diagnosis invocation
Self-consistency analysis	Codex CLI agent, inside the same diagnosis invocation
Harness candidate generation	Codex CLI agent with writable filesystem
Candidate task re-solving	Codex CLI agent
Pairwise candidate ranking	Codex CLI agent
Historical-task difficulty scoring	Ordinary LLM client call
Abstract fingerprint generation	Same ordinary LLM call as difficulty scoring
Fingerprint embeddings	Local embedding model, not Codex CLI
DPP selection and score aggregation	Deterministic Python computation, not an LLM

The repository explicitly says that RHO “drives the Codex CLI as its base agent,” and the source separates agent-based orchestration from the selector’s LLMClient.

1. Task solving uses Codex CLI

The ordinary solve stage gives the agent an isolated filesystem containing:

harness/
task/
    prompt.md
    repo/       if applicable


The solve instructions tell the agent to inspect the harness, understand the task, perform the task, modify repository files when needed, and provide the required final response. This is a complete agent execution rather than a simple chat-completions request.

Conceptually:

trajectory = codex_cli.run(
    working_directory=solve_workspace,
    instructions=SOLVE_INSTRUCTIONS,
)


Therefore, with:

k=10,G=3k=10,\qquad G=3

the baseline group-rollout stage launches:

kG=30kG=30

separate Codex CLI solves.

Each solve receives:

The same task
The same baseline harness
A fresh workspace
An independent agent execution

The repeated trajectories are not produced by asking one ordinary LLM request to generate three hypothetical solutions.

2. Self-validation and self-consistency use one Codex CLI diagnosis agent

This is the key nuance.

RHO does not necessarily make separate model calls for:

self-validation
self-consistency
failure analysis
severity estimation
improvement direction


Instead, the implementation invokes a diagnosis agent for each selected task. That diagnosis agent reads the three sibling trajectories and produces a structured diagnosis. The diagnosis prompt supports feature switches controlling whether validation and consistency are included.

The logical structure is:

One selected task
    ├── rollout 0
    ├── rollout 1
    └── rollout 2
            ↓
One diagnosis Codex CLI call
            ↓
    ├── per-trajectory self-validation
    ├── cross-trajectory self-consistency
    ├── failure-mode analysis
    ├── severity
    └── harness-improvement direction


So, for k=10k=10, there are normally:

1010

diagnosis-agent invocations, not 20 separate calls for validation and consistency.

Why Codex CLI is useful here

The diagnosis is filesystem-rich. The diagnosing agent may inspect:

task/prompt.md
The original harness
Multiple trajectory event streams
Final responses
Workspace diffs
Files modified during each solve

That is more naturally represented as an agent workspace than as one enormous serialized API prompt. The diagnosis orchestration explicitly constructs this workspace and invokes an Agent.

But it is still an LLM judgment

Using Codex CLI does not make validation external or objective. It is still model-generated self-evaluation.

The distinction is:

Ordinary LLM call:
  prompt text → completion text

Codex CLI agent call:
  instructions + filesystem + tools
  → multi-step trajectory + final response


RHO’s self-validation and self-consistency belong to the second category.

3. Harness optimization uses Codex CLI

Candidate harness generation definitely requires an agentic execution because the optimizer modifies a real filesystem.

For every candidate sample, RHO creates a fresh workspace:

workspace/
    harness/
    diagnoses/
        task_0000/
            prompt.md
            diagnosis.md
        task_0001/
            prompt.md
            diagnosis.md
        ...


The optimizer reads the diagnoses, inspects the current harness, and edits files under harness/. The resulting candidate is captured from the filesystem, rather than parsed from the optimizer’s final textual answer.

Conceptually:

candidate_1 = codex_cli.run(fresh_workspace_1)
candidate_2 = codex_cli.run(fresh_workspace_2)
candidate_3 = codex_cli.run(fresh_workspace_3)


The three candidates are independent Codex CLI invocations.

This lets the optimizer do more than rewrite one prompt. It can potentially:

Modify instructions
Add procedural skills
Add scripts
Add executable tools
Reorganize resources
Delete harmful or redundant resources

A plain API call could propose those changes as JSON, but the published RHO implementation uses the filesystem-capable agent abstraction.

4. Pairwise ranking also uses the agent abstraction

The pairwise evaluator is implemented in:

src/rho/orchestrators/evaluate.py


It accepts an Agent, creates an evaluation workspace, invokes the evaluator, captures an evaluation trajectory, and parses a structured score from the result. The evaluation is annotated as an evaluate stage.

Conceptually:

Original task
Baseline harness and trajectory
Candidate harness and trajectory
        ↓
Codex CLI evaluation agent
        ↓
Structured signed preference score


The score represents the transition:

baseline harness → candidate harness


A positive score favors the candidate, a negative score favors the baseline, and zero represents a tie or indeterminate comparison.

Thus, the pairwise evaluator is not implemented as the same lightweight LLM client used for historical difficulty scoring. It goes through the repository’s full Agent interface and therefore normally means a Codex CLI invocation in the reported setup.

5. Difficulty and fingerprint generation use ordinary LLM calls

The historical-task difficulty selector is different.

Its implementation lives in:

src/rho/selection/difficulty_selector.py


and uses:

LLMClient


The judge receives:

The original task
A bounded trajectory digest
Instructions for difficulty scoring
Instructions for abstract fingerprint generation

It returns structured JSON:

{
  "difficulty": 7.5,
  "abstract_fingerprint": "..."
}


This is a normal LLM request, not a Codex CLI agent working through a mutable task workspace.

The likely data flow is:

Task + historical trajectory
        ↓
Digest and sanitization
        ↓
LLMClient request
        ↓
Difficulty score + abstract fingerprint


There is no need for a filesystem-editing agent because this stage needs only a bounded classification and abstraction result.

6. Embeddings and DPP do not use an LLM agent

After the LLM generates the abstract fingerprints, RHO embeds them and constructs the similarity matrix used by the DPP selector. The repository has distinct modules for embedding, local embedding, caching, difficulty selection, and DPP selection.

This part is conventional computation:

Abstract fingerprints
        ↓
Embedding model
        ↓
Normalized vectors
        ↓
Cosine-similarity matrix
        ↓
Difficulty-weighted DPP kernel
        ↓
Greedy subset selection


No Codex CLI reasoning is required for:

Cosine similarity
Kernel construction
Difficulty weighting
DPP greedy MAP selection
Candidate score averaging
Selecting the maximum-scoring candidate
Applying the strict positive-score gate
7. Exact call architecture

For the main RHO loop with:

k=10,G=3,N=3k=10,\quad G=3,\quad N=3

the agentic calls are approximately:

Codex CLI agent invocations
Baseline group solves
kG=10×3=30kG=10\times3=30
Group diagnoses
k=10k=10

Each diagnosis handles validation and consistency together.

Candidate harness editors
N=3N=3
Candidate re-solves
Nk=3×10=30Nk=3\times10=30
Pairwise evaluator calls
Nk=3×10=30Nk=3\times10=30
Total
30+10+3+30+30=10330+10+3+30+30=103

These are the full agent invocations in the primary optimization accounting:

30 solve
10 diagnose
3 optimize
30 candidate solve
30 pairwise evaluate
----------------------
103 agent invocations


The difficulty-judging calls are an additional preprocessing expense and use the selector’s ordinary LLMClient, rather than the same Codex CLI orchestration. Fingerprint embedding and DPP selection are local computational stages. The repository’s structure reflects this separation between the agent orchestrators and the selection subsystem.

8. Same underlying model does not mean same execution mechanism

Another important distinction is between:

model identity


and:

invocation interface


It is possible for the system to use the same underlying model family for several roles while invoking that model differently.

For example:

Difficulty judge:
  direct LLMClient request
  bounded task and trajectory digest
  JSON response

Diagnosis judge:
  Codex CLI agent
  filesystem workspace
  multiple trajectory files
  multi-step inspection
  structured final response

Harness optimizer:
  Codex CLI agent
  writable harness directory
  tool execution and file modifications

Pairwise judge:
  Codex CLI agent
  evaluation workspace
  structured preference output


Therefore, saying “RHO uses the same model for everything” does not imply “RHO uses the same API path for everything.”

9. Implication for your AgentEvolve architecture

You do not have to duplicate RHO’s choice of running every major judge through Codex CLI.

A cleaner AgentEvolve design would provide two execution interfaces.

Interface A: structured stateless model call

Use for operations where the model only needs bounded structured context:

class StructuredLLM:
    def complete(
        self,
        system_prompt,
        input_payload,
        output_schema,
    ):
        ...


Suitable for:

Difficulty scoring
Fingerprint generation
Mechanism normalization
Lightweight cluster labeling
Schema repair
Possibly pairwise judging when all evidence fits safely in context
Interface B: workspace agent execution

Use where tools, files, or iterative inspection are essential:

class WorkspaceAgent:
    def run(
        self,
        workspace,
        instructions,
        permissions,
    ):
        ...


Suitable for:

Task solves
Multi-trajectory diagnosis
Harness editing
Crossover conflict refinement
Complex evaluations that inspect artifacts and diffs
My recommendation by role
Task solver:
  Workspace agent

RHO group analyzer:
  Workspace agent

Self-validation:
  Part of group-analyzer call initially

Self-consistency:
  Part of group-analyzer call initially

GEPA mechanism judge:
  Structured LLM call if sanitized evidence is compact
  Workspace agent only if trace inspection requires tools

Harness editor:
  Workspace agent with restricted write set

Pairwise evaluator:
  Structured LLM call if trajectories are already serialized well
  Workspace agent if large diffs and artifacts require inspection

Difficulty and fingerprint judge:
  Structured LLM call

Embeddings, clustering, DPP, score consolidation:
  Deterministic services, no agent

Bottom line

The published RHO implementation uses Codex CLI agents for the central trajectory-rich and filesystem-rich stages:

Solving
Combined self-validation and self-consistency diagnosis
Harness optimization
Candidate re-solving
Pairwise self-preference evaluation

It uses a lighter direct LLM client for historical-task difficulty and abstract-fingerprint generation. Embeddings, DPP, and final aggregation are ordinary deterministic computations.

The most important detail is that self-validation and self-consistency are generally two diagnostic signals extracted within one diagnosis-agent invocation per task, not two independent Codex calls.