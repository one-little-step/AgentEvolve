I’ll unpack the entire RHO pipeline from input to deployment, including what each agent call sees, the DPP mathematics, the diagnostic signals, candidate generation, pairwise selection, compute cost, and the exact experimental settings. I’ll also distinguish what RHO actually learns from what it does not learn.

1. What RHO is trying to optimize

RHO stands for Retrospective Harness Optimization.

The central idea is:

Instead of changing the underlying model or requiring a labeled validation set, let an agent examine its accumulated task trajectories, discover recurring weaknesses, and modify the persistent harness surrounding the model.

Here, the harness is the collection of resources that governs how the agent works:

Instructions and workflow rules
Skills and reusable procedural knowledge
Executable helper tools
Scripts
Configuration files
Context and memory-like resources

The model weights remain fixed. RHO changes the environment and operating procedures around the model.

A task execution is represented as:

τ=solve⁡(h,t)\tau = \operatorname{solve}(h,t)

where:

tt is a task
hh is the current harness
τ\tau is the resulting trajectory

A trajectory includes the agent’s actions, observations, tool calls, workspace changes, and final response. In the formal definition, RHO wants a harness h⋆h^\star that maximizes expected performance on future tasks:

h⋆=arg⁡max⁡h′Et,τ∼solve⁡(h′,t)[U(t,τ)]h^\star = \arg\max_{h'} \mathbb{E}_{t,\tau \sim \operatorname{solve}(h',t)} \left[U(t,\tau)\right]

But the true utility U(t,τ)U(t,\tau) cannot be observed because RHO assumes that no ground-truth labels or representative validation set are available. Therefore, it replaces ground-truth utility with the agent’s own relative preference between trajectories.

2. RHO in one picture

At the highest level, the process is:

Deployed agent
      │
      ▼
Past tasks and trajectories
      │
      ▼
[Stage 1] Select difficult and diverse tasks
      │
      ▼
[Stage 2] Re-solve each selected task several times
      │
      ▼
Diagnose failures and disagreements
      │
      ▼
[Stage 3] Generate several modified harnesses
      │
      ▼
Test each candidate on the selected tasks
      │
      ▼
Agent compares candidate trajectories with baseline trajectories
      │
      ▼
Deploy the best candidate only if it appears better


The important distinction from conventional optimization is that RHO does not repeatedly edit a harness and score it against labeled validation examples. It performs one retrospective optimization pass using previously accumulated trajectories.

3. Inputs and hyperparameters

A single RHO round receives:

D={(ti,τi)}i=1nD=\{(t_i,\tau_i)\}_{i=1}^{n}

and the original harness:

h0h_0

The primary hyperparameters are:

Parameter	Meaning	Paper settingnn	Number of available past task trajectories	Dataset-dependent
kk	Coreset size	10
GG	New rollouts per selected task	3
NN	Candidate harnesses	3
θ\theta	Difficulty/diversity trade-off	0.7
ϵ\epsilon	Difficulty-score floor	0.1
Ranking range	Candidate comparison score	[−10,10][-10,10]
Acceptance threshold	Minimum candidate mean score	Strictly >0>0

The experiments use the same model backbone for solving, judging difficulty, diagnosing trajectories, editing the harness, and ranking candidate trajectories. Role separation is implemented through different prompts and isolated workspaces, not through separate specialist models.

4. Stage 1: Coreset selection
4.1 Why not use every old trajectory?

If the agent has thousands of past trajectories, examining and replaying all of them would be expensive. More importantly, easy and repetitive tasks could overwhelm the relatively small number of trajectories containing useful failure information.

RHO therefore selects a small set:

Dcore⊂D,∣Dcore∣=kD_{\text{core}}\subset D, \qquad |D_{\text{core}}|=k

The selected set should satisfy two goals:

Include difficult trajectories, because they are likely to expose harness weaknesses.
Remain diverse, so the optimizer does not learn from only one narrow failure mode.
4.2 Difficulty judgment

For every past pair (ti,τi)(t_i,\tau_i), an LLM judge produces:

{
  "difficulty": 7.8,
  "abstract_fingerprint": "A multi-file change requiring..."
}


The difficulty score is:

ri∈[0,10]r_i \in [0,10]

The abstract fingerprint describes the structural form of the task and failure mode without using repository names, filenames, framework names, function names, or other task-specific identifiers.

For example, instead of writing:

“The agent failed to update the Django serializer.”

the fingerprint should say something like:

“A cross-module change where a shared contract must be propagated consistently across several call sites.”

This abstraction is important because RHO wants semantic similarity based on general failure structures, not superficial similarity based on project vocabulary.

The judge sees:

The task description
A bounded digest of one previous trajectory
The beginning and end of the trajectory if truncation is needed

Commands that might expose expected-answer files are scrubbed before the trajectory digest is shown to the difficulty judge. The digest budget in the experiments is 10,000 BPE tokens.

4.3 Fingerprint embeddings and similarity

Each fingerprint is embedded into a vector xix_i. In the experiments, the paper uses:

BAAI/bge-large-en-v1.5
1,024-dimensional embeddings
Local embedding execution

The semantic similarity between two tasks is computed with cosine similarity:

Sij=cos⁡(xi,xj)S_{ij}=\cos(x_i,x_j)

If vectors are normalized, this can be represented as:

S=XX⊤S = XX^\top

Tasks with similar structural failure modes receive high similarity values.

4.4 Difficulty weighting

The raw difficulty scores are normalized and transformed into weights:

r~i=(max⁡(ri,ϵ)max⁡jmax⁡(rj,ϵ))α\tilde r_i = \left( \frac{\max(r_i,\epsilon)} {\max_j\max(r_j,\epsilon)} \right)^\alpha

where:

α=θ2(1−θ)\alpha = \frac{\theta}{2(1-\theta)}

For the paper’s value θ=0.7\theta=0.7:

α=0.72(0.3)≈1.1667\alpha = \frac{0.7}{2(0.3)} \approx 1.1667

The floor ϵ=0.1\epsilon=0.1 prevents a zero difficulty score from receiving an exact zero weight.

4.5 DPP kernel

RHO constructs the Determinantal Point Process kernel:

K=diag⁡(r~)Sdiag⁡(r~)K = \operatorname{diag}(\tilde r) S \operatorname{diag}(\tilde r)

For a candidate subset YY, the DPP assigns a quantity proportional to:

det⁡(KY)\det(K_Y)

This determinant becomes large when:

The selected items have large difficulty weights.
Their representations point in different semantic directions.
Therefore, they are simultaneously difficult and non-redundant.

RHO uses greedy MAP selection to pick k=10k=10 tasks.

4.6 Meaning ofθ\theta
θ=1\theta=1: prioritize difficulty.
θ=0\theta=0: prioritize diversity or coverage.
θ=0.7\theta=0.7: place greater emphasis on difficulty while preserving diversity.

The paper’s ablation found that pure difficulty caused selected tasks to cluster in a narrow task region, while pure coverage could select diverse but insufficiently informative tasks. The combination achieved the strongest downstream improvement.

5. Stage 2: Group rollout

Once the coreset is selected, RHO does not rely only on the original historical trajectory. It re-solves every selected task multiple times using the original harness h0h_0.

For every task t∈Dcoret\in D_{\text{core}}:

{τt,g}g=1G←solve⁡(h0,t)\{\tau_{t,g}\}_{g=1}^{G} \leftarrow \operatorname{solve}(h_0,t)

With the paper’s settings:

k=10,G=3k=10,\qquad G=3

RHO therefore creates:

10×3=3010 \times 3 = 30

new group-rollout trajectories.

The tasks and repeated solves are executed in parallel where infrastructure allows.

Why repeat the same task?

A single trajectory is a noisy sample. An agent may fail because of:

A bad early assumption
An inefficient tool sequence
Premature stopping
Random exploration
Failure to inspect an important file
Incorrect interpretation of the task
An actual gap in the harness

Three attempts reveal whether the behavior is stable or uncertain.

For example:

Task: Fix an API compatibility problem

Rollout 1: Edits the first visible function and stops.
Rollout 2: Traces the call chain, edits two modules, and tests.
Rollout 3: Changes configuration but never executes a runtime test.


This group reveals more than any one trajectory:

Rollout 1 may show premature stopping.
Rollout 3 may show misplaced diagnosis.
Only rollout 2 may expose a reliable workflow.
6. Diagnosis: self-validation and self-consistency

For each selected task, the diagnosis agent receives:

The original task
The shared original harness
All G=3G=3 trajectory directories
Each trajectory’s event stream
Its final message
Its workspace diff

It then produces one structured diagnosis ItI_t.

6.1 Self-validation

Self-validation asks:

Looking at each trajectory individually, does the execution appear to have completed the requested task correctly and efficiently?

The diagnosing agent checks for:

Incorrect tool calls
False assumptions
Missing evidence
Failure to inspect relevant material
Incomplete implementation
Mismatch with exact task requirements
Premature stopping
Unnecessary work
Failure to verify outputs

Each trajectory receives a binary internal judgment:

{
  "trajectory": "trajectory_0",
  "successful": 0,
  "quality_analysis": "...",
  "issues": "..."
}


Crucially, this is self-validation, not external grading. The model infers likely correctness using the prompt, observations, file changes, tool output, and final answer.

6.2 Self-consistency

Self-consistency asks:

Do the repeated trajectories agree on the important parts of solving this task?

The diagnosis agent examines disagreements in:

Plans
Tool sequences
Files selected for editing
Interpretation of the requirement
Final answers
Stopping conditions
Verification strategies

Not every difference matters. RHO focuses on consequential disagreements.

For example:

Different wording in the final answer may be harmless.
Editing different modules for the same bug may signal uncertainty.
One trajectory testing the code and another claiming success without testing may reveal a harness-level verification gap.

Low consistency is interpreted as evidence that the harness does not sufficiently constrain or support the agent’s strategy.

6.3 Severity

The diagnosis contains:

severityt∈[0,1]\text{severity}_t \in [0,1]

Interpretation:

0.00.0: all trajectories are good, with no meaningful issue.
0.10.1 to 0.30.3: minor inefficiency or weak concern.
0.40.4 to 0.70.7: inconsistent results or a plausible harness gap.
0.80.8 to 1.01.0: clear failure or strong evidence of a missing capability.

Severity is a soft attention weight, not ground truth and not a hard filter.

6.4 Final diagnostic object

The combined diagnostic is conceptually:

It=rank⁡val(t,{τt,g})∪rank⁡con(t,{τt,g})I_t = \operatorname{rank}^{\mathrm{val}} (t,\{\tau_{t,g}\}) \cup \operatorname{rank}^{\mathrm{con}} (t,\{\tau_{t,g}\})

The output includes:

{
  "task_id": "...",
  "severity": 0.85,
  "trajectory_analyses": [...],
  "failure_mode_analysis": "...",
  "inconsistency_analysis": "...",
  "harness_improvement_direction": "..."
}


All task diagnoses are then collected:

I=⋃t∈DcoreItI = \bigcup_{t\in D_{\text{core}}} I_t

These diagnostic summaries, rather than raw trajectories alone, become the main instructions for optimizing the harness.

7. Stage 3: Best-of-NN harness proposal
7.1 Why generate multiple candidates?

Harness editing is stochastic. Even when two optimizer calls receive the same diagnoses, they may:

Prioritize different failure modes
Write different instructions
Create different helper scripts
Overfit to a single task
Produce a no-op
Introduce a regression

RHO therefore samples NN candidates independently:

hj=optimize⁡(h0,I),j=1,…,Nh_j = \operatorname{optimize}(h_0,I), \qquad j=1,\ldots,N

The paper uses:

N=3N=3

These optimization calls can run in parallel.

7.2 What the optimizer sees

Each optimizer gets:

harness/
    Current harness; writable

diagnoses/
    task_0001/
        diagnosis.md
        prompt.md
    task_0002/
        diagnosis.md
        prompt.md
    ...


Diagnoses are ordered by descending severity.

The optimizer is instructed to:

Read all diagnoses.
Treat severity as a soft signal.
Look for recurring patterns across tasks.
Prioritize recurring, high-severity failures.
Avoid making an edit based on one isolated low-severity observation.
Make surgical, generalizable improvements.
Avoid task-specific hardcoding.
7.3 The optimizer edits a filesystem, not merely a prompt

This is one of RHO’s important design choices.

The optimizer may:

Add Markdown instruction files
Rewrite existing procedures
Add a checklist
Add executable shell or Python utilities
Add configuration
Remove unhelpful resources
Reorganize skills
Modify any file in the harness directory

The output harness is recovered from the resulting filesystem. RHO does not depend on parsing an edit from the model’s final textual answer.

A candidate is removed before evaluation if:

Optimization fails
The call times out
The resulting harness is identical to the input harness

Thus, a no-op is not treated as a meaningful candidate.

8. Candidate re-solving

Every surviving candidate harness is tested on each coreset task:

τt(j)=solve⁡(hj,t)\tau_t^{(j)} = \operatorname{solve}(h_j,t)

With:

N=3,k=10N=3,\qquad k=10

this produces:

3×10=303 \times 10 = 30

candidate-harness trajectories.

For comparison, RHO also fixes one original-harness rollout for each task:

τt(0)\tau_t^{(0)}

The same baseline trajectory is used for all candidates on that task. This common reference prevents one candidate from being compared against an unusually weak baseline while another is evaluated against a strong baseline.

This also means that G>1G>1 is primarily a diagnostic mechanism. The three original rollouts help expose inconsistency, but they do not vote directly on which candidate harness wins.

9. Pairwise self-preference ranking

For every candidate-task pair, an evaluator compares:

The task
Candidate and baseline harnesses
Candidate and baseline trajectories
Final messages
Event histories

It returns an integer:

pj,t∈[−10,10]p_{j,t}\in[-10,10]

Conceptually:

Positive means the candidate transition is better.
Zero means comparable or indeterminate.
Negative means the candidate is worse.

The implementation deliberately controls presentation order and flips the parsed orientation where necessary so that the stored score consistently represents the improvement from baseline to candidate. Candidate-first presentation is used to reduce observed later-option preference bias.

The candidate’s aggregate score is:

Sj=1k∑t∈Dcorerank⁡(t,τt(j),τt(0))S_j = \frac{1}{k} \sum_{t\in D_{\text{core}}} \operatorname{rank} \left( t,\tau_t^{(j)},\tau_t^{(0)} \right)

Then:

j⋆=arg⁡max⁡jSjj^\star = \arg\max_j S_j
Strict acceptance gate

The final update rule is:

h⋆={hj⋆,Sj⋆>0h0,Sj⋆≤0h^\star = \begin{cases} h_{j^\star}, & S_{j^\star}>0\\ h_0, & S_{j^\star}\leq 0 \end{cases}

Therefore, RHO is allowed to make no update.

A tie does not count as improvement. This conservative gate is intentional because the pairwise judge is noisy. Accepting a zero-score candidate would introduce change without estimated benefit.

If a ranking call fails or its JSON cannot be parsed, the implementation assigns a score of zero. It does not retry. This pulls the candidate toward rejection rather than accidentally rewarding an evaluation failure.

10. Exact algorithm in compact pseudocode
Input:
    Historical task-trajectory dataset D
    Original harness h0
    Coreset size k
    Group size G
    Candidate count N
    DPP trade-off theta

Stage 1: Coreset selection
    for every historical pair (ti, taui):
        ri, fingerprinti = difficulty_judge(ti, taui)

    embed all fingerprints
    construct similarity matrix S
    construct difficulty-weighted DPP kernel K
    Dcore = greedy_DPP_MAP(K, k)

Stage 2: Group rollout and diagnosis
    for every task t in Dcore, in parallel:
        run solve(h0, t) G times
        fix one rollout as baseline tau_t^(0)

        self_validation =
            inspect each trajectory for likely correctness

        self_consistency =
            inspect disagreements across trajectories

        It =
            structured diagnosis containing:
            severity
            per-trajectory assessment
            failure modes
            inconsistency analysis
            improvement direction

    I = union of all task diagnoses

Stage 3: Best-of-N proposal
    for j = 1 to N, in parallel:
        hj = optimize a writable copy of h0 using I

        if hj failed, timed out, or is identical to h0:
            discard hj
            continue

        for every task t in Dcore:
            tau_t^(j) = solve(hj, t)
            p_jt = pairwise_rank(
                candidate trajectory,
                fixed baseline trajectory
            )

        Sj = mean of p_jt across all k tasks

    choose candidate with largest Sj

    if maximum Sj > 0:
        return that candidate
    else:
        return h0 unchanged


This corresponds directly to Algorithm 1 and the implementation details in the paper.

11. Agent-call accounting

For the paper’s settings k=10k=10, G=3G=3, and N=3N=3, one RHO optimization round requires:

Difficulty selection
One auxiliary difficulty judgment per available historical candidate.
Local embedding of fingerprints.
These calls are not included in the 103 Codex-agent invocation count reported for the optimization phase.
Group rollout
kG=10×3=30kG = 10 \times 3 = 30
Diagnosis
k=10k = 10
Harness optimization
N=3N = 3
Candidate re-solving
Nk=3×10=30Nk = 3 \times 10 = 30
Pairwise ranking
Nk=3×10=30Nk = 3 \times 10 = 30
Total optimization-phase agent invocations
30+10+3+30+30=10330+10+3+30+30 = 103

Held-out evaluation calls are additional. On SWE-Bench Pro, the reported accounting is 103 optimization calls plus 100 held-out test solves, for 203 total agent invocations.

12. What the optimized harness actually contained

RHO did not simply produce vague memories such as “test more carefully.” It created benchmark-specific, task-agnostic operating resources.

SWE-Bench Pro

The optimized harness added rules and tools related to:

Maintaining a requirement ledger
Tracing the actual code path before editing
Reusing proven historical fixes when applicable
Running behavior-level smoke tests
Locating non-standard toolchains
Excluding caches and generated artifacts from patches
Checking build, lint, formatting, and targeted tests

The paper gives the example of learning that the Go toolchain may exist outside the default path and that Python cache artifacts must be removed before producing the final patch.

Terminal-Bench 2

The harness added procedures for:

Black-box recovery using query evidence
Final grader-shaped validation
Package installation and import verification
Output and artifact checks
Polygon-mask structural validation
Broad package smoke tests
GAIA-2

The harness added procedures for:

Anchoring deadlines to the simulated clock
Decomposing mandatory and conditional actions
Resolving ambiguous targets before irreversible actions
Inspecting exact tool schemas
Paginating APIs completely
Verifying writes by reading them back
Sending the actual user-facing response through the environment interface

13. Why the diagnostic stage matters

The paper’s ablation compares full diagnosis with weakened versions:

Variant	SWE-Bench Pro	Terminal-Bench 2	GAIA-2Full diagnosis	0.78	0.76	0.37
Without self-consistency	0.56	0.75	0.27
Without self-validation	0.70	0.73	0.30
Raw trajectories only	0.60	0.75	0.29

This shows that simply giving raw trajectories to the harness editor is not equivalent to explicitly extracting failure and disagreement signals.

The effect also differs by domain:

On SWE-Bench Pro, removing self-consistency caused a severe drop.
On all three benchmarks, removing self-validation reduced performance.
Full diagnosis performed best in every reported domain.

The likely mechanism is information compression. The diagnostic stage converts long, noisy event histories into structured, cross-run evidence that tells the optimizer:

What repeatedly failed
Why it failed
Whether different attempts disagreed
How significant the issue is
What general harness capability may be missing
14. Results and behavioral effect

Reported held-out pass rates were:

Benchmark	Original harness	RHO harness	Absolute gainSWE-Bench Pro	0.59	0.78	+0.19
Terminal-Bench 2	0.71	0.76	+0.05
GAIA-2	0.29	0.37	+0.08

These results come from a single retrospective optimization round using no ground-truth validation labels during harness selection.

RHO also changed how the agent behaved:

SWE-Bench Pro: verification actions increased by 61 percent.
Terminal-Bench 2: navigation increased while unnecessary editing decreased.
GAIA-2: execution increased while editing decreased.
Improvements were concentrated particularly on longer-horizon tasks.

The interpretation is that RHO does not merely append information. The new procedures and tools change the agent’s action policy, such as when it verifies, executes, edits, searches, or navigates.

15. What RHO is not

RHO is not:

Model fine-tuning
Reinforcement learning over model weights
Training a new reward model
Validation-set optimization
Simple memory accumulation
Majority voting over final answers
Guaranteed self-correction
Repeated iterative optimization in the reported main setup

It is best understood as:

Offline, label-free, best-of-NN search over editable agent harnesses, guided by structured retrospective diagnosis and self-preference.

16. Important limitations
Self-judgment can be wrong

The same model family is solving, diagnosing, optimizing, and ranking. A shared misconception can survive every stage.

Repeated execution must be safe

Group rollout assumes a task can be reset and attempted multiple times. It is unsuitable for irreversibly destructive or one-shot environments unless they can be safely simulated.

Historical poisoning is possible

If past trajectories contain prompt injections, malicious observations, or adversarial instructions, RHO could turn those artifacts into persistent harness behavior.

The coreset can miss a capability

RHO only optimizes against weaknesses exposed by the selected tasks. A failure mode absent from the coreset may remain unaddressed.

Candidate selection is imperfect

The self-preferred harness does not always achieve the highest held-out test score. The paper reports that selection reliably avoided the worst candidate, but did not invariably choose the true best candidate.

Human oversight is still needed

The paper recommends:

Persistent audit logs
Human approval for sensitive harness edits
Domain-specific safety checks
Inspection of harness diffs before deployment
Rejection of suspicious or overly task-specific tools
Final mental model

You can think of RHO as an agent conducting a structured postmortem:

Find informative incidents
 Select difficult but non-redundant past tasks.

Reproduce each incident
 Retry each selected task several times under the old harness.

Compare the reproductions
 Identify likely mistakes within each run and disagreements across runs.

Write engineering recommendations
 Convert those observations into structured harness-improvement directions.

Build several fixes
 Independently generate multiple edited harness directories.

Replay the incidents under every fix
 Re-solve the same selected tasks with each candidate harness.

Perform pairwise review
 Compare candidate executions against a fixed old-harness baseline.

Use a conservative deployment gate
 Deploy only the highest-scoring candidate, and only if its average preference score is strictly positive.

The essential insight is that variation among repeated attempts acts as a diagnostic instrument. RHO uses that variation to infer where the harness is underspecified, then changes persistent instructions, skills, and executable tools so that future trajectories become more reliable.