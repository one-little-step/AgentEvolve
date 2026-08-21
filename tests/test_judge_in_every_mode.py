"""The preference judge must reach every mode, not only the RHO ones.

**The gap.** ``compare_preference`` was bound in exactly one place --
``build_rho_hooks`` -- and ``scripts/run_evolution.py`` only reaches that when
``rho_config is not None``, i.e. when ``--mode`` is *not* ``genetic``::

    :1099   if args.mode != "genetic":   ->  build rho_config
    :1146   if rho_config is not None:   ->  _run_rho_rounds -> build_rho_hooks
                                              └─ binds compare_preference
    :1149   else:                        ->  stack.run_iterations()   (no judge)

So ``--mode genetic`` produced candidates with ``preference is None`` forever, and
the SV-4 gate reads "never judged" as "no evidence of improvement". A genetic-only
run could therefore never export anything but the base, no matter how well its
offspring scored.

**Why that is not acceptable even under ``--experimental-candidate-promotion``.**
That flag disables the *gate*; it does not supply a judge. Without a judge there is
no pairwise verdict, no generational retirement, and no trajectory-level evidence --
the run measures aggregate score only, which is the ranking SV-2/SV-3 show to be
wrong. An ablation arm that silently loses the instrument it is meant to be compared
against is not an experiment.

The judge is therefore bound at **stack construction**, which every mode goes
through, rather than in the RHO hook builder. ``build_rho_hooks`` still overrides it
with the injected RHO judge so a test can substitute a fake.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_evolve.pipeline import build_offline_stack, build_rho_hooks


class _Verdict:
    def __init__(self, score: float, available: bool = True) -> None:
        self.score = score
        self.available = available


class _Judge:
    def __init__(self, score: float = 0.8) -> None:
        self.score = score
        self.calls = 0

    def compare_symmetric(self, task, baseline, candidate, **kw):  # noqa: ANN001
        self.calls += 1
        return _Verdict(self.score)


# --------------------------------------------------------------------------- #
# 1. The judge is available without any RHO wiring
# --------------------------------------------------------------------------- #


def test_a_stack_can_be_built_with_a_preference_judge() -> None:
    """The seam that closes the gap: no ``build_rho_hooks`` call anywhere."""
    judge = _Judge()

    stack = build_offline_stack(preference_judge=judge)

    assert stack.runner.compare_preference is not None


def test_the_genetic_loop_uses_the_injected_judge() -> None:
    """A pure genetic attempt must reach the judge, which is what makes its
    offspring promotable at all."""
    judge = _Judge()
    stack = build_offline_stack(preference_judge=judge)

    stack.runner.run_attempt(stack.tasks)

    assert judge.calls > 0, "the genetic path never consulted the judge"


def test_a_genetic_offspring_records_a_preference() -> None:
    """The concrete defect: ``preference`` stayed ``None`` for every genetic
    candidate, so the SV-4 gate disqualified all of them."""
    judge = _Judge()
    stack = build_offline_stack(preference_judge=judge)

    outcome = stack.runner.run_attempt(stack.tasks)

    assert outcome.accepted, outcome.reason
    assert outcome.result_candidate_id is not None
    entry = stack.pool.get(outcome.result_candidate_id)
    assert entry.preference is not None
    assert entry.preference > 0.0


def test_a_genetic_offspring_can_now_be_exported() -> None:
    """End of the chain. Before this, a perfect-scoring genetic offspring lost to
    the base because it carried no preference evidence."""
    judge = _Judge()
    stack = build_offline_stack(preference_judge=judge)

    outcome = stack.runner.run_attempt(stack.tasks)

    assert stack.champion_version() == stack.pool.get(
        outcome.result_candidate_id  # type: ignore[arg-type]
    ).version


def test_a_genetic_run_retires_its_superseded_parent() -> None:
    """Generational retirement was equally unreachable in genetic mode: no judge
    meant no verdict, so no parent could ever be superseded."""
    judge = _Judge()
    stack = build_offline_stack(preference_judge=judge)

    outcome = stack.runner.run_attempt(stack.tasks)

    assert outcome.retired_parent_id == outcome.parent_candidate_id


# --------------------------------------------------------------------------- #
# 2. Defaults and overrides
# --------------------------------------------------------------------------- #


def test_no_judge_supplied_leaves_the_offline_stack_unjudged() -> None:
    """The offline default stays judge-free so the deterministic suite does not
    silently acquire a model-shaped dependency."""
    stack = build_offline_stack()

    assert stack.runner.compare_preference is None


def test_build_rho_hooks_still_overrides_the_bound_judge() -> None:
    """RHO must keep its own injectable judge, or an offline RHO test could not
    substitute a fake for the real adapter."""
    construction_judge = _Judge(score=0.1)
    rho_judge = _Judge(score=0.9)
    stack = build_offline_stack(preference_judge=construction_judge)

    build_rho_hooks(stack, preference_judge=rho_judge)  # type: ignore[arg-type]

    stack.runner.run_attempt(stack.tasks)
    assert rho_judge.calls > 0
    assert construction_judge.calls == 0


def test_an_experimental_promotion_run_still_has_a_judge() -> None:
    """The user's requirement, stated directly.

    ``--experimental-candidate-promotion`` disables the acceptance *gate* so the
    aggregate ranking can be studied on its own. It must not also remove the
    judge: retirement and pairwise resolution are the instruments the ablation is
    being compared against, and a run missing them measures nothing.
    """
    judge = _Judge()
    stack = build_offline_stack(
        preference_judge=judge,
        config_overrides={"experimental_candidate_promotion": True},
    )

    outcome = stack.runner.run_attempt(stack.tasks)

    assert stack.runner.config is not None
    assert stack.runner.config.experimental_candidate_promotion is True
    assert judge.calls > 0, "the ablation arm lost its judge"
    assert stack.pool.get(
        outcome.result_candidate_id  # type: ignore[arg-type]
    ).preference is not None
