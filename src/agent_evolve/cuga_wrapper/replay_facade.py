"""W3: ``ReplayExperimentFacade`` -- the editor-facing replay entrypoint.

Gap 4 (``docs/plans/editor-tools-live-wiring.md``): replay was proven in scripts
only, with no reachable entrypoint from the pipeline. The facade owns the four
pieces the scripts did by hand -- hybrid-model build, ``set_llm`` injection,
fresh workspace, scrub registry -- behind one ``run(...)`` that returns a RAW
report. It is deliberately NOT a runner method: it imports the CUGA-facing
replay model, so it lives adapter-side and keeps ``core/`` agent-neutral.

Design (from §5 architecture correction, user-settled):
* raw observations only -- NO verdict enum. The editor interprets gates, final
  output and cost itself;
* gate on/off is an internal parameter (``gate_enabled``), not a user-facing
  decision: ``True`` = control arm (re-verify the taped prefix), ``False`` =
  mutation arm (an injected artifact legitimately rewrites prompts from call
  one);
* ``resume=None`` + an ``analysis`` resolves the boundary via
  ``boundary_for_fault``; an explicit ``resume`` wins; neither means replay the
  whole run is pointless -> the caller falls back to full validation itself.

The one thing that cannot run offline -- the actual CUGA wrapper run -- is an
injectable seam (``run_task``), so the orchestration and report contract are
unit-tested without the SDK or a live endpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent_evolve.core.tape import TapeIndex, boundary_for_fault

__all__ = ["ReplayExperimentFacade", "ReplayExperimentReport", "default_live_factory"]


def default_live_factory() -> Any:
    """Build the live model for the hybrid tape's tail (mirrors the scripts).

    Clears the pre-instantiated override so the manager falls through to the
    configured platform model, then returns it. This is what makes the taped
    prefix free and the tail a real provider call.
    """
    from cuga.backend.llm.models import LLMManager
    from cuga.config import settings as cuga_settings

    manager = LLMManager()
    manager.clear_pre_instantiated_model()
    return manager.get_model(cuga_settings.agent.code.model)


@dataclass(frozen=True)
class ReplayExperimentReport:
    """RAW structured observations of one replay experiment (no verdict)."""

    status: str  # "ok" | "error"
    resume_boundary: int | None
    gate_enabled: bool
    taped_calls: int
    live_calls: int
    elapsed_s: float
    trace_dir: str | None
    final_output: str
    final_output_chars: int
    #: First-failure excerpt when the taped prefix diverged (None = clean).
    divergence: str | None
    error: str | None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "resume_boundary": self.resume_boundary,
            "gate_enabled": self.gate_enabled,
            "taped_calls": self.taped_calls,
            "live_calls": self.live_calls,
            "elapsed_s": round(self.elapsed_s, 2),
            "trace_dir": self.trace_dir,
            "final_output": self.final_output,
            "final_output_chars": self.final_output_chars,
            "divergence": self.divergence,
            "error": self.error,
        }


@dataclass
class ReplayExperimentFacade:
    """Builds + injects a hybrid tape model and runs one replay experiment."""

    scrub_patterns: tuple[Any, ...] = ()
    live_factory: Callable[[], Any] | None = None
    workspace_root: Path | None = None
    #: Cost cap: refuse to go live beyond this many live calls (None = uncapped).
    max_live_calls: int | None = None
    #: Injectable seams (tests): build the hybrid model + run the task.
    _build_model: Callable[..., Any] | None = field(default=None, repr=False)
    _run_task: Callable[..., Mapping[str, Any]] | None = field(
        default=None, repr=False
    )
    _inject_model: Callable[[Any], None] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    def resolve_resume(self, parent_trace_dir: Path, analysis: object | None) -> int | None:
        """Resume boundary from the blame graph, or None (full validation)."""
        if analysis is None:
            return None
        index = TapeIndex.load(Path(parent_trace_dir))
        return boundary_for_fault(index, analysis)

    def build_model(
        self,
        parent_trace_dir: Path,
        resume: int,
        *,
        gate_enabled: bool,
    ):
        """A hybrid tape model: taped prefix up to ``resume``, then live."""
        if self._build_model is not None:
            return self._build_model(parent_trace_dir, resume, gate_enabled)

        from agent_evolve.cuga_wrapper.tape_replay import HybridTapeModel

        model = HybridTapeModel.from_trace(
            parent_trace_dir,
            cutoff=resume,
            scrub_patterns=self.scrub_patterns,
            gate_enabled=gate_enabled,
        )
        if self.live_factory is not None:
            model._live_factory = self.live_factory  # type: ignore[attr-defined]
        return model

    def run(
        self,
        *,
        parent_trace_dir: Path,
        task_id: str,
        harness_config: Mapping[str, Any],
        resume: int | None = None,
        analysis: object | None = None,
        gate_enabled: bool = True,
    ) -> ReplayExperimentReport:
        """Run one replay experiment; return raw observations.

        ``resume`` explicit wins; else derived from ``analysis`` via
        ``boundary_for_fault``; if neither resolves, there is no useful replay
        (fault before the first boundary or at the tail) and the report says so
        with ``status="error"`` -- the caller falls back to full validation.
        """
        resolved = resume
        if resolved is None:
            resolved = self.resolve_resume(parent_trace_dir, analysis)
        if resolved is None:
            return ReplayExperimentReport(
                status="error",
                resume_boundary=None,
                gate_enabled=gate_enabled,
                taped_calls=0,
                live_calls=0,
                elapsed_s=0.0,
                trace_dir=None,
                final_output="",
                final_output_chars=0,
                divergence=None,
                error=(
                    "no useful resume boundary: the fault precedes the first "
                    "LLM boundary or sits at the tail; fall through to full "
                    "validation"
                ),
            )

        model = self.build_model(parent_trace_dir, resolved, gate_enabled=gate_enabled)
        self._inject(model)

        started = time.time()
        try:
            result = self._run(model, task_id, harness_config)
        except Exception as exc:  # noqa: BLE001 - a crashed arm is a report
            return ReplayExperimentReport(
                status="error",
                resume_boundary=resolved,
                gate_enabled=gate_enabled,
                taped_calls=_taped(model),
                live_calls=_live(model),
                elapsed_s=time.time() - started,
                trace_dir=None,
                final_output="",
                final_output_chars=0,
                divergence=_divergence(model),
                error=f"{type(exc).__name__}: {exc}",
            )

        trace_dir = result.get("causal_trace_path")
        final_output = str(result.get("final_output", ""))
        return ReplayExperimentReport(
            status="ok",
            resume_boundary=resolved,
            gate_enabled=gate_enabled,
            taped_calls=_taped(model),
            live_calls=_live(model),
            elapsed_s=time.time() - started,
            trace_dir=str(trace_dir) if trace_dir else None,
            final_output=final_output,
            final_output_chars=len(final_output),
            divergence=_divergence(model),
            error=None,
        )

    # ------------------------------------------------------------------ #
    def _inject(self, model: Any) -> None:
        if self._inject_model is not None:
            self._inject_model(model)
            return
        from cuga.backend.llm.models import LLMManager

        manager = LLMManager()
        manager.clear_models()
        manager.set_llm(model)

    def _run(self, model: Any, task_id: str, harness_config: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._run_task is not None:
            return self._run_task(model, task_id, harness_config)

        from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

        settings = RuntimeSettings.from_env()
        settings.configure_cuga_environment()
        trace_config = TraceConfig(
            enabled=True,
            output_root=(
                self.workspace_root / "traces"
                if self.workspace_root is not None
                else Path("terminal_output/replay-experiment/traces")
            ),
        )
        wrapper = CugaWrapper.from_cuga(settings, trace_config)
        if self.workspace_root is not None:
            wrapper._runtime._workspace_root = (  # type: ignore[attr-defined]
                self.workspace_root / "workspaces"
            )
        return dict(wrapper.run_task(task_id, dict(harness_config)))


def _taped(model: Any) -> int:
    pointer = getattr(model, "pointer", 0)
    return pointer - _live(model)


def _live(model: Any) -> int:
    return max(0, getattr(model, "live_calls", 0))


def _divergence(model: Any) -> str | None:
    # The hybrid model raises TapeDivergence on the first misaligned prefix
    # call; a clean prefix leaves nothing here. Surface the first failure if
    # the model recorded one (a subclass/observer may set it).
    return getattr(model, "divergence", None)
