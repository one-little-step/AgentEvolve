"""?17 Layers 1+2 — retry hardening against Console Go degradation windows.

Facts established by diagnosis (see plan Decisions 2026-08-25):

* The endpoint dies with 503 "Console Go ... Endpoint is unavailable" under
  load windows lasting minutes; requests over a MOVING ~24-32KB threshold are
  hit hardest while small ones survive.
* Every call already retries twice in ~1.5s (openai SDK default
  ``max_retries=2``) — useless against minute-long windows.
* CUGA's ``_create_llm_instance`` whitelists constructor keys, so a TOML
  ``max_retries`` would be silently dropped; and langchain bakes
  ``max_retries`` into its HTTP clients AT CONSTRUCTION, so post-construction
  mutation cannot help either.

Therefore: Layer 1 patches the model-construction FACTORY (the same
adapter-side seam as the response-cache policy) to supply a subclass whose
``max_retries`` field defaults higher — landing in the built clients because
it exists before validation. Layer 2 sends ``num_retries`` on OUR litellm
roles. Both honor ``AE_RETRY_MAX_RETRIES`` (0 disables).
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def fresh_manager():
    from cuga.backend.llm.models import LLMManager

    from agent_evolve.cuga_wrapper.retry_policy import uninstall_retry_policy

    uninstall_retry_policy()  # pristine factory regardless of test order
    manager = LLMManager()
    manager.clear_models()
    yield manager
    manager.clear_models()
    uninstall_retry_policy()


@pytest.fixture(autouse=True)
def _isolate_factory():
    """Hard guarantee: no test inherits another's patched factory."""
    from agent_evolve.cuga_wrapper.retry_policy import uninstall_retry_policy

    uninstall_retry_policy()
    yield
    uninstall_retry_policy()


def _openai_settings() -> dict:
    return {
        "platform": "openai",
        "model": "fixture-model",
        "max_tokens": 64,
        "temperature": 0.1,
    }


class TestFactoryRetryPolicy:
    def test_built_models_carry_patient_retries(
            self, fresh_manager, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fixture")
        monkeypatch.delenv("AE_RETRY_MAX_RETRIES", raising=False)
        from agent_evolve.cuga_wrapper.retry_policy import install_retry_policy

        assert install_retry_policy(fresh_manager) is True
        model = fresh_manager.get_model(_openai_settings())
        assert model.max_retries == 4
        # Baked into the constructed HTTP clients, not just the field:
        assert model.root_async_client.max_retries == 4
        assert model.root_client.max_retries == 4

    def test_install_is_idempotent(self, fresh_manager, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fixture")
        monkeypatch.delenv("AE_RETRY_MAX_RETRIES", raising=False)
        from agent_evolve.cuga_wrapper.retry_policy import install_retry_policy

        first = install_retry_policy(fresh_manager)
        second = install_retry_policy(fresh_manager)
        assert first is True and second is False  # second is a no-op
        model = fresh_manager.get_model(_openai_settings())
        assert model.max_retries == 4  # still exactly one layer deep

    def test_env_zero_disables_entirely(
            self, fresh_manager, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fixture")
        monkeypatch.setenv("AE_RETRY_MAX_RETRIES", "0")
        from agent_evolve.cuga_wrapper.retry_policy import install_retry_policy

        assert install_retry_policy(fresh_manager) is False
        model = fresh_manager.get_model(_openai_settings())
        assert model.max_retries is None  # langchain defers to the SDK default
        assert model.root_client.max_retries == 2  # openai SDK default


class TestLitellmRolesRetry:
    def test_our_roles_send_num_retries(
            self, monkeypatch) -> None:
        monkeypatch.delenv("AE_RETRY_MAX_RETRIES", raising=False)
        import litellm

        from agent_evolve.adapters.cuga_positivity_judge import CugaPositivityJudge
        from agent_evolve.core.correlation import correlation_scope

        seen: dict = {}

        def fake_completion(**request: object) -> object:
            seen.update(request)  # type: ignore[arg-type]
            return {
                "choices": [{"message": {"content": json.dumps({"findings": []})}}]
            }

        monkeypatch.setattr(litellm, "completion", fake_completion)

        judge = CugaPositivityJudge(model="test-model")

        with correlation_scope(
            run=None, candidate="cand-X", task="task-a", rollout=0,
            phase="positivity",
        ):
            judge.analyze_success(_task(), _passing_trace())

        assert seen.get("num_retries") == 4


def _task():
    from agent_evolve.core.contracts import EvolutionTask

    return EvolutionTask(
        task_id="task-a", input_text=f"analyze {_TOKEN}", expected_contract="c",
    )


def _passing_trace():
    from agent_evolve.core.contracts import ExecutionTrace

    return ExecutionTrace(
        trace_id="tr-pass", candidate_id="v1", task_id="task-a",
        events=(), final_output=f"ok {_TOKEN}", status="success",
    )


_TOKEN = "alpha-token-123"
