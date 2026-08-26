"""Retry hardening for Console Go degradation windows (?17, Layers 1+2).

Diagnosed 2026-08-25: the provider intermittently refuses requests above a
MOVING ~24-32KB threshold with 503 "Console Go ... Endpoint is unavailable"
for minutes at a time. The openai SDK already retries twice in ~1.5s —
useless against such windows. CUGA's ``_create_llm_instance`` whitelists its
constructor keys (a TOML ``max_retries`` would be dropped), and langchain
bakes ``max_retries`` into its HTTP clients at construction time, so only a
construction-time default reaches the wire.

Layer 1 (:func:`install_retry_policy`) therefore wraps CUGA's cached
``_get_reasoning_chat_openai`` factory — the same adapter-side seam as the
response-cache policy — returning a subclass whose ``max_retries`` field
defaults higher. Every manager-built openai-platform model then constructs
patients clients. This covers ALL CUGA-internal callers (proven uniform in
?16-dissolution). Other platforms (azure/groq) construct separate branches
and are NOT covered; note for an upstream contribution.

Layer 2 lives in ``RuntimeSettings``/wrapper paths that call litellm
directly: they pass ``num_retries=resolve_max_retries()``.

Both layers honor ``AE_RETRY_MAX_RETRIES``; ``0`` disables the layer.

This is deliberately NOT vendored-edit: if CUGA later adds a passthrough
``max_retries`` key upstream, this module becomes redundant and removable.
"""

from __future__ import annotations

import os

__all__ = [
    "RETRY_MAX_RETRIES_DEFAULT",
    "ENV_MAX_RETRIES",
    "resolve_max_retries",
    "install_retry_policy",
]

RETRY_MAX_RETRIES_DEFAULT = 4
ENV_MAX_RETRIES = "AE_RETRY_MAX_RETRIES"


def resolve_max_retries() -> int | None:
    """Effective retry count from env; ``None`` means the layer is disabled."""
    raw = os.environ.get(ENV_MAX_RETRIES)
    if raw is None:
        return RETRY_MAX_RETRIES_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return RETRY_MAX_RETRIES_DEFAULT
    return value if value > 0 else None


def install_retry_policy(manager: object | None = None,
                         max_retries: int | None = None) -> bool:
    """Patch CUGA's reasoning-chat factory with a patient-retry subclass.

    Returns ``True`` when the policy was installed now, ``False`` when it was
    already installed or disabled via env. Idempotent by module marker.
    """
    if max_retries is None:
        max_retries = resolve_max_retries()
    if not max_retries:
        return False

    from cuga.backend.llm import models as llm_models

    if getattr(llm_models, "_AE_RETRY_INSTALLED", False):
        return False

    # CUGA caches the class as an attribute ON THE FACTORY FUNCTION and reads
    # it unconditionally through the MODULE GLOBAL name. After we repoint that
    # name, every read/write must land on one coherent cache. So: resolve the
    # base class eagerly through the ORIGINAL factory, wrap it, then publish
    # the subclass AS the cache and swap in a reader that trusts the cache.
    factory = llm_models._get_reasoning_chat_openai
    if getattr(factory, "_cls", None) is None:
        factory._cls = None  # mirror upstream's own initialization shape
    base = factory()  # builds ReasoningChatOpenAI on first ever use
    subclass = type(
        "ReasoningChatOpenAIWithRetries",
        (base,),
        {
            "__annotations__": {"max_retries": int},  # pydantic v2 field override
            "max_retries": int(max_retries),
            "__doc__": f"{base.__doc__ or ''}\n"
                       f"?17 retry hardening: max_retries={max_retries}.",
        },
    )
    factory._cls = subclass

    def patched() -> type:
        return subclass  # closed over: no cache-attribute lookups at all

    # Keep BOTH cache locations coherent: the original function object (for
    # any captured direct references) and the module-global name (which now
    # resolves to `patched`) must agree with what we serve.
    try:
        setattr(patched, "_cls", subclass)
    except Exception:  # noqa: BLE001
        pass

    llm_models._get_reasoning_chat_openai = patched  # type: ignore[assignment]
    llm_models._AE_RETRY_INSTALLED = True  # type: ignore[attr-defined]
    llm_models._AE_RETRY_ORIGINAL_FACTORY = factory  # type: ignore[attr-defined]
    if manager is not None and hasattr(manager, "clear_models"):
        manager.clear_models()  # drop instances built before the patch
    return True


def uninstall_retry_policy() -> None:
    """Restore the pristine factory. Mainly for test isolation."""
    from cuga.backend.llm import models as llm_models

    original = getattr(llm_models, "_AE_RETRY_ORIGINAL_FACTORY", None)
    if original is not None:
        llm_models._get_reasoning_chat_openai = original  # type: ignore[assignment]
        # Drop any subclass left in the original's cache so a restored
        # factory genuinely rebuilds the pristine class.
        try:
            original._cls = None
        except Exception:  # noqa: BLE001
            pass
    llm_models._AE_RETRY_INSTALLED = False  # type: ignore[attr-defined]
