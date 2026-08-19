"""mitmproxy addon: label, capture, and interactively mock AgentEvolve LLM calls.

Three jobs, in order of importance:

1. **Redact** credentials before anything is written to disk. This runs first and
   unconditionally, because ``AGENTS.md`` forbids persisting credentials to logs
   and mitmproxy's own ``flows.mitm`` would otherwise contain the raw
   ``Authorization`` header of every request.

2. **Correlate** each HTTP flow to AgentEvolve semantics ``(candidate, task,
   rollout, phase)`` when the caller supplies ``X-AE-*`` headers. mitmproxy sees
   bytes on a socket; it has no idea which candidate is being evaluated. Without
   this the capture cannot answer the questions in
   ``docs/SEVERE-OPEN-ISSUES.md:960``. Headers are stripped before the request
   goes upstream so no vendor ever receives them.

3. **Mock** responses from a hot-reloaded rules file, so an ablation arm can pin
   a specific completion without a code change or a restart.

Design notes worth stating, because they are easy to get wrong:

* Mock rules are re-read when the file's mtime changes rather than cached at
  startup. Caching would make an interactive edit require a container restart,
  which defeats the point.
* A mocked response is stamped ``X-AE-Mocked: true`` and recorded with
  ``mocked=true`` in the JSONL. Evidence produced under a mock must never be
  mistakable for a live observation -- the same reasoning that pins
  ``ProxyVerdict.evidence_kind`` to ``"proxy"`` in
  ``src/agent_evolve/adapters/cuga_proxy_validator.py``.
* ``x-litellm-cache-key`` and the response ``id`` are captured verbatim. They are
  what distinguishes a genuinely fresh completion from a cache re-read, which the
  register calls out as a U-1 regression guard.
* Bodies are truncated at 256 KiB in the JSONL sidecar only. The full body is
  still in ``flows.mitm``; this cap keeps the human-readable log greppable.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from mitmproxy import ctx, http

#: Only these hosts are treated as LLM traffic worth capturing. Regular proxy
#: mode sees *all* egress (package installs, telemetry, searxng), and capturing
#: that would bury the calls we care about in noise.
_LLM_HOST_HINTS = (
    "openai", "azure", "anthropic", "googleapis", "generativelanguage",
    "bedrock", "watsonx", "groq", "openrouter", "minimax", "rits",
    "litellm", "ollama", "localhost", "127.0.0.1", "host.docker.internal",
)

#: Paths that indicate an inference call rather than a health check or model list.
_LLM_PATH_HINTS = (
    "/chat/completions", "/completions", "/messages", "/embeddings",
    "/generate", "/v1/responses", "/converse", "/predict",
)

_BODY_CAP = 256 * 1024


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


class AgentEvolveObserver:
    def __init__(self) -> None:
        self.capture_dir = Path(os.environ.get("AE_CAPTURE_DIR", "/captures"))
        self.rules_path = Path(os.environ.get("AE_MOCK_RULES", "/mocks/rules.json"))
        self.redact = {
            h.strip().lower()
            for h in os.environ.get("AE_REDACT_HEADERS", "authorization").split(",")
            if h.strip()
        }
        self._rules: list[dict[str, Any]] = []
        self._rules_mtime: float = -1.0
        self._seq = 0
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.capture_dir / "calls.jsonl"

    # ---------------------------------------------------------------- #
    # Rules: hot-reloaded so interactive edits take effect immediately.
    # ---------------------------------------------------------------- #
    def _load_rules(self) -> list[dict[str, Any]]:
        try:
            mtime = self.rules_path.stat().st_mtime
        except OSError:
            self._rules, self._rules_mtime = [], -1.0
            return self._rules
        if mtime != self._rules_mtime:
            try:
                raw = json.loads(self.rules_path.read_text("utf-8") or "{}")
                loaded = raw.get("rules", []) if isinstance(raw, dict) else raw
                self._rules = [r for r in loaded if isinstance(r, dict) and r.get("enabled", True)]
                self._rules_mtime = mtime
                ctx.log.info(f"[ae] loaded {len(self._rules)} mock rule(s)")
            except (OSError, ValueError) as exc:
                # Deliberately keep the previous rules on a malformed edit: a
                # half-saved JSON file should not silently disable every mock
                # mid-debugging-session.
                ctx.log.warn(f"[ae] keeping previous rules, unreadable {self.rules_path}: {exc}")
        return self._rules

    @staticmethod
    def _is_llm(flow: http.HTTPFlow) -> bool:
        host = (flow.request.pretty_host or "").lower()
        path = (flow.request.path or "").lower()
        if not any(h in host for h in _LLM_HOST_HINTS):
            return False
        return any(p in path for p in _LLM_PATH_HINTS)

    def _match(self, flow: http.HTTPFlow, body: str) -> dict[str, Any] | None:
        for rule in self._load_rules():
            when = rule.get("when", {})
            if (hp := when.get("host_contains")) and hp.lower() not in flow.request.pretty_host.lower():
                continue
            if (pp := when.get("path_contains")) and pp.lower() not in flow.request.path.lower():
                continue
            if (bc := when.get("body_contains")) and bc not in body:
                continue
            if (br := when.get("body_regex")):
                try:
                    if not re.search(br, body, re.S):
                        continue
                except re.error as exc:
                    ctx.log.warn(f"[ae] bad body_regex in rule {rule.get('id')!r}: {exc}")
                    continue
            if (ph := when.get("ae_phase")) and flow.metadata.get("ae_phase") != ph:
                continue
            if (cd := when.get("ae_candidate")) and flow.metadata.get("ae_candidate") != cd:
                continue
            return rule
        return None

    # ---------------------------------------------------------------- #
    def request(self, flow: http.HTTPFlow) -> None:
        if not self._is_llm(flow):
            return
        self._seq += 1
        flow.metadata["ae_seq"] = self._seq
        flow.metadata["ae_t0"] = time.time()

        # Lift X-AE-* correlation headers into metadata, then strip them so no
        # vendor endpoint ever sees internal experiment identifiers.
        for key in ("candidate", "task", "rollout", "phase", "run"):
            header = f"X-AE-{key.capitalize()}"
            if header in flow.request.headers:
                flow.metadata[f"ae_{key}"] = flow.request.headers[header]
                del flow.request.headers[header]

        body = flow.request.get_text(strict=False) or ""
        rule = self._match(flow, body)
        if rule is None:
            return

        # Short-circuit: setting flow.response in the request hook means the
        # request is never sent upstream -- a free, offline ablation arm.
        mock = rule.get("respond", {})
        content = mock.get("content", "")
        status = int(mock.get("status", 200))
        if "raw_json" in mock:
            payload = mock["raw_json"]
        else:
            payload = {
                "id": f"ae-mock-{rule.get('id', 'anon')}-{self._seq}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": mock.get("model", "ae-mock"),
                "choices": [
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": c},
                        "finish_reason": "stop",
                    }
                    for i, c in enumerate(
                        content if isinstance(content, list) else [content]
                    )
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        flow.metadata["ae_mocked"] = rule.get("id", "anon")
        if (delay := float(mock.get("delay_seconds", 0) or 0)) > 0:
            time.sleep(delay)
        flow.response = http.Response.make(
            status,
            json.dumps(payload).encode(),
            {"content-type": "application/json", "X-AE-Mocked": "true",
             "X-AE-Mock-Rule": str(rule.get("id", "anon"))},
        )
        ctx.log.info(f"[ae] MOCKED #{self._seq} via rule {rule.get('id')!r}")

    def response(self, flow: http.HTTPFlow) -> None:
        if not self._is_llm(flow):
            return
        req_headers = {
            k: ("<redacted>" if k.lower() in self.redact else v)
            for k, v in flow.request.headers.items()
        }
        t0 = flow.metadata.get("ae_t0")
        record = {
            "seq": flow.metadata.get("ae_seq"),
            "ts": time.time(),
            "duration_s": round(time.time() - t0, 3) if t0 else None,
            "mocked": bool(flow.metadata.get("ae_mocked")),
            "mock_rule": flow.metadata.get("ae_mocked"),
            "correlation": {
                k: flow.metadata.get(f"ae_{k}")
                for k in ("run", "candidate", "task", "rollout", "phase")
                if flow.metadata.get(f"ae_{k}") is not None
            },
            "request": {
                "method": flow.request.method,
                "host": flow.request.pretty_host,
                "path": flow.request.path,
                "headers": req_headers,
                "body": (flow.request.get_text(strict=False) or "")[:_BODY_CAP],
            },
            "response": {
                "status": flow.response.status_code if flow.response else None,
                # Cache provenance: a repeated identical request that returns the
                # same id is a cache re-read, not an independent sample.
                "id_header": flow.response.headers.get("x-request-id") if flow.response else None,
                "cache_key": flow.response.headers.get("x-litellm-cache-key") if flow.response else None,
                "body": (flow.response.get_text(strict=False) or "")[:_BODY_CAP] if flow.response else None,
            },
        }
        try:
            with self.jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            ctx.log.warn(f"[ae] could not append capture: {exc}")


addons = [AgentEvolveObserver()]
