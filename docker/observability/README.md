# STEP 0 — realtime interactive LLM interception

Intercept, inspect, and **mock every LLM call in realtime** for debugging and
ablation studies. This is the "STEP 0" prerequisite node in
`docs/SEVERE-OPEN-ISSUES.md` (it is *not* a numbered SV issue).

```bash
./docker/observability/proxy.sh up                    # start
open http://127.0.0.1:8083                            # UI, password: agentevolve
./docker/observability/proxy.sh run -- python scripts/run_evolution.py ...
./docker/observability/proxy.sh tail                  # follow captured calls
./docker/observability/proxy.sh down
```

## Why these tools

**mitmproxy, not a bespoke shim.** Pausing an in-flight response, editing the
body, and resuming is mitmproxy's `intercept` feature. `--server-replay` gives
deterministic replay from a saved flow file, which also sidesteps the upstream
response cache problem documented in `adapters/cuga_proxy_validator.py`: an
identical repeated request *re-reads one observation* rather than sampling a
second one, so any confidence built by repeating an identical A/B is invalid.
Replaying a recorded flow makes ablation arms reproducible without depending on
upstream cache behaviour at all.

**Regular proxy mode, not reverse mode.** Reverse mode only captures callers that
honour `LITELLM_BASE_URL`. CUGA ships its own per-agent model config
(`cuga/configurations/models/settings.*.toml`, each with its own `url=`), and
`cuga_wrapper/__init__.py:457` exports `OPENAI_BASE_URL` only when `base_url` is
truthy. Reverse mode would capture *our* calls while silently missing
CUGA-internal ones — the precise blind spot SV-7/SV-8/SV-11 need closed. Partial
capture would defeat the purpose, so the CA-trust cost is worth paying.

**Port 8082/8083, not 8080.** searxng already holds 8080
(`reference/cuga_example_wrapper/searxng/docker-compose.yml`).

## Interactive mocking

Edit `mocks/rules.json` and save — rules are re-read on mtime change, **no
restart**. First match wins; all `when` keys present must hold.

```json
{"rules":[{"id":"pin-judge","enabled":true,
  "when":{"path_contains":"/chat/completions","body_contains":"preference"},
  "respond":{"content":"{\"winner\":\"candidate\",\"score\":0.8}"}}]}
```

| `when` | matches |
| --- | --- |
| `host_contains`, `path_contains`, `body_contains` | case-insensitive substring |
| `body_regex` | `re.search`, DOTALL |
| `ae_phase`, `ae_candidate` | requires caller to send `X-AE-*` headers |

| `respond` | effect |
| --- | --- |
| `content` | string, or list of strings for multiple choices (`n>1`) |
| `raw_json` | full response object, bypassing the chat-completion envelope |
| `status` | e.g. `429`/`500` for failure-path ablations |
| `delay_seconds` | latency injection for timeout paths |

A matched rule sets the response **in the request hook**, so the call never
reaches upstream — a free, offline, zero-cost ablation arm.

## Correlation

mitmproxy sees bytes on a socket; it has no idea which candidate is being
evaluated. Send `X-AE-Candidate`, `X-AE-Task`, `X-AE-Rollout`, `X-AE-Phase`,
`X-AE-Run` and the addon lifts them into the capture record, then **strips them
before the request goes upstream** so no vendor receives internal identifiers.

Without these, captures cannot answer the questions in the STEP 0 table at
`docs/SEVERE-OPEN-ISSUES.md` — which is why the structured layer exists.

## Verified behaviour

Measured live against `ete-litellm.ai-models.vpc-int.res.ibm.com`:

| Claim | Result |
| --- | --- |
| Live call intercepted through CA trust | `PROXY_OK`, status 200 in 1.65s |
| Correlation captured | `{candidate: cand-A, task: t1, rollout: 0, phase: rollout}` |
| `X-AE-*` stripped before upstream | none present on forwarded request |
| `Authorization` redacted in capture | `<redacted>`, no leak |
| Hot-reload mock, no restart | `MOCKED_RESPONSE_NO_UPSTREAM`, id `ae-mock-...` |
| Mock marked in audit trail | `mocked=True rule=live-test-mock` |

Reproduce: `terminal_output/proxy_step0/verify.log`.

## Evidence integrity

Mocked responses carry `X-AE-Mocked: true` and are logged with `mocked=true`.
Evidence produced under a mock must never be mistakable for a live observation —
the same reasoning that pins `ProxyVerdict.evidence_kind` to `"proxy"` in
`adapters/cuga_proxy_validator.py`. **Never** feed a mocked run into a measured
baseline comparison.

`x-litellm-cache-key` and the response `id` are captured verbatim: a repeated
identical request returning the same `id` is a cache re-read, not an independent
sample (U-1 regression guard).

## Not committed, deliberately

`certs/` holds the CA **private key** — a committed CA key lets anyone MITM any
host that trusts it. `captures/` holds verbatim prompts and completions, which
would persist task content and evaluator internals that `AGENTS.md` forbids
writing to shared artifacts. Both are gitignored; `mocks/rules.json` is local
scratch state with `mocks/rules.example.json` as the committed template.

## Optional structured layer

```bash
./docker/observability/proxy.sh structured    # adds litellm on :4000
```

Adds model routing, token cost, and cache-key visibility that raw HTTP capture
cannot provide. Needs `LITELLM_API_KEY`/`LITELLM_BASE_URL`. Not required for
interactive debugging — mitmproxy alone covers that.

## Known limits

- Addon log lines (`[ae] ...`) go to the **mitmweb event log**, not container
  stdout; `docker compose logs` is empty by design. Use the UI or `calls.jsonl`.
- Host/path filters (`_LLM_HOST_HINTS`, `_LLM_PATH_HINTS` in `addons/correlate.py`)
  keep package installs and telemetry out of the capture. A new provider whose
  host does not match will be proxied but **not captured** — add the hint.
- JSONL bodies are capped at 256 KiB; the full body is still in `flows.mitm`.
- Whether CUGA has an internal client that ignores `HTTPS_PROXY` is **not yet
  verified**. Regular mode makes complete capture *possible*, not *proven*.
