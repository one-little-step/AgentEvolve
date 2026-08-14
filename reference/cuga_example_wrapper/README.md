# CUGA Complete Wrapper

A runnable CUGA SDK wrapper for experiments with:

- CUGA's normal `CugaAgent` runtime
- custom LangChain tools
- tracked tools
- CUGA Knowledge/RAG
- CUGA Agent Skills via `.cuga/skills`
- CUGA policies
- streaming execution
- tool-call trace capture
- JSON trajectory persistence
- optional MCP registry configuration
- OpenAI-compatible / LiteLLM model endpoints
- an adapter boundary suitable for later causal/counterfactual replay

## Requirements

- Python 3.12+
- CUGA installed and importable (`python -c "from cuga import CugaAgent; print('ok')"`)
- `python-dotenv`
- `langchain-core`

The CUGA documentation currently recommends Python 3.12+ and `uv`. See:
https://docs.cuga.dev/docs/getting-started/installation/

## Setup

From this directory:

```bash
cp .env.example .env
# edit .env
```

For OpenAI:

```env
CUGA_MODEL=gpt-4o
OPENAI_API_KEY=...
```

For an OpenAI-compatible/LiteLLM endpoint:

```env
CUGA_MODEL=your-model
CUGA_BASE_URL=http://localhost:4000/v1
CUGA_API_KEY=your-key
```

The wrapper maps these onto CUGA's OpenAI configuration.

## Run

```bash
python run.py "what tools and skills do you have?"
```

Try:

```bash
python run.py "multiply 7 by 9"
python run.py "use the web research skill to explain how to verify a source"
python run.py "remember that the baseline harness uses version B0"
```

A trace is written to:

```text
data/traces/<run_id>.json
```

## Important capability distinction

The SDK constructor gives you the high-level `CugaAgent` interface. CUGA's documented Agent Skills system is configured through the CUGA settings and `.cuga/skills/<name>/SKILL.md`, not by passing arbitrary skill dictionaries to `CugaAgent`.

CUGA's browser-only/web and MCP/OpenAPI registry facilities are runtime/configuration capabilities. They are not equivalent to automatically adding `web_search` and `web_fetch` Python tools to `CugaAgent`.

This wrapper therefore:
1. enables Knowledge through the SDK;
2. exposes custom tracked LangChain tools;
3. places a real `SKILL.md` under `.cuga/skills`;
4. records tool calls and streamed events;
5. supports MCP registry configuration through `MCP_SERVERS_FILE` when you supply one;
6. does not pretend that browser/MCP services are available if they are not configured.

## Causal/counterfactual research note

The JSON trace is deliberately separated into:
- run metadata
- configuration/artifact versions
- streamed events
- tool calls
- final result

It is **not claimed to be a perfect deterministic checkpoint/replay snapshot**. For exact counterfactual replay, use the exposed `agent.graph` and a verified LangGraph checkpointer/state-history configuration, then extend `CugaRuntime` accordingly.

