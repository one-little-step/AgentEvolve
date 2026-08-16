do cuga support 'temperature config' ?

Yes. **CUGA supports temperature configuration.** In the current SDK/configuration, temperature is an agent/model setting exposed under the `[agent]` configuration section. The official docs show examples such as `temperature = 0.1` for Fast mode and `temperature = 0.05` for Accurate mode. ([CUGA Documentation][1])

### 1. Through TOML configuration

For example:

```toml
[agent]
temperature = 0.2
max_tokens = 4000
```

CUGA's CLI also exposes this setting directly:

```bash
cuga config set agent.temperature 0.2
```

and you can inspect it with:

```bash
cuga config show --section agent
```

([CUGA Documentation][2])

### 2. Through execution modes

CUGA's built-in modes already set different temperatures. For example, the current docs show:

```toml
# fast.toml
[agent]
temperature = 0.1
```

and:

```toml
# accurate.toml
[agent]
temperature = 0.05
```

([CUGA Documentation][1])

So changing:

```bash
cuga mode fast
```

versus:

```bash
cuga mode accurate
```

can change temperature indirectly through the selected configuration.

### 3. Programmatically through `CugaAgent`

There is an important nuance here.

The current `CugaAgent` constructor accepts a **`model`** parameter:

```python
agent = CugaAgent(
    tools=tools,
    model=model,
)
```

where `model` is a `BaseChatModel`. ([CUGA Documentation][3])

So you can also configure temperature at the model layer, depending on the model/provider you're using. For example, with a LangChain chat model:

```python
from langchain_openai import ChatOpenAI
from cuga import CugaAgent

model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.2,
)

agent = CugaAgent(
    tools=tools,
    model=model,
)
```

That is slightly different from CUGA's TOML-level:

```toml
[agent]
temperature = 0.2
```

because the latter is part of CUGA's configuration system, whereas the former explicitly constructs the underlying chat model with a temperature.

### 4. Very important for your experiments

Since you're comparing **vanilla CUGA vs your self-evolving CUGA**, don't overlook that CUGA has **multiple model layers**.

The current CUGA app documentation explicitly describes an **outer model** plus CUGA's internal nodes such as planner, coder, and final-answer components, with those internal components configured through the settings TOML. ([GitHub][4])

So you can conceptually have:

```text
                    CUGA
                      │
          ┌───────────┴───────────┐
          │                       │
     Outer Agent Model       Internal nodes
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 Planner      Coder     Final Answer
                    │           │           │
              temperature   temperature  temperature
```

This means **"CUGA temperature" isn't necessarily one single knob for every LLM call**.

For a rigorous benchmark comparison, I'd record at least:

```text
model
temperature
max_tokens
execution mode
planner configuration
executor/coder configuration
```

Otherwise an apparent gain from your evolutionary harness could actually be coming from a model-generation parameter change.

So, in your setup, **yes: temperature is configurable, and `agent.temperature` is the main CUGA configuration knob; but the exact effect across all CUGA internal LLM nodes depends on the model/configuration path you're using.** ([CUGA Documentation][2])

[1]: https://docs.cuga.dev/docs/usage/modes/?utm_source=chatgpt.com "Execution Modes | CUGA AGENT"
[2]: https://docs.cuga.dev/docs/usage/control-commands/?utm_source=chatgpt.com "Control Commands | CUGA AGENT"
[3]: https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com "CugaAgent | CUGA AGENT"
[4]: https://github.com/cuga-project/cuga-apps?utm_source=chatgpt.com "GitHub - cuga-project/cuga-apps: A showcase of conversational and pipeline apps built on CUGA — a planner / executor agent runtime. Each app is a single-file FastAPI server wrapping a CugaAgent with a tool list and a system prompt; the right-hand panel of every app shows live structured state pushed from the agent. · GitHub"
