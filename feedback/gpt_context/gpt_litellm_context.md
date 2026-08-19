do we have any direct docker image for it (rather than building it) ? (bcoz, cuga-sdk agent dont support temperature parameter internally ,...so we need a litellm proxy for that)...

Yes — **you do not need to build LiteLLM yourself**. LiteLLM publishes official Docker images, and the official docs explicitly support running the proxy in Docker. ([LiteLLM][1])

For your use case, I would use the **GHCR image**:

```bash
docker pull ghcr.io/berriai/litellm:main-stable
```

GHCR is particularly useful on Apple Silicon/ARM because the published GHCR images include `linux/arm64`; Docker Hub's images have had ARM availability limitations. ([GitHub][2])

### Minimal setup for your CUGA case

Create:

```text
litellm/
├── config.yaml
└── .env
```

`config.yaml`:

```yaml
model_list:
  - model_name: cuga-model
    litellm_params:
      model: openai/your-upstream-model
      api_base: os.environ/UPSTREAM_API_BASE
      api_key: os.environ/UPSTREAM_API_KEY
```

`.env`:

```env
UPSTREAM_API_BASE=https://your-upstream-provider.com/v1
UPSTREAM_API_KEY=sk-your-upstream-key
LITELLM_MASTER_KEY=sk-litellm-local
```

Then:

```bash
docker run --rm \
  -p 4000:4000 \
  --env-file .env \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  ghcr.io/berriai/litellm:main-stable \
  --config /app/config.yaml \
  --port 4000
```

Now you have:

```text
CUGA SDK
   │
   │ OpenAI-compatible
   ▼
localhost:4000
   │
   │ LiteLLM
   ▼
your OpenAI-compatible API
```

And your CUGA wrapper can point its OpenAI client/base URL at:

```text
http://localhost:4000/v1
```

### About your temperature problem

This is actually a good reason to put LiteLLM in the middle.

You can have:

```text
CUGA
  │
  │ request
  │ model=foo
  │ messages=...
  ▼
LiteLLM Proxy
  │
  │ modify/route request
  │ temperature=0.7
  ▼
Upstream API
```

However, there's an important distinction:

**If CUGA doesn't send a `temperature` field at all**, simply putting LiteLLM in front does **not automatically mean LiteLLM will invent one**. We need to configure LiteLLM so that it injects the parameter or use a request hook/custom callback to modify the request.

That is probably the part you actually want for your CUGA experiment.

For example, conceptually:

```text
CUGA SDK
   ↓
LiteLLM
   ↓
add temperature=0.7
   ↓
upstream OpenAI-compatible endpoint
```

LiteLLM also gives you proxy hooks for request processing, logging, auth, rate limiting, etc. ([LiteLLM][1])

And since you're using this as an experimental **CUGA → self-evolving harness** layer, I'd keep the proxy completely separate from your harness code:

```text
┌─────────────────────┐
│   CUGA Agent        │
│   (unchanged)       │
└──────────┬──────────┘
           │ OpenAI API
           ▼
┌─────────────────────┐
│   LiteLLM Docker    │
│                     │
│  request mutation   │
│  routing            │
│  logging            │
└──────────┬──────────┘
           │ OpenAI API
           ▼
┌─────────────────────┐
│ Upstream LLM API    │
└─────────────────────┘
```

**This lets you keep vanilla CUGA unchanged while controlling things like `temperature` outside the SDK**, which is ideal for comparing your vanilla and self-evolving versions.

One thing I'd change from my previous answer: use a **pinned stable LiteLLM tag** rather than `main-stable` for reproducible experiments. LiteLLM publishes versioned images such as `v1.90.0` and stable releases on GHCR. ([GitHub][3])

I can give you a **ready-to-run `docker-compose.yml + config.yaml` that specifically injects `temperature` into every CUGA request**.

[1]: https://docs.litellm.ai/?utm_source=chatgpt.com "LiteLLM - Getting Started | liteLLM"
[2]: https://github.com/BerriAI/litellm/discussions/19348?utm_source=chatgpt.com "Dockerhub doesn't have ARM images · BerriAI litellm · Discussion #19348 · GitHub"
[3]: https://github.com/berriai/litellm/pkgs/container/litellm/979860960?tag=latest&utm_source=chatgpt.com "litellm versions · berriai · GitHub"


Yes — **the official LiteLLM Docker image supports those customizations**. The Docker image isn't a stripped-down forwarding proxy.

The important distinction is:

| Customization               | Official Docker image                               |
| --------------------------- | --------------------------------------------------- |
| Set `temperature`           | ✅                                                   |
| Modify request parameters   | ✅                                                   |
| Modify prompts/messages     | ✅                                                   |
| Logging                     | ✅                                                   |
| Callbacks/hooks             | ✅                                                   |
| Custom Python callback code | ✅, by mounting/copying your code into the container |
| Routing/fallbacks           | ✅                                                   |
| Authentication              | ✅                                                   |
| Usage/cost tracking         | ✅                                                   |

For your CUGA case, you can run:

```text
CUGA SDK
   ↓
LiteLLM official Docker image
   ↓
custom callback/hook
   ├── temperature=0.7
   ├── prompt modification
   └── logging
   ↓
OpenAI-compatible upstream API
```

The only caveat is that **the stock Docker image does not magically perform arbitrary prompt rewriting just because you run it**. For simple configuration, you use `config.yaml`; for arbitrary logic, you mount your own Python callback/plugin into the container.

For example, conceptually:

```bash
docker run \
  -p 4000:4000 \
  -v ./config.yaml:/app/config.yaml \
  -v ./callbacks:/app/callbacks \
  ghcr.io/berriai/litellm:main-stable \
  --config /app/config.yaml
```

So **yes, you can use the official prebuilt Docker image and still customize its behavior**. You don't need to fork/build LiteLLM just to add your CUGA-specific middleware.

For your particular requirement — **CUGA doesn't expose `temperature`, but you want the proxy to inject it before forwarding** — a custom LiteLLM callback is the clean approach.


can you show me how to do it , ... and how to modify prompts (for eg., i modify the cuga wrapper and wrap the temperaure inside a unique tag (say , <tempvnidsi> 0.7 </tempvnidsi>>) , ...then our litellm proxy cut and extract it ... and the fwd with that right temperature.... , do  that docker img allow that ? , how to do it ?

Yes. **Your `<tempvnidsi>...</tempvnidsi>` idea is a good fit for a LiteLLM pre-call hook**, and you can do it while still using the **official prebuilt Docker image**—you only mount your small Python hook into the container. LiteLLM's current proxy supports `async_pre_call_hook`, whose `data` argument can be modified before the upstream call. ([GitHub][1])

For your CUGA setup, I would structure it like this:

```text
CUGA wrapper
    │
    │  messages include:
    │  <tempvnidsi>0.7</tempvnidsi>
    ▼
┌──────────────────────────────┐
│ LiteLLM Proxy Docker         │
│                              │
│ async_pre_call_hook          │
│   1. scan messages           │
│   2. extract temperature     │
│   3. remove control tag      │
│   4. optionally modify prompt│
│   5. data["temperature"]=0.7 │
└──────────────┬───────────────┘
               │
               │ clean request
               │ temperature=0.7
               ▼
       OpenAI-compatible API
```

## 1. Directory

Create:

```text
litellm-proxy/
├── config.yaml
├── custom_hook.py
└── .env
```

## 2. `custom_hook.py`

Here's a working version of the idea:

```python
import re
import sys

from litellm.integrations.custom_logger import CustomLogger


TEMP_PATTERN = re.compile(
    r"<tempvnidsi>\s*([0-9]*\.?[0-9]+)\s*</tempvnidsi>",
    re.IGNORECASE,
)


class CUGAGatewayHook(CustomLogger):

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type,
        **kwargs,
    ):
        """
        Runs before LiteLLM sends the request upstream.

        We look for:
            <tempvnidsi>0.7</tempvnidsi>

        anywhere inside the message contents.
        """

        messages = data.get("messages", [])

        for message in messages:
            content = message.get("content")

            # Normal OpenAI message
            if isinstance(content, str):
                match = TEMP_PATTERN.search(content)

                if match:
                    temperature = float(match.group(1))

                    # Validate
                    if not 0.0 <= temperature <= 2.0:
                        raise ValueError(
                            f"Invalid temperature: {temperature}. "
                            "Expected value between 0 and 2."
                        )

                    # Tell LiteLLM what temperature to send upstream
                    data["temperature"] = temperature

                    # Remove our private control tag
                    message["content"] = TEMP_PATTERN.sub("", content)

                    print(
                        f"[CUGA-HOOK] temperature={temperature}",
                        file=sys.stderr,
                    )

        return data


proxy_handler_instance = CUGAGatewayHook()
```

The important part is:

```python
data["temperature"] = temperature
```

because `data` is the request LiteLLM will continue processing after the pre-call hook. ([GitHub][1])

---

# 3. `config.yaml`

For an OpenAI-compatible upstream:

```yaml
model_list:
  - model_name: cuga-model
    litellm_params:
      model: openai/your-upstream-model
      api_base: os.environ/UPSTREAM_API_BASE
      api_key: os.environ/UPSTREAM_API_KEY

litellm_settings:
  callbacks:
    - custom_hook.proxy_handler_instance

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

The callback registration pattern is supported by LiteLLM's proxy configuration. ([GitHub][1])

---

# 4. `.env`

```env
UPSTREAM_API_BASE=https://your-provider.example.com/v1
UPSTREAM_API_KEY=sk-your-upstream-key

LITELLM_MASTER_KEY=sk-cuga-proxy
```

---

# 5. Run the official Docker image

The current LiteLLM docs show the prebuilt image as:

```text
docker.litellm.ai/berriai/litellm:main-latest
```

and mount the config into `/app/config.yaml`. ([LiteLLM][2])

Run:

```bash
docker run --rm \
  -p 4000:4000 \
  --env-file .env \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  -v "$(pwd)/custom_hook.py:/app/custom_hook.py" \
  docker.litellm.ai/berriai/litellm:main-latest \
  --config /app/config.yaml \
  --detailed_debug
```

Notice what's happening:

```text
Official LiteLLM image
        +
your config.yaml
        +
your custom_hook.py
```

You **aren't building LiteLLM**.

You're simply injecting your customization into the official container.

---

# 6. What your CUGA wrapper sends

Suppose your modified CUGA wrapper produces:

```json
{
  "model": "cuga-model",
  "messages": [
    {
      "role": "system",
      "content": "You are a coding agent."
    },
    {
      "role": "user",
      "content": "<tempvnidsi>0.7</tempvnidsi>\nFix this bug."
    }
  ]
}
```

LiteLLM receives that.

Your hook sees:

```text
<tempvnidsi>0.7</tempvnidsi>
```

extracts:

```python
temperature = 0.7
```

then changes the request to effectively:

```json
{
  "model": "cuga-model",
  "temperature": 0.7,
  "messages": [
    {
      "role": "system",
      "content": "You are a coding agent."
    },
    {
      "role": "user",
      "content": "\nFix this bug."
    }
  ]
}
```

And **that cleaned request is what LiteLLM sends to the upstream provider**.

---

# 7. You can use the same mechanism for prompt modifications

This is where your idea becomes much more powerful.

For example, suppose CUGA emits:

```text
<tempvnidsi>0.7</tempvnidsi>
<promptvnidsi>
Always verify tool results before claiming success.
</promptvnidsi>

Fix the bug.
```

Your hook could extract both.

For example:

```python
TEMP_PATTERN = re.compile(
    r"<tempvnidsi>\s*([0-9]*\.?[0-9]+)\s*</tempvnidsi>",
    re.IGNORECASE,
)

PROMPT_PATTERN = re.compile(
    r"<promptvnidsi>(.*?)</promptvnidsi>",
    re.IGNORECASE | re.DOTALL,
)
```

Then:

```python
prompt_match = PROMPT_PATTERN.search(content)

if prompt_match:
    injected_prompt = prompt_match.group(1).strip()

    # Remove private control tag
    content = PROMPT_PATTERN.sub("", content)

    # Modify the actual system prompt
    system_message = next(
        (
            m
            for m in messages
            if m.get("role") == "system"
        ),
        None,
    )

    if system_message is not None:
        system_message["content"] += (
            "\n\n" + injected_prompt
        )
    else:
        messages.insert(
            0,
            {
                "role": "system",
                "content": injected_prompt,
            },
        )
```

So you can have CUGA secretly communicate **gateway control information** to your proxy through the otherwise-standard OpenAI request.

---

# 8. I would actually make the protocol slightly richer

Since you're building this for your **self-evolving CUGA harness**, I'd make the tags namespaced:

```text
<cuga:temperature>0.7</cuga:temperature>
<cuga:system_append>...</cuga:system_append>
<cuga:model>model-x</cuga:model>
<cuga:experiment>evo-v17</cuga:experiment>
```

Then the gateway becomes:

```text
CUGA
 │
 │ OpenAI request
 │
 ├── <cuga:temperature>0.7</cuga:temperature>
 ├── <cuga:system_append>...</cuga:system_append>
 └── <cuga:experiment>evo-v17</cuga:experiment>
 │
 ▼
LiteLLM Hook
 │
 ├── extract control metadata
 ├── remove tags
 ├── rewrite request
 ├── attach logging metadata
 │
 ▼
Provider
```

That gives you a **control plane embedded inside the CUGA request**, while the upstream model never sees the control tags.

---

## 9. Even better: put experiment information in metadata

For your research setup, you could have CUGA send:

```json
{
  "model": "cuga-model",
  "messages": [...],
  "metadata": {
    "evolution_version": "v17",
    "candidate_id": "candidate_42",
    "temperature_control": "<tempvnidsi>0.7</tempvnidsi>"
  }
}
```

Then your hook can manipulate the request based on `metadata` without putting control tokens into the actual prompt.

But your **tag approach is perfectly workable**, especially if modifying the CUGA wrapper is easy and you want a very lightweight protocol.

---

## 10. One important limitation

Your hook should target the **OpenAI `/v1/chat/completions` path** you're planning to use. Current LiteLLM reports that `async_pre_call_hook` works on the OpenAI chat-completions path, while there are known issues where the hook can be bypassed on some other endpoints such as Anthropic `/v1/messages`. ([GitHub][3])

So for your architecture:

```text
CUGA
 ↓
POST /v1/chat/completions
 ↓
LiteLLM
 ↓
async_pre_call_hook
 ↓
upstream OpenAI-compatible /v1/chat/completions
```

is the path I'd use.

### The nice part for your experiment

You can now keep **vanilla CUGA almost completely untouched**:

```text
CUGA
  ↓
LiteLLM
  ↓
[temperature / prompt / logging / routing manipulation]
  ↓
LLM
```

and your experimental CUGA version only needs to emit those control tags. That makes the **gateway layer independently swappable** and keeps the comparison between vanilla and evolved harnesses much cleaner. LiteLLM itself already provides proxy hooks for logging, auth, cost tracking and rate limiting in addition to the custom pre-call hook. ([LiteLLM][2])

[1]: https://github.com/BerriAI/litellm/discussions/15012?utm_source=chatgpt.com "How to add extra_body in litellm proxy before going to upstream LLM ? · BerriAI litellm · Discussion #15012 · GitHub"
[2]: https://docs.litellm.ai/?utm_source=chatgpt.com "LiteLLM - Getting Started | liteLLM"
[3]: https://github.com/BerriAI/litellm/issues/27518?utm_source=chatgpt.com "[Bug]: Proxy-level callbacks (`async_pre_call_hook`) are bypassed on Anthropic endpoint (`/v1/messages`) · Issue #27518 · BerriAI/litellm · GitHub"
