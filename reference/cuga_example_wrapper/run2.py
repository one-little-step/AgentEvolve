from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "data" / "traces"


def normalize_cuga_configuration_directory() -> None:
    """Treat a blank optional CUGA configuration directory as unset."""
    value = os.getenv("CUGA_CONFIGURATIONS_DIR")
    if value is not None and not value.strip():
        os.environ.pop("CUGA_CONFIGURATIONS_DIR", None)


# CUGA reads its configuration while importing the SDK, so these values must
# be set before importing CugaAgent.
load_dotenv(ROOT / ".env")
normalize_cuga_configuration_directory()
os.environ["AGENT_SETTING_CONFIG"] = Path(
    os.getenv("AGENT_SETTING_CONFIG", "settings.openai.toml")
).expanduser().name
skills_root = os.getenv("SKILLS_ROOT", ".cuga/skills")
if skills_root == "cuga":
    skills_root = ".cuga/skills"
skills_path = Path(skills_root).expanduser()
if not skills_path.is_absolute():
    skills_path = ROOT / skills_path
os.environ["SKILLS_ROOT"] = str(skills_path)

from cuga import CugaAgent, tracked_tool

from cuga.config import settings

if not settings.advanced_features.force_autonomous_mode:
    raise RuntimeError(
        "CUGA autonomous mode is disabled. "
        "Set DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true."
    )


def configure_environment() -> dict[str, Any]:
    load_dotenv(ROOT / ".env")

    model = os.getenv("CUGA_MODEL") or os.getenv("MODEL_NAME")
    if not model:
        raise RuntimeError(
            "Set CUGA_MODEL in .env, e.g. CUGA_MODEL=gpt-4o"
        )

    base_url = os.getenv("CUGA_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("CUGA_API_KEY") or os.getenv("OPENAI_API_KEY")

    os.environ["AGENT_SETTING_CONFIG"] = Path(
        os.getenv("AGENT_SETTING_CONFIG", "settings.openai.toml")
    ).expanduser().name
    os.environ["MODEL_NAME"] = model.removeprefix("openai/")

    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    if not Path(os.environ["SKILLS_ROOT"]).is_dir():
        raise FileNotFoundError(
            f"CUGA skills directory not found: {os.environ['SKILLS_ROOT']}"
        )

    mcp_file = os.getenv("MCP_SERVERS_FILE")
    if mcp_file:
        os.environ["MCP_SERVERS_FILE"] = str(Path(mcp_file).expanduser().resolve())

    # Optional observability switches.
    if os.getenv("LANGFUSE_TRACING", "").lower() == "true":
        os.environ["LANGFUSE_TRACING"] = "true"
    if os.getenv("OPENLIT", "").lower() == "true":
        os.environ["OPENLIT"] = "true"

    return {
        "model": model,
        "base_url": base_url,
        "configuration": os.environ["AGENT_SETTING_CONFIG"],
        "skills_root": os.environ["SKILLS_ROOT"],
        "mcp_servers_file": os.getenv("MCP_SERVERS_FILE"),
    }


# Custom tools: generic calculator + web search + web fetch.
#
# CUGA does not automatically inject generic web_search/web_fetch Python tools
# into CugaAgent; this wrapper supplies and tracks them explicitly.

def _safe_eval(expression: str):
    allowed_binops={__import__('ast').Add:lambda a,b:a+b,__import__('ast').Sub:lambda a,b:a-b,__import__('ast').Mult:lambda a,b:a*b,__import__('ast').Div:lambda a,b:a/b,__import__('ast').FloorDiv:lambda a,b:a//b,__import__('ast').Mod:lambda a,b:a%b,__import__('ast').Pow:lambda a,b:a**b}
    allowed_unary={__import__('ast').UAdd:lambda a:+a,__import__('ast').USub:lambda a:-a}
    names={'pi':__import__('math').pi,'e':__import__('math').e,'tau':__import__('math').tau}
    funcs={'sqrt':__import__('math').sqrt,'abs':abs,'round':round,'floor':__import__('math').floor,'ceil':__import__('math').ceil,'sin':__import__('math').sin,'cos':__import__('math').cos,'tan':__import__('math').tan,'log':__import__('math').log,'log10':__import__('math').log10,'exp':__import__('math').exp}
    ast=__import__('ast'); tree=ast.parse(expression,mode='eval')
    def visit(n):
        if isinstance(n,ast.Expression): return visit(n.body)
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
        if isinstance(n,ast.Name) and n.id in names: return names[n.id]
        if isinstance(n,ast.UnaryOp) and type(n.op) in allowed_unary: return allowed_unary[type(n.op)](visit(n.operand))
        if isinstance(n,ast.BinOp) and type(n.op) in allowed_binops: return allowed_binops[type(n.op)](visit(n.left),visit(n.right))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in funcs and not n.keywords: return funcs[n.func.id](*[visit(a) for a in n.args])
        raise ValueError('Unsupported calculator expression')
    r=visit(tree)
    if not isinstance(r,(int,float)) or not __import__('math').isfinite(float(r)): raise ValueError('Result is not finite')
    return r

@tool
@tracked_tool(app_name="calculator")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try: return str(_safe_eval(expression))
    except Exception as exc: return f"Calculator error: {exc}"

def _http_get(url: str, timeout: int = 20):
    from urllib.request import Request, urlopen
    req=Request(url,headers={'User-Agent':'Mozilla/5.0 CUGA-Research-Agent','Accept':'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8'})
    with urlopen(req,timeout=timeout) as r:
        charset=r.headers.get_content_charset() or 'utf-8'
        return r.status,r.headers.get('Content-Type',''),r.read().decode(charset,errors='replace')

def _search_searxng(query,max_results):
    from urllib.parse import quote_plus
    base=os.getenv('SEARXNG_URL','').strip().rstrip('/')
    if not base: return None
    status,_,body=_http_get(f'{base}/search?q={quote_plus(query)}&format=json&language=en')
    if status != 200: raise RuntimeError(f'SearXNG HTTP {status}')
    data=json.loads(body); results=data.get('results',[])[:max_results]
    return '\n'.join(f"{i}. {x.get('title','')}\n   URL: {x.get('url','')}\n   {x.get('content','')}" for i,x in enumerate(results,1)) or 'No SearXNG results found.'

def _search_tavily(query,max_results):
    key=os.getenv('TAVILY_API_KEY','').strip()
    if not key: return None
    from urllib.request import Request,urlopen
    payload=json.dumps({'api_key':key,'query':query,'max_results':max_results,'search_depth':os.getenv('TAVILY_SEARCH_DEPTH','basic')}).encode()
    req=Request('https://api.tavily.com/search',data=payload,headers={'Content-Type':'application/json','Accept':'application/json'},method='POST')
    with urlopen(req,timeout=30) as r: data=json.loads(r.read().decode())
    results=data.get('results',[])[:max_results]
    return '\n'.join(f"{i}. {x.get('title','')}\n   URL: {x.get('url','')}\n   {x.get('content','')}" for i,x in enumerate(results,1)) or 'No Tavily results found.'

@tool
@tracked_tool(app_name="web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web using SearXNG, Tavily, or DuckDuckGo fallback."""
    try:
        n=max(1,min(int(max_results),10)); result=_search_searxng(query,n)
        if result is not None: return result
        result=_search_tavily(query,n)
        if result is not None: return result
        from urllib.parse import quote_plus
        import re
        _,_,html=_http_get(f'https://html.duckduckgo.com/html/?q={quote_plus(query)}')
        blocks=re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',html,re.I|re.S)
        return '\n'.join(f"{i}. {re.sub(r'<.*?>','',title).strip()}\n   URL: {url}" for i,(url,title) in enumerate(blocks[:n],1)) or 'No search results found.'
    except Exception as exc: return f'Web search error: {exc}'

@tool
@tracked_tool(app_name="web_fetch")
def web_fetch(url: str, max_chars: int = 12000) -> str:
    """Fetch a public HTTP/HTTPS page and return extracted text; JavaScript is not executed."""
    if not url.startswith(('http://','https://')): return 'Web fetch error: URL must start with http:// or https://'
    try:
        status,ctype,body=_http_get(url); import re
        if 'application/json' in ctype:
            try: text=json.dumps(json.loads(body),indent=2,ensure_ascii=False)
            except Exception: text=body
        else:
            body=re.sub(r'<(script|style|noscript).*?>.*?</\1>',' ',body,flags=re.I|re.S)
            text=re.sub(r'<[^>]+>',' ',body); text=re.sub(r'\s+',' ',text).strip()
        return f'HTTP {status}\nURL: {url}\n\n{text[:max(1000,min(int(max_chars),50000))]}'
    except Exception as exc: return f'Web fetch error: {exc}'


@tool
@tracked_tool(app_name="research")
def save_note(note: str) -> str:
    """Save a small local research note for this demo run."""
    path = ROOT / "data" / "notes.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(note.rstrip() + "\n")
    return f"Saved note to {path}"


@tool
@tracked_tool(app_name="wikipedia")
def wikipedia_search(query: str, max_results: int = 5) -> str:
    """Search Wikipedia using the official English Wikipedia MediaWiki API.

    Returns matching article titles, URLs, and short search-result extracts.
    Use this for Wikipedia-specific factual/background lookup; use
    web_search for broader web research.
    """
    max_results = max(1, min(int(max_results), 10))

    api_url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={quote_plus(query)}"
        f"&srlimit={max_results}&format=json&utf8=1"
    )

    try:
        status, _, body = _http_get(api_url)
        if status != 200:
            return f"Wikipedia search error: HTTP {status}"

        data = json.loads(body)
        hits = data.get("query", {}).get("search", [])

        if not hits:
            return f"No Wikipedia articles found for: {query}"

        lines = []
        for i, hit in enumerate(hits, 1):
            title = hit.get("title", "")
            page_url = (
                "https://en.wikipedia.org/wiki/"
                + quote_plus(title.replace(" ", "_"))
            )
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            snippet = re.sub(r"&quot;", '"', snippet)
            snippet = re.sub(r"&#39;", "'", snippet)
            snippet = re.sub(r"&amp;", "&", snippet)
            snippet = re.sub(r"\s+", " ", snippet).strip()

            lines.append(
                f"{i}. {title}\n"
                f"   URL: {page_url}\n"
                f"   {snippet}"
            )

        return "\n".join(lines)
    except Exception as exc:
        return f"Wikipedia search error: {exc}"


CUSTOM_TOOLS=[calculator,web_search,web_fetch,wikipedia_search,save_note]


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return repr(value)


def extract_stream_result(streamed: list[Any]) -> dict[str, Any]:
    """Extract the final answer and tracked calls from one CUGA stream."""
    answer = ""
    error = None
    thread_id = None
    tool_calls: list[Any] = []
    seen_calls: set[str] = set()

    for event in streamed:
        state = event[1] if isinstance(event, (list, tuple)) and len(event) == 2 else event
        if not isinstance(state, dict):
            continue

        for node_state in state.values():
            if not isinstance(node_state, dict):
                continue

            if node_state.get("final_answer") is not None:
                answer = str(node_state["final_answer"])
            if node_state.get("answer") is not None:
                answer = str(node_state["answer"])
            if node_state.get("error") is not None:
                error = node_state["error"]
            if node_state.get("thread_id") is not None:
                thread_id = node_state["thread_id"]

            for call in node_state.get("tool_calls") or []:
                safe_call = json_safe(call)
                fingerprint = json.dumps(safe_call, sort_keys=True, default=repr)
                if fingerprint not in seen_calls:
                    seen_calls.add(fingerprint)
                    tool_calls.append(safe_call)

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "error": json_safe(error),
        "thread_id": json_safe(thread_id),
    }


class CugaExperiment:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.agent: CugaAgent | None = None

    async def create_agent(self) -> CugaAgent:
        # These are documented CugaAgent constructor surfaces.
        #
        # Knowledge is enabled explicitly because the SDK exposes
        # agent.knowledge only when enable_knowledge=True.
        #
        # Skills are configured through CUGA's [skills] settings and
        # .cuga/skills root; current SDK docs do not list an
        # enable_skills constructor parameter, so we do NOT invent one.
        self.agent = CugaAgent(
            tools=CUSTOM_TOOLS,
            special_instructions=(
                """
                You are an autonomous general-purpose agent.

                Solve the user's task carefully and accurately.
                Use the available tools when they are useful.
                For calculations, use the calculator tool.
                For web research, use web_search and web_fetch.
                For Wikipedia-specific information, use wikipedia_search.

                Do not claim to have performed an action or accessed information
                unless you actually did so.

                Return the best final answer to the user's question.
                """
            ),
            enable_knowledge=True,
        )
        return self.agent

    async def run(self, prompt: str) -> dict[str, Any]:
        agent = await self.create_agent()

        run_id = str(uuid.uuid4())
        started = datetime.now(timezone.utc).isoformat()
        events: list[dict[str, Any]] = []
        streamed: list[Any] = []

        try:
            # A CUGA stream contains the full agent execution and final state.
            # Do not invoke again: a second call repeats tool side effects.
            async for state in agent.stream(prompt):
                item = json_safe(state)
                streamed.append(item)
                events.append({
                    "kind": "stream_event",
                    "index": len(streamed) - 1,
                    "state": item,
                })

            stream_result = extract_stream_result(streamed)
            tool_calls = stream_result["tool_calls"]

            for index, call in enumerate(tool_calls):
                events.append({
                    "kind": "tool_call",
                    "index": index,
                    "tool_call": call,
                })

            output = {
                "run_id": run_id,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": self.config,
                "runtime": {
                    "cuga_agent": "CugaAgent",
                    "knowledge_enabled": True,
                    "custom_tools": [t.name for t in CUSTOM_TOOLS],
                    "reference_tools": ["wikipedia_search"],
                    "skill_root": os.getenv("SKILLS_ROOT"),
                    "mcp_servers_file": os.getenv("MCP_SERVERS_FILE"),
                },
                "prompt": prompt,
                "events": events,
                "stream_events": streamed,
                "tool_calls": tool_calls,
                "answer": stream_result["answer"],
                "error": stream_result["error"],
                "thread_id": stream_result["thread_id"],
                "note": (
                    "This is an event/tool trajectory. Exact replayable "
                    "LangGraph checkpoints require a configured checkpointer "
                    "and direct use of agent.graph."
                ),
            }
            return output
        finally:
            await agent.aclose()

    async def knowledge_demo(self) -> None:
        agent = await self.create_agent()
        try:
            # Demonstrates that the KnowledgeManager is exposed.
            docs = await agent.knowledge.list_documents()
            print(json.dumps(json_safe(docs), indent=2))
        finally:
            await agent.aclose()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="*", default=["what tools and skills are available?"])
    parser.add_argument(
        "--knowledge-list",
        action="store_true",
        help="List documents through CUGA's KnowledgeManager and exit.",
    )
    args = parser.parse_args()

    config = configure_environment()
    experiment = CugaExperiment(config)

    if args.knowledge_list:
        await experiment.knowledge_demo()
        return 0

    prompt = " ".join(args.prompt).strip()
    trace = await experiment.run(prompt)

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{trace['run_id']}.json"
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "run_id": trace["run_id"],
        "answer": trace["answer"],
        "tool_calls": trace["tool_calls"],
        "trace": str(path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
