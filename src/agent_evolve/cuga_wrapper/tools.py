"""CUGA custom tools exposed by the wrapper.

Each tool is a plain, deterministic-in-shape function that returns a string and
converts internal failures into a structured JSON error string so a single
failing tool never crashes the whole agent execution. CUGA imports
(``tracked_tool``) are deferred into :func:`build_tools` so the SDK is not
imported until the wrapper explicitly constructs a runtime.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTES_PATH = PROJECT_ROOT / "data" / "cuga_notes.txt"


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARY = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}
_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in _NAMES:
            return _NAMES[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[visit(arg) for arg in node.args])
        raise ValueError("Unsupported calculator expression")

    result = visit(tree)
    if not isinstance(result, (int, float)) or not math.isfinite(float(result)):
        raise ValueError("Result is not finite")
    return result


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely and return its value."""
    try:
        return str(_safe_eval(expression))
    except Exception as exc:  # noqa: BLE001 - tool must not raise into the agent
        return _error(f"calculator failed: {exc}")


def _http_get(url: str, timeout: int = 20) -> tuple[int, str, str]:
    from urllib.request import Request, urlopen

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 CUGA-Research-Agent",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.status, response.headers.get("Content-Type", ""), response.read().decode(charset, errors="replace")


def _search_searxng(query: str, max_results: int) -> str | None:
    from urllib.parse import quote_plus

    base = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
    if not base:
        return None
    status, _, body = _http_get(f"{base}/search?q={quote_plus(query)}&format=json&language=en")
    if status != 200:
        raise RuntimeError(f"SearXNG HTTP {status}")
    data = json.loads(body)
    results = data.get("results", [])[:max_results]
    return "\n".join(
        f"{i}. {item.get('title', '')}\n   URL: {item.get('url', '')}\n   {item.get('content', '')}"
        for i, item in enumerate(results, 1)
    ) or "No SearXNG results found."


def _search_tavily(query: str, max_results: int) -> str | None:
    from urllib.request import Request, urlopen

    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return None
    payload = json.dumps(
        {
            "api_key": key,
            "query": query,
            "max_results": max_results,
            "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        }
    ).encode()
    request = Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode())
    results = data.get("results", [])[:max_results]
    return "\n".join(
        f"{i}. {item.get('title', '')}\n   URL: {item.get('url', '')}\n   {item.get('content', '')}"
        for i, item in enumerate(results, 1)
    ) or "No Tavily results found."


def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web using SearXNG, Tavily, or a DuckDuckGo fallback."""
    from urllib.parse import quote_plus

    try:
        count = max(1, min(int(max_results), 10))
        result = _search_searxng(query, count)
        if result is not None:
            return result
        result = _search_tavily(query, count)
        if result is not None:
            return result
        _, _, html = _http_get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}")
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.I | re.S
        )
        return "\n".join(
            f"{i}. {re.sub(r'<.*?>', '', title).strip()}\n   URL: {url}"
            for i, (url, title) in enumerate(blocks[:count], 1)
        ) or "No search results found."
    except Exception as exc:  # noqa: BLE001
        return _error(f"web_search failed: {exc}")


def web_fetch(url: str, max_chars: int = 12000) -> str:
    """Fetch a public HTTP/HTTPS page and return extracted text; JavaScript is not executed."""
    if not url.startswith(("http://", "https://")):
        return _error("web_fetch failed: URL must start with http:// or https://")
    try:
        status, content_type, body = _http_get(url)
        if "application/json" in content_type:
            try:
                text = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                text = body
        else:
            body = re.sub(r"<(script|style|noscript).*?>.*?</\1>", " ", body, flags=re.I | re.S)
            text = re.sub(r"<[^>]+>", " ", body)
            text = re.sub(r"\s+", " ", text).strip()
        return f"HTTP {status}\nURL: {url}\n\n{text[:max(1000, min(int(max_chars), 50000))]}"
    except Exception as exc:  # noqa: BLE001
        return _error(f"web_fetch failed: {exc}")


def wikipedia_search(query: str, max_results: int = 5) -> str:
    """Search Wikipedia for matching article titles, URLs, and short extracts."""
    from urllib.parse import quote_plus

    try:
        count = max(1, min(int(max_results), 10))
        api_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={quote_plus(query)}"
            f"&srlimit={count}&format=json&utf8=1"
        )
        status, _, body = _http_get(api_url)
        if status != 200:
            return _error(f"wikipedia_search failed: HTTP {status}")
        data = json.loads(body)
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return f"No Wikipedia articles found for: {query}"
        lines = []
        for i, hit in enumerate(hits, 1):
            title = hit.get("title", "")
            page_url = "https://en.wikipedia.org/wiki/" + quote_plus(title.replace(" ", "_"))
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            snippet = re.sub(r"&quot;", '"', snippet)
            snippet = re.sub(r"&#39;", "'", snippet)
            snippet = re.sub(r"&amp;", "&", snippet)
            snippet = re.sub(r"\s+", " ", snippet).strip()
            lines.append(f"{i}. {title}\n   URL: {page_url}\n   {snippet}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return _error(f"wikipedia_search failed: {exc}")


def save_note(note: str) -> str:
    """Save a small local research note and return its path."""
    try:
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with NOTES_PATH.open("a", encoding="utf-8") as handle:
            handle.write(note.rstrip() + "\n")
        return f"Saved note to {NOTES_PATH}"
    except Exception as exc:  # noqa: BLE001
        return _error(f"save_note failed: {exc}")


_TOOL_APP_NAMES = {
    "calculator": "calculator",
    "web_search": "web_search",
    "web_fetch": "web_fetch",
    "wikipedia_search": "wikipedia",
    "save_note": "research",
}

_RAW_TOOLS: tuple[Callable[..., str], ...] = (
    calculator,
    web_search,
    web_fetch,
    wikipedia_search,
    save_note,
)


def build_tools() -> list:
    """Construct the tracked LangChain tools, deferring the CUGA import.

    This is the only place the CUGA SDK is imported from the tools module.
    """
    from langchain_core.tools import tool

    from cuga import tracked_tool

    return [
        tool(tracked_tool(app_name=_TOOL_APP_NAMES[func.__name__])(func))
        for func in _RAW_TOOLS
    ]
