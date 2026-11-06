"""Live web search for fresh information (local model + free web search).

Providers (set WEB_PROVIDER in config.py):
  "ddgs"   - DuckDuckGo, no API key needed (pip install ddgs)
  "tavily" - needs TAVILY_API_KEY in .env (pip install tavily-python)

Steps: (1) optionally rewrite the question into a search query, (2) search,
(3) read the top page(s), (4) build a prompt that makes the model write a real answer.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import date

from dotenv import load_dotenv

import config
from core.llm import chat_once

load_dotenv()

KEY_NAMES = {"tavily": "TAVILY_API_KEY"}


def key_problem() -> str | None:
    """Return a message if the chosen provider needs a key that is missing."""
    name = KEY_NAMES.get(config.WEB_PROVIDER)
    if name and not os.getenv(name):
        return f"Add {name} to your .env file to use web search ({config.WEB_PROVIDER})."
    return None


def make_query(model: str, question: str) -> str:
    """Turn a chat question into a short search query (falls back to the question)."""
    if not config.WEB_REWRITE_QUERY:
        return question
    prompt = ("Rewrite the user's question as a short web search query (max 8 words). "
              f"Today is {date.today():%B %d, %Y}. Add the year only if the question is "
              "about recent events. Reply with ONLY the query, nothing else.\n\n"
              f"Question: {question}")
    try:
        lines = chat_once(model, prompt).strip().strip('"').splitlines()
        return lines[0].strip().strip('"') if lines and lines[0].strip() else question
    except Exception:
        return question


# ---------- providers ----------
def _result(title, url, snippet) -> dict:
    return {"title": title or url or "Untitled", "url": url or "", "snippet": snippet or "", "text": ""}


def _search_ddgs(query: str, n: int) -> list[dict]:
    from ddgs import DDGS  # pip install ddgs
    hits = DDGS().text(query, max_results=n)
    return [_result(h.get("title"), h.get("href"), h.get("body")) for h in hits]


def _search_tavily(query: str, n: int) -> list[dict]:
    from tavily import TavilyClient  # pip install tavily-python
    data = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")).search(query, max_results=n)
    return [_result(r.get("title"), r.get("url"), r.get("content")) for r in data.get("results", [])]


PROVIDERS = {"ddgs": _search_ddgs, "tavily": _search_tavily}


def _fetch_page(url: str) -> str:
    """Get the readable text of a page. Returns '' if the site blocks us."""
    try:
        import trafilatura  # pip install trafilatura
        html = trafilatura.fetch_url(url)
        return (trafilatura.extract(html) or "") if html else ""
    except Exception:
        return ""  # we still have the snippet


def search_web(query: str, max_results: int = config.WEB_MAX_RESULTS) -> list[dict]:
    """Return [{title, url, snippet, text}, ...]. Raises on failure (no internet, rate limit...)."""
    if config.WEB_PROVIDER not in PROVIDERS:
        raise ValueError(f"Unknown WEB_PROVIDER '{config.WEB_PROVIDER}'. Use: {', '.join(PROVIDERS)}")
    results = PROVIDERS[config.WEB_PROVIDER](query, max_results)

    n = min(config.WEB_FETCH_PAGES, len(results))
    if n > 0:
        pool = ThreadPoolExecutor(max_workers=n)
        futures = {pool.submit(_fetch_page, r["url"]): r for r in results[:n]}
        try:
            for fut in as_completed(futures, timeout=config.WEB_FETCH_TIMEOUT):
                futures[fut]["text"] = fut.result()[:config.WEB_PAGE_CHARS]
        except FuturesTimeout:
            pass  # slow pages are skipped; their snippets are used instead
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    return results


def build_web_prompt(question: str, results: list[dict]) -> str:
    per_source = config.WEB_MAX_CHARS // max(1, len(results))  # every source gets space
    blocks = []
    for i, r in enumerate(results, 1):
        body = (r["text"] or r["snippet"])[:per_source]
        blocks.append(f"[{i}] {r['title']} ({r['url']})\n{body}")
    context = "\n\n".join(blocks)

    return (f"Today's date is {date.today():%B %d, %Y}.\n"
            "Use the fresh web search results below to answer the question. "
            "Write a clear, complete answer in your own words (a short paragraph or a few "
            "bullet points) with the key facts, and cite them like [1], [2]. "
            "If the results only partly answer the question, share what they say and mention "
            "what is missing. Say you couldn't find it only if the results are completely "
            "unrelated. Do not just list the sources.\n\n"
            f"Web results:\n{context}\n\nQuestion: {question}")


def sources_markdown(results: list[dict]) -> str:
    links = "\n".join(f"{i}. [{r['title']}]({r['url']})" for i, r in enumerate(results, 1))
    return f"\n\n**Sources:**\n{links}"