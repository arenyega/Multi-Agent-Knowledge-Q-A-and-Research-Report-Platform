"""Web search node for Agentic RAG system."""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from ..state import GraphState, ImageDocument

load_dotenv()


def _extract_tavily_contents(raw_results) -> List[str]:
    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", raw_results.get("answer", []))
    if isinstance(raw_results, str):
        return [raw_results]
    contents = []
    for item in raw_results or []:
        if isinstance(item, dict):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "") or item.get("snippet", "")
            contents.append(f"{title}\n{url}\n{content}".strip())
        elif isinstance(item, str):
            contents.append(item)
    return [c for c in contents if c]


def web_search(state: GraphState) -> Dict[str, Any]:
    """Perform web search and append text result to documents."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", []) or []

    if not os.getenv("TAVILY_API_KEY"):
        print("TAVILY_API_KEY is not configured; skipping web search.")
        return {"documents": documents, "question": question, "web_search": False}

    try:
        web_search_tool = TavilySearch(max_results=3)
        raw = web_search_tool.invoke({"query": question})
        contents = _extract_tavily_contents(raw)
        joined = "\n\n".join(contents)
    except Exception as exc:
        print(f"Tavily search failed: {exc}")
        joined = f"Web search failed: {exc}"

    web_doc = ImageDocument(
        page_content=joined,
        page_number=-1,
        metadata={"source": "web_search", "type": "text"},
    )
    documents.append(web_doc)
    return {"documents": documents, "question": question, "web_search": True}
