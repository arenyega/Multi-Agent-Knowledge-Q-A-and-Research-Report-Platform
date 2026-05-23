"""Researcher Node Module

Generates search queries using a ModelScope/OpenAI-compatible LLM and executes web searches using Tavily.
"""

import ast
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

from ..state import GraphState
from ..prompts import get_search_queries_prompt

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
)


def _extract_results(raw_results) -> List[str]:
    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", raw_results.get("answer", []))
    if isinstance(raw_results, str):
        return [raw_results]
    contents: List[str] = []
    for result in raw_results or []:
        if isinstance(result, dict):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "") or result.get("snippet", "")
            item = f"{title}\n{url}\n{content}".strip()
            if item:
                contents.append(item)
        elif isinstance(result, str):
            contents.append(result)
    return contents


def _parse_queries(queries_response: str, question: str) -> List[str]:
    try:
        search_queries = ast.literal_eval(queries_response)
        if isinstance(search_queries, list) and search_queries:
            return [str(q).strip() for q in search_queries if str(q).strip()][:4]
    except Exception:
        pass

    lines = queries_response.strip().split("\n")
    search_queries = []
    for line in lines:
        line = line.strip().lstrip("-*").strip()
        line = line.lstrip("1234567890.、) ").strip('"\'')
        if line:
            search_queries.append(line)

    if len(search_queries) < 4:
        search_queries.extend([
            f"{question} latest developments",
            f"{question} current status",
            f"{question} analysis",
            f"{question} expert opinion",
        ])
    return search_queries[:4]


def researcher_node(state: GraphState) -> Dict[str, Any]:
    """Generate search queries and execute them with Tavily."""
    print("---RESEARCHER NODE---")
    question = state["question"]

    prompt_template = ChatPromptTemplate.from_template(get_search_queries_prompt(question))
    chain = prompt_template | llm | StrOutputParser()
    queries_response = chain.invoke({"query": question})
    print(f"LLM response for queries: {queries_response[:200]}...")

    search_queries = _parse_queries(queries_response, question)
    print(f"Generated {len(search_queries)} search queries")

    all_results: List[str] = []
    if not os.getenv("TAVILY_API_KEY"):
        all_results.append("TAVILY_API_KEY is not configured; web search was skipped.")
    else:
        tavily_search = TavilySearch(max_results=2)
        for i, query in enumerate(search_queries):
            try:
                print(f"Searching: {query}")
                raw_results = tavily_search.invoke({"query": query})
                extracted = _extract_results(raw_results)
                all_results.extend(extracted)
            except Exception as e:
                print(f"Search error for query {i}: {e}")
                all_results.append(f"Search failed for: {query}. Error: {e}")

    print(f"Collected {len(all_results)} search results")
    return {
        "question": question,
        "persona_prompt": state.get("persona_prompt", ""),
        "search_results": all_results,
        "markdown_answer": state.get("markdown_answer", ""),
    }
