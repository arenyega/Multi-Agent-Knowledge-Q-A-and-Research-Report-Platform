"""State definition for the text-based Agentic RAG graph."""

from typing import Any, Dict, List, TypedDict


class ImageDocument(TypedDict):
    """Backward-compatible name for a retrieved text document chunk."""
    page_content: str
    page_number: int
    metadata: Dict[str, Any]


class GraphState(TypedDict):
    """Graph state for document QA and optional web search."""
    question: str
    generation: str
    web_search: bool
    documents: List[ImageDocument]
