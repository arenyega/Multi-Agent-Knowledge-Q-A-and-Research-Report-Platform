"""Research Agent State Module

Defines the state structure for the research agent graph.
"""

from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the research graph.

    Attributes:
        question: The original user question (changed from user_query to match Agentic RAG)
        persona_prompt: Generated researcher persona/instructions
        search_results: List of search result strings
        markdown_answer: Final markdown report
    """

    question: str
    persona_prompt: str
    search_results: List[str]
    markdown_answer: str 