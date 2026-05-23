"""Research Agent package for generating comprehensive research reports."""

from .graph import build_graph as build_research_graph
from .state import GraphState

__all__ = ["build_research_graph", "GraphState"] 