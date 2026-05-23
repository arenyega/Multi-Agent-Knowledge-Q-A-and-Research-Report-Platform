"""Generate node for text-based Agentic RAG system."""

from typing import Any, Dict
from ..chains.generation import generation_chain
from ..state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer using retrieved text chunks."""
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])

    try:
        generation = generation_chain({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    except Exception as e:
        print(f"---ERROR IN GENERATION: {e}---")
        fallback_generation = f"抱歉，处理文档并回答问题时出现错误：{e}"
        return {"documents": documents, "question": question, "generation": fallback_generation}
