"""Task Node Module

Handles the initial task input and query processing.
"""

from typing import Dict, Any
from ..state import GraphState


def task_node(state: GraphState) -> Dict[str, Any]:
    """
    Initial task processing node that takes user query as input.
    
    Args:
        state: Current graph state containing question
        
    Returns:
        Updated state with question maintained
    """
    print("---TASK NODE---")
    print(f"🔥 TASK DEBUG: Initial state keys: {list(state.keys())}")
    print(f"🔥 TASK DEBUG: Initial state content: {state}")
    
    question = state["question"]
    print(f"Processing query: {question}")
    
    # Return complete state to ensure proper merging
    result = {
        "question": question,
        "persona_prompt": state.get("persona_prompt", ""),
        "search_results": state.get("search_results", []),
        "markdown_answer": state.get("markdown_answer", "")
    }
    print(f"🔥 TASK DEBUG: Complete state being returned: {list(result.keys())}")
    return result 