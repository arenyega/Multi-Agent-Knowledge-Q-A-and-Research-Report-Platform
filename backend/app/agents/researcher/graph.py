"""Research Agent Graph Module

Builds the main StateGraph for the research workflow.
"""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .state import GraphState
from .nodes.task import task_node
from .nodes.planner import planner_node
from .nodes.researcher import researcher_node
from .nodes.publisher import publisher_node

load_dotenv()

# Node names
TASK = "task"
PLANNER = "planner"
RESEARCHER = "researcher"
PUBLISHER = "publisher"


def build_graph():
    """
    Build and compile the research agent graph.
    
    Returns:
        Compiled StateGraph for the research workflow
    """
    # Create the state graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node(TASK, task_node)
    workflow.add_node(PLANNER, planner_node)
    workflow.add_node(RESEARCHER, researcher_node)
    workflow.add_node(PUBLISHER, publisher_node)
    
    # Set entry point
    workflow.set_entry_point(TASK)
    
    # Add edges for linear flow: Task -> Planner -> Researcher -> Publisher
    workflow.add_edge(TASK, PLANNER)
    workflow.add_edge(PLANNER, RESEARCHER)
    workflow.add_edge(RESEARCHER, PUBLISHER)
    workflow.add_edge(PUBLISHER, END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


if __name__ == "__main__":
    # Test the graph compilation
    graph = build_graph()
    print("Research agent graph compiled successfully!")
    
    # Test with a sample query
    test_state = {"question": "What are the latest trends in AI research?", "persona_prompt": "", "search_results": [], "markdown_answer": ""}
    result = graph.invoke(test_state)
    print(f"Test result keys: {result.keys()}") 