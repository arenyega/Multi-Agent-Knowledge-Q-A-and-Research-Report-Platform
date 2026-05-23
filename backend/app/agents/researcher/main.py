"""Research Agent Main Module

Demonstrates usage of the researcher agent.
"""

from graph import build_graph


def main():
    """Run a sample research query through the agent."""
    # Build the research graph
    research_agent = build_graph()
    
    # Sample query
    query = "how to ruin an neahgbour economy"
    
    print(f"Running research query: {query}")
    print("=" * 60)
    
    # Execute the research workflow
    result = research_agent.invoke({
        "question": query,
        "persona_prompt": "",
        "search_results": [],
        "markdown_answer": ""
    })
    
    print("\n" + "=" * 60)
    print("FINAL MARKDOWN REPORT:")
    print("=" * 60)
    print(result["markdown_answer"])


if __name__ == "__main__":
    main() 