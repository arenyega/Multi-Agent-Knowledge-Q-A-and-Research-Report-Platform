"""Research Agent Prompts Module

Contains reusable prompt templates for the research workflow.
"""

from datetime import datetime, timezone


def get_planner_prompt() -> str:
    """Generate a persona prompt template for the researcher based on the query."""
    return """Based on the user query: "{query}"

Determine the most appropriate researcher persona and create a concise instruction prompt.
Consider the field/domain of the question and craft a "You are a..." prompt that would help 
a researcher provide the most relevant and expert perspective.

Examples:
- For finance questions: "You are a seasoned finance analyst..."  
- For technology questions: "You are an experienced tech researcher..."
- For travel questions: "You are a knowledgeable travel expert..."

Return ONLY the persona prompt starting with "You are..." and ending with relevant expertise."""


def get_search_queries_prompt(question: str) -> str:
    """Generate prompt for creating 4 distinct search queries."""
    current_date = datetime.now(timezone.utc).strftime('%B %d, %Y')
    
    return f"""Generate exactly 4 distinct Google search queries to research the following question: "{question}"

Current date: {current_date}

The queries should:
1. Cover different angles or aspects of the question
2. Be specific and focused 
3. Help form an objective, comprehensive understanding
4. Include relevant keywords and context

IMPORTANT: Each query must be different and explore different aspects of the topic.

Examples for "climate change effects":
- "climate change effects on agriculture 2024"
- "global warming sea level rise impacts"
- "climate change economic consequences developing countries"
- "renewable energy solutions climate crisis"

Return your response as exactly 4 lines, one query per line, without quotes or brackets:
query 1
query 2  
query 3
query 4"""


def get_publisher_prompt(question: str, search_results: list[str]) -> str:
    """Generate prompt for synthesizing search results into a markdown report."""
    results_text = "\n\n---\n\n".join(search_results)
    
    return f"""Based on the following search results, create a comprehensive markdown report answering: "{question}"

Search Results:
{results_text}

Requirements:
- Use proper markdown formatting with H2 (##) headers for main sections
- Include bullet points for key findings
- Cite sources where relevant using [Source](url) format when URLs are available
- Structure the report logically with clear sections
- Provide a balanced, objective analysis
- Include specific facts, data, and examples from the search results
- Aim for 500-800 words

Generate ONLY the markdown content, no additional commentary.""" 