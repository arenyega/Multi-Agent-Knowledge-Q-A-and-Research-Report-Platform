"""Planner Node Module

Generates researcher persona and instructions based on the user query.
"""

import os
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..state import GraphState
from ..prompts import get_planner_prompt

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
)

def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate researcher persona and instructions.
    
    Args:
        state: Current graph state with question
        
    Returns:
        Updated state with persona_prompt
    """
    print("---PLANNER NODE---")
    print(f"🔥 PLANNER DEBUG: State keys: {list(state.keys())}")
    print(f"🔥 PLANNER DEBUG: State content: {state}")
    
    question = state["question"]
    
    # Generate researcher persona
    prompt_template = ChatPromptTemplate.from_template(get_planner_prompt())
    chain = prompt_template | llm | StrOutputParser()
    
    persona_prompt = chain.invoke({"query": question})
    print(f"Generated persona: {persona_prompt[:100]}...")
    
    # Return complete state to ensure proper merging
    result = {
        "question": question,
        "persona_prompt": persona_prompt,
        "search_results": state.get("search_results", []),
        "markdown_answer": state.get("markdown_answer", "")
    }
    print(f"🔥 PLANNER DEBUG: Complete state being returned: {list(result.keys())}")
    return result 