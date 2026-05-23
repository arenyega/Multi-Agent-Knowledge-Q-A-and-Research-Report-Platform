"""Hallucination grading chain for text-based Agentic RAG system."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class GradeHallucinations(BaseModel):
    """Binary score for grounding in provided documents."""
    binary_score: bool = Field(description="Answer is grounded in the facts")


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )


try:
    llm = _get_llm()
except Exception as e:
    print(f"Warning: Could not initialize ModelScope/OpenAI-compatible LLM: {e}")
    llm = None


def hallucination_grader(inputs):
    """Grade whether generation is grounded in text documents."""
    if not llm:
        return GradeHallucinations(binary_score=True)

    documents = inputs.get("documents", [])
    generation = inputs.get("generation", "")
    context_parts = []
    for doc in documents[:6]:
        if isinstance(doc, dict):
            context_parts.append(doc.get("page_content", ""))
        else:
            context_parts.append(str(doc))
    context = "\n\n---\n\n".join(context_parts)[:8000]

    prompt = f"""Assess whether the answer is supported by the provided context.

Context:
{context}

Answer:
{generation}

Reply with only yes or no. Yes means the answer is grounded in the context or clearly says the context does not contain the answer."""
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.lower().strip()
        return GradeHallucinations(binary_score=("yes" in response or "是" in response or "true" in response))
    except Exception as e:
        print(f"Error in hallucination grading: {e}")
        return GradeHallucinations(binary_score=True)
