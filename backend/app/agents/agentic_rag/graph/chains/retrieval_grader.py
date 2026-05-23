"""Retrieval grading chain for text-based Agentic RAG system."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


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


def retrieval_grader(inputs):
    """Grade text document relevance."""
    if not llm:
        return GradeDocuments(binary_score="yes")

    document = inputs.get("document", {})
    question = inputs.get("question", "")
    content = document.get("page_content", "") if isinstance(document, dict) else str(document)
    content = content[:4000]

    prompt = f"""You are grading whether a retrieved text chunk is relevant to a user question.

Document content:
{content}

User question:
{question}

If the document contains keywords or semantic meaning related to the question, answer yes. Otherwise answer no.
Reply with only yes or no."""
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.lower().strip()
        return GradeDocuments(binary_score="yes" if ("yes" in response or "是" in response or "true" in response) else "no")
    except Exception as e:
        print(f"Error in document grading: {e}")
        return GradeDocuments(binary_score="yes")
