"""Answer grading chain for text-based Agentic RAG system."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class GradeAnswer(BaseModel):
    """Binary score for whether an answer addresses the question."""
    binary_score: bool = Field(description="Answer addresses the question")


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


class SimpleAnswerGrader:
    def invoke(self, inputs):
        if not llm:
            return GradeAnswer(binary_score=True)
        question = inputs.get("question", "")
        generation = inputs.get("generation", "")
        prompt = f"""Assess whether the answer addresses the user question.

User question: {question}
Answer: {generation}

Reply with only yes or no."""
        try:
            response = llm.invoke([HumanMessage(content=prompt)]).content.lower().strip()
            return GradeAnswer(binary_score=("yes" in response or "是" in response or "true" in response))
        except Exception as exc:
            print(f"Answer grading failed: {exc}")
            return GradeAnswer(binary_score=True)


answer_grader = SimpleAnswerGrader()
