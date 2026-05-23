"""Router chain for text-based Agentic RAG system."""

import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Choose websearch for current/general web information, otherwise vectorstore.",
    )


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


class SimpleQuestionRouter:
    def invoke(self, inputs):
        force_vectorstore = os.getenv("FORCE_VECTORSTORE", "false").lower() in {"1", "true", "yes"}
        if force_vectorstore or not os.getenv("TAVILY_API_KEY"):
            return RouteQuery(datasource="vectorstore")
        if not llm:
            return RouteQuery(datasource="vectorstore")

        question = inputs.get("question", "")
        prompt = f"""Route this question to one datasource.

Use vectorstore for questions about an uploaded PDF/document, document content, resumes, reports, contracts, local files, or follow-up questions about the uploaded file.
Use websearch for current events, latest information, general web knowledge, prices, laws, or anything not likely inside the uploaded document.

Question: {question}

Reply with exactly one word: vectorstore or websearch."""
        try:
            response = llm.invoke([HumanMessage(content=prompt)]).content.lower().strip()
            if "web" in response:
                return RouteQuery(datasource="websearch")
            return RouteQuery(datasource="vectorstore")
        except Exception as exc:
            print(f"Question routing failed: {exc}")
            return RouteQuery(datasource="vectorstore")


question_router = SimpleQuestionRouter()
