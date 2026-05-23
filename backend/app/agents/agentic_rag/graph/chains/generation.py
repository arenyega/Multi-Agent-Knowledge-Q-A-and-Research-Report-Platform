"""Text generation chain for Agentic RAG system."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def _get_llm(temperature: float = 0.2):
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )


try:
    llm = _get_llm(temperature=0.2)
except Exception as e:
    print(f"Warning: Could not initialize ModelScope/OpenAI-compatible LLM: {e}")
    llm = None


def _doc_to_text(doc) -> str:
    page_number = doc.get("page_number", 0) if isinstance(doc, dict) else getattr(doc, "page_number", 0)
    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
    content = doc.get("page_content", "") if isinstance(doc, dict) else getattr(doc, "page_content", "")
    source = metadata.get("source", "document") if isinstance(metadata, dict) else "document"
    chunk = metadata.get("chunk", 0) if isinstance(metadata, dict) else 0
    if page_number == -1 or metadata.get("source") == "web_search":
        label = "Web search result"
    else:
        label = f"Document page {page_number + 1}, chunk {chunk + 1}"
    return f"[{label} | source: {source}]\n{content}"


def generation_chain(inputs):
    """Generate answer using retrieved text chunks and optional Tavily web text."""
    if not llm:
        return "Error: LLM not initialized. Please check OPENAI_API_KEY, OPENAI_BASE_URL, and LLM_MODEL."

    question = inputs["question"]
    documents = inputs.get("context", [])
    context = "\n\n---\n\n".join(_doc_to_text(doc) for doc in documents if doc)

    if not context.strip():
        context = "No document context was retrieved."

    prompt = f"""你是一个严谨的文档问答助手。请基于给定的文档片段和可能的联网搜索结果回答用户问题。

用户问题：
{question}

可用上下文：
{context}

回答要求：
1. 优先依据文档内容回答；如果使用了联网搜索结果，请明确说明。
2. 如果上下文没有相关信息，请直接说“文档中没有找到相关信息”。
3. 不要编造上下文中不存在的事实。
4. 使用用户提问的语言回答，结构清晰、简洁准确。

答案："""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
