"""Retrieve node for text-based Agentic RAG system."""

import os
from typing import Any, Dict
from ..state import GraphState, ImageDocument
from ...ingestion import get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant text chunks from the vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    retriever = get_retriever(collection="pdf_pages", host=qdrant_host, port=qdrant_port)
    documents = retriever.invoke(question)

    text_docs = []
    for doc in documents:
        text_docs.append(ImageDocument(
            page_content=doc.page_content,
            page_number=doc.metadata.get("page", 0),
            metadata=doc.metadata,
        ))
    return {"documents": text_docs, "question": question}
