"""Grade documents node for text-based Agentic RAG system."""

from typing import Any, Dict
from ..chains.retrieval_grader import retrieval_grader
from ..state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Filter retrieved text chunks by relevance."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs = []
    web_search = False
    for doc in documents:
        page = doc.get("page_number", 0)
        try:
            score = retrieval_grader({"question": question, "document": doc})
            grade = score.binary_score
            if grade.lower() == "yes":
                print(f"---GRADE: DOCUMENT PAGE {page + 1} RELEVANT---")
                filtered_docs.append(doc)
            else:
                print(f"---GRADE: DOCUMENT PAGE {page + 1} NOT RELEVANT---")
                web_search = True
        except Exception as e:
            print(f"---ERROR GRADING DOCUMENT PAGE {page + 1}: {e}---")
            filtered_docs.append(doc)

    if not filtered_docs and documents:
        filtered_docs = documents[:2]
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
