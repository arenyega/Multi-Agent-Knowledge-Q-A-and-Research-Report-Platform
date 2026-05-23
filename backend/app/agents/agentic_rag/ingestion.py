"""Text-based PDF ingestion and retrieval for Agentic RAG.

This version removes the original Cohere multimodal/image embedding dependency.
It extracts text from each PDF page, chunks the text, embeds text, and stores
vectors in Qdrant. If EMBEDDING_MODEL is not configured, it falls back to a
deterministic local hash embedding so the project can still run end-to-end.
"""

import hashlib
import math
import os
import re
import warnings
from typing import Iterable, List

from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "768"))


def pdf_to_text_pages(path: str) -> List[str]:
    """Extract plain text page-by-page from a PDF file."""
    doc = fitz.open(path)
    pages: List[str] = []
    try:
        for page_index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                text = f"[Empty page {page_index + 1}]"
            pages.append(text)
    finally:
        doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split long page text into overlapping chunks."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _tokenize(text: str) -> List[str]:
    """Tokenize mixed English/Chinese text for local fallback embeddings."""
    text = text.lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(cjk_chars[i:i + 2]) for i in range(max(0, len(cjk_chars) - 1))]
    return latin_tokens + cjk_chars + cjk_bigrams


class LocalHashEmbeddings:
    """Deterministic local text embeddings used as a no-API fallback.

    This is not as semantically strong as a real embedding model, but it keeps
    upload/query flows working when only a chat model API key is available.
    """

    def __init__(self, dim: int = DEFAULT_LOCAL_EMBEDDING_DIM):
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_one(query)


class ModelScopeTextEmbeddings:
    """OpenAI-compatible embeddings with local fallback.

    Configure EMBEDDING_MODEL to use a remote ModelScope embedding model.
    Leave EMBEDDING_MODEL empty to use LocalHashEmbeddings.
    """

    def __init__(self):
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "").strip()
        self.local = LocalHashEmbeddings()
        self.remote = None
        if self.embedding_model:
            self.remote = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
            )

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if self.remote:
            try:
                return self.remote.embed_documents(texts)
            except Exception as exc:
                print(f"Remote embedding failed, falling back to local hash embeddings: {exc}")
        return self.local.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        if self.remote:
            try:
                return self.remote.embed_query(query)
            except Exception as exc:
                print(f"Remote query embedding failed, falling back to local hash embeddings: {exc}")
        return self.local.embed_query(query)


def _recreate_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Recreate a Qdrant collection to avoid stale vectors/dimension mismatch."""
    try:
        client.delete_collection(collection_name=collection)
        print(f"Deleted existing collection '{collection}'")
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{collection}' with vector size {vector_size}")


def ingest_pdf(pdf_path: str, collection: str = "pdf_pages", host: str = "localhost", port: int = 6333):
    """Ingest a PDF as text chunks into Qdrant vector store."""
    print(f"Processing PDF as text: {pdf_path}")

    pages = pdf_to_text_pages(pdf_path)
    chunks = []
    for page_index, page_text in enumerate(pages):
        for chunk_index, chunk in enumerate(chunk_text(page_text)):
            chunks.append({
                "text": chunk,
                "page": page_index,
                "chunk": chunk_index,
                "source": pdf_path,
            })

    if not chunks:
        raise ValueError("No text could be extracted from the PDF.")

    print(f"Extracted {len(pages)} pages and {len(chunks)} text chunks")

    embeddings = ModelScopeTextEmbeddings()
    client = QdrantClient(host=host, port=port)

    print("Creating text embeddings...")
    vectors = embeddings.embed_documents([chunk["text"] for chunk in chunks])
    if not vectors or not vectors[0]:
        raise ValueError("Embedding generation returned empty vectors.")
    print(f"Created {len(vectors)} embeddings")

    _recreate_collection(client, collection, len(vectors[0]))

    points = []
    for idx, (vector, chunk) in enumerate(zip(vectors, chunks)):
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={
                "page_content": chunk["text"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "source": chunk["source"],
                "type": "text",
            },
        ))

    print("Inserting points into Qdrant...")
    client.upsert(collection_name=collection, points=points)
    print(f"Successfully inserted {len(points)} text chunks into collection '{collection}'")
    return client, collection


class QdrantRetriever:
    """Simple Qdrant retriever for compatibility with existing code."""

    def __init__(self, collection: str = "pdf_pages", host: str = "localhost", port: int = 6333, k: int = 4):
        self.client = QdrantClient(host=host, port=port)
        self.embeddings = ModelScopeTextEmbeddings()
        self.collection = collection
        self.k = int(os.getenv("RETRIEVER_TOP_K", str(k)))

    def invoke(self, query: str):
        """Search for similar text chunks and return LangChain Documents."""
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=self.k,
        )

        docs = []
        for result in results:
            payload = result.payload or {}
            docs.append(Document(
                page_content=payload.get("page_content", ""),
                metadata={
                    "page": payload.get("page", 0),
                    "chunk": payload.get("chunk", 0),
                    "source": payload.get("source", ""),
                    "type": payload.get("type", "text"),
                    "score": result.score,
                },
            ))
        return docs


def get_retriever(collection: str = "pdf_pages", host: str = "localhost", port: int = 6333):
    """Get retriever from existing Qdrant collection."""
    return QdrantRetriever(collection=collection, host=host, port=port)


retriever = None


if __name__ == "__main__":
    pdf_path = "sample.pdf"
    client, collection = ingest_pdf(pdf_path)
    retriever = get_retriever()
    results = retriever.invoke("What is this document about?")
    for i, doc in enumerate(results):
        print(f"Document {i + 1}: Page {doc.metadata['page'] + 1}, Score: {doc.metadata.get('score', 'N/A')}")
