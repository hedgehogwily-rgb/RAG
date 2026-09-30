from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import weaviate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chunker import chunk_documents
from src.document_loader import load_documents
from src.generator import answer_question
from src.weaviate_store import connect, count_objects, upsert_chunks

KNOWLEDGE_BASE = Path(__file__).parent / "knowledge_base"

DEMO_QUESTIONS = [
    "Что такое RAG и зачем нужна внешняя база знаний?",
    "Зачем нужен overlap при разбиении текста на чанки?",
    "Чем embeddings отличаются от обычного ключевого поиска?",
    "Для чего используют векторные базы данных вроде Weaviate?",
    "Как оценивать качество retrieval в RAG-системе?",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = connect()
    app.state.weaviate = client
    try:
        yield
    finally:
        client.close()


app = FastAPI(title="RAG mini-product", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    context: str
    weak_retrieval: bool
    best_score: float | None


class IndexResponse(BaseModel):
    documents: int
    chunks: int
    objects_in_weaviate: int


class HealthResponse(BaseModel):
    weaviate: str
    objects: int


def _client() -> weaviate.WeaviateClient:
    client = app.state.weaviate
    if not client.is_ready():
        raise HTTPException(status_code=503, detail="Weaviate is not ready")
    return client


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    client = _client()
    return HealthResponse(weaviate="ready", objects=count_objects(client))


@app.get("/demo-questions", response_model=list[str])
def demo_questions() -> list[str]:
    return DEMO_QUESTIONS


@app.post("/index", response_model=IndexResponse)
def index_knowledge_base() -> IndexResponse:
    client = _client()
    documents = load_documents(KNOWLEDGE_BASE)
    chunks = chunk_documents(documents)
    uploaded = upsert_chunks(client, chunks)
    return IndexResponse(
        documents=len(documents),
        chunks=uploaded,
        objects_in_weaviate=count_objects(client),
    )


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    client = _client()
    if count_objects(client) == 0:
        raise HTTPException(
            status_code=409,
            detail="Индекс пуст. Сначала вызовите POST /index.",
        )
    result = answer_question(client, body.question.strip(), include_baseline=False)
    return AskResponse(
        question=result["query"],
        answer=result["with_retrieval"],
        sources=result["sources"],
        context=result["context"],
        weak_retrieval=result["weak_retrieval"],
        best_score=result["best_score"],
    )
