from __future__ import annotations

import os

import weaviate
from dotenv import load_dotenv
from openai import OpenAI

from src.retriever import retrieve

load_dotenv()

MODEL = "gpt-4o-mini"
TOP_K = 3

NO_ANSWER = "В базе знаний этого нет."

RAG_SYSTEM = f"""
Ты отвечаешь только по блокам контекста в сообщении пользователя.
Если ответа в контексте нет, напиши ровно: «{NO_ANSWER}»
Не добавляй факты из своих знаний.
Строку «Источники» не пиши.
""".strip()

PLAIN_SYSTEM = (
    "Ответь на вопрос своими знаниями, кратко, на русском. "
    "Источники указывать не нужно."
)


def format_context(hits: list[dict]) -> str:
    if not hits:
        return ""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{index}] source={hit['source_name']} chunk_id={hit['chunk_id']}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(blocks)


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан. Добавьте его в .env")
    return OpenAI(api_key=api_key)


def complete(system: str, user: str) -> str:
    response = _client().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def answer_without_retrieval(query: str) -> str:
    return complete(PLAIN_SYSTEM, query)


def answer_with_retrieval(query: str, context: str) -> str:
    user = f"Вопрос:\n{query}\n\nКонтекст:\n{context}"
    return complete(RAG_SYSTEM, user)


def _drop_source_lines(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not line.strip().lower().startswith("источники:")
    ]
    return "\n".join(lines).strip()


def _with_sources(answer: str, sources: list[str]) -> str:
    body = _drop_source_lines(answer)
    if body.startswith(NO_ANSWER) or not sources:
        return NO_ANSWER if body.startswith(NO_ANSWER) else body
    return f"{body}\n\nИсточники: {', '.join(sources)}"


def answer_question(weaviate_client: weaviate.WeaviateClient, query: str) -> dict:
    hits = retrieve(weaviate_client, query, top_k=TOP_K)
    context = format_context(hits)
    sources = [f"{hit['source_name']}#{hit['chunk_id']}" for hit in hits]
    with_retrieval = _with_sources(answer_with_retrieval(query, context), sources)
    return {
        "query": query,
        "sources": sources,
        "context": context,
        "without_retrieval": answer_without_retrieval(query),
        "with_retrieval": with_retrieval,
    }