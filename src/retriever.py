from __future__ import annotations

import weaviate

from src.embeddings import embed_texts
from src.weaviate_store import hybrid_search, search_by_vector

DEFAULT_TOP_K = 5


def retrieve(
    client: weaviate.WeaviateClient,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    vector = embed_texts([query])[0]
    return search_by_vector(client, vector, limit=top_k)


def retrieve_hybrid(
    client: weaviate.WeaviateClient,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = 0.5,
) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    vector = embed_texts([query])[0]
    return hybrid_search(client, query, vector, limit=top_k, alpha=alpha)
