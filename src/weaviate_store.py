from __future__ import annotations

import logging
from uuid import NAMESPACE_URL, uuid5

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery
from weaviate.collections import Collection

from src.chunker import Chunk
from src.embeddings import embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "KnowledgeChunk"
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8081
WEAVIATE_GRPC_PORT = 50052


def connect() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )


def ensure_collection(client: weaviate.WeaviateClient) -> Collection:
    if client.collections.exists(COLLECTION_NAME):
        return client.collections.get(COLLECTION_NAME)

    return client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="document_id", data_type=DataType.INT),
            Property(name="source_name", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.INT),
            Property(name="text", data_type=DataType.TEXT),
        ],
        vector_config=Configure.Vectors.self_provided(),
    )


def chunk_uuid(chunk: Chunk) -> str:
    return str(uuid5(NAMESPACE_URL, f"{chunk.source_name}::{chunk.chunk_id}"))


def upsert_chunks(client: weaviate.WeaviateClient, chunks: list[Chunk]) -> int:
    collection = ensure_collection(client)
    desired_ids = {chunk_uuid(chunk) for chunk in chunks}

    if chunks:
        vectors = embed_texts([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            properties = {
                "document_id": chunk.document_id,
                "source_name": chunk.source_name,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
            object_id = chunk_uuid(chunk)
            if collection.data.exists(object_id):
                collection.data.replace(
                    uuid=object_id,
                    properties=properties,
                    vector=vector,
                )
            else:
                collection.data.insert(
                    properties=properties,
                    uuid=object_id,
                    vector=vector,
                )

    stale_ids = [
        str(obj.uuid)
        for obj in collection.iterator(return_properties=[])
        if str(obj.uuid) not in desired_ids
    ]
    for object_id in stale_ids:
        collection.data.delete_by_id(object_id)
    if stale_ids:
        logger.info("Removed %d stale chunk(s) from Weaviate", len(stale_ids))

    return len(chunks)


def count_objects(client: weaviate.WeaviateClient) -> int:
    collection = ensure_collection(client)
    response = collection.aggregate.over_all(total_count=True)
    return int(response.total_count or 0)


def fetch_sample(
    client: weaviate.WeaviateClient,
    limit: int = 3,
) -> list[dict]:
    collection = ensure_collection(client)
    response = collection.query.fetch_objects(limit=limit, include_vector=True)

    samples: list[dict] = []
    for obj in response.objects:
        vector = obj.vector
        has_vector = False
        vector_dim = 0
        if isinstance(vector, dict):
            default = vector.get("default") or next(iter(vector.values()), None)
            if default is not None:
                has_vector = True
                vector_dim = len(default)
        elif vector is not None:
            has_vector = True
            vector_dim = len(vector)

        samples.append(
            {
                "uuid": str(obj.uuid),
                "document_id": obj.properties.get("document_id"),
                "source_name": obj.properties.get("source_name"),
                "chunk_id": obj.properties.get("chunk_id"),
                "text": obj.properties.get("text", ""),
                "has_vector": has_vector,
                "vector_dim": vector_dim,
            }
        )
    return samples


def search_by_vector(
    client: weaviate.WeaviateClient,
    vector: list[float],
    limit: int = 5,
) -> list[dict]:
    collection = ensure_collection(client)
    response = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
    )
    results: list[dict] = []
    for obj in response.objects:
        distance = obj.metadata.distance
        results.append(
            {
                "text": obj.properties.get("text", ""),
                "source_name": obj.properties.get("source_name"),
                "chunk_id": obj.properties.get("chunk_id"),
                "document_id": obj.properties.get("document_id"),
                "distance": distance,
                "score": None if distance is None else 1.0 - float(distance),
            }
        )
    return results


def hybrid_search(
    client: weaviate.WeaviateClient,
    query: str,
    vector: list[float],
    limit: int = 5,
    alpha: float = 0.5,
) -> list[dict]:
    collection = ensure_collection(client)
    response = collection.query.hybrid(
        query=query,
        vector=vector,
        alpha=alpha,
        limit=limit,
        return_metadata=MetadataQuery(score=True, distance=True),
    )
    results: list[dict] = []
    for obj in response.objects:
        distance = obj.metadata.distance
        score = obj.metadata.score
        results.append(
            {
                "text": obj.properties.get("text", ""),
                "source_name": obj.properties.get("source_name"),
                "chunk_id": obj.properties.get("chunk_id"),
                "document_id": obj.properties.get("document_id"),
                "distance": distance,
                "score": score,
            }
        )
    return results
