import logging
from pathlib import Path

from src.chunker import CHUNK_SIZE, OVERLAP, chunk_documents
from src.document_loader import load_documents
from src.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL
from src.weaviate_store import (
    connect,
    count_objects,
    fetch_sample,
    upsert_chunks,
)

logger = logging.getLogger(__name__)

LONG_DOC_NAME = "rag_pipeline_long.txt"
OVERLAP_PREVIEW = 40
SAMPLE_LIMIT = 3
TEXT_PREVIEW = 120


def main() -> None:
    knowledge_base_dir = Path(__file__).parent / "knowledge_base"
    documents = load_documents(knowledge_base_dir)

    logger.info("Loaded %d documents from %s", len(documents), knowledge_base_dir)
    logger.info("%-30s %15s", "Filename", "Length (chars)")
    logger.info("-" * 47)
    for doc in documents:
        logger.info("%-30s %15d", doc.filename, doc.length)
    total_length = sum(doc.length for doc in documents)
    logger.info("-" * 47)
    logger.info("%-30s %15d", "Total", total_length)

    # --- Chunking ---
    logger.info("")
    logger.info("Chunking: chunk_size=%d, overlap=%d", CHUNK_SIZE, OVERLAP)
    chunks = chunk_documents(documents)
    logger.info("Total chunks: %d", len(chunks))

    # --- Длинный документ: несколько чанков + overlap ---
    long_chunks = [c for c in chunks if c.source_name == LONG_DOC_NAME]
    if len(long_chunks) < 2:
        raise RuntimeError(
            f"{LONG_DOC_NAME} must produce >= 2 chunks "
            f"(got {len(long_chunks)}; check length vs CHUNK_SIZE={CHUNK_SIZE})"
        )

    logger.info("")
    logger.info(
        "=== Long document demo: %s → %d chunks ===",
        LONG_DOC_NAME,
        len(long_chunks),
    )
    for chunk in long_chunks:
        logger.info(
            "[doc_id=%d | source=%s | chunk_id=%d | len=%d]",
            chunk.document_id,
            chunk.source_name,
            chunk.chunk_id,
            chunk.length,
        )
        logger.info("%s", chunk.text)
        logger.info("-" * 60)

    first, second = long_chunks[0], long_chunks[1]
    from_prev = first.text[-OVERLAP:]
    from_next = second.text[:OVERLAP]
    logger.info(
        "Overlap check (last %d of chunk_id=0 vs first %d of chunk_id=1):",
        OVERLAP,
        OVERLAP,
    )
    logger.info("  chunk_id=0: %s...", from_prev[:OVERLAP_PREVIEW])
    logger.info("  chunk_id=1: %s...", from_next[:OVERLAP_PREVIEW])
    logger.info("  overlap text matches: %s", from_prev == from_next)

    # --- Weaviate: embeddings + upsert ---
    logger.info("")
    logger.info(
        "=== Weaviate upload (model=%s, dim=%d) ===",
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
    )

    client = connect()
    try:
        uploaded = upsert_chunks(client, chunks)
        total_in_db = count_objects(client)
        logger.info("Upserted chunks: %d", uploaded)
        logger.info("Objects in Weaviate: %d", total_in_db)

        if total_in_db != len(chunks):
            raise RuntimeError(
                f"Expected {len(chunks)} objects in Weaviate, found {total_in_db}"
            )

        samples = fetch_sample(client, limit=SAMPLE_LIMIT)
        logger.info("")
        logger.info("=== Sample objects from Weaviate ===")
        for sample in samples:
            text = sample["text"] or ""
            preview = text if len(text) <= TEXT_PREVIEW else text[:TEXT_PREVIEW] + "..."
            logger.info(
                "[uuid=%s | doc_id=%s | source=%s | chunk_id=%s | vector=%s dim=%s]",
                sample["uuid"],
                sample["document_id"],
                sample["source_name"],
                sample["chunk_id"],
                sample["has_vector"],
                sample["vector_dim"],
            )
            logger.info("%s", preview)
            logger.info("-" * 60)

        logger.info(
            "Re-run main.py to verify idempotent reload (object count must stay %d).",
            total_in_db,
        )
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
