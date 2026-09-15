import logging
from pathlib import Path

from src.chunker import CHUNK_SIZE, OVERLAP, chunk_documents
from src.document_loader import load_documents

logger = logging.getLogger(__name__)

LONG_DOC_NAME = "rag_pipeline_long.txt"
OVERLAP_PREVIEW = 40  # сколько символов overlap показать на стыке чанков


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
    # Overlap — это конец chunk_id=0 ≈ начало chunk_id=1 (до ~OVERLAP символов)
    from_prev = first.text[-OVERLAP:]
    from_next = second.text[:OVERLAP]
    logger.info("Overlap check (last %d of chunk_id=0 vs first %d of chunk_id=1):", OVERLAP, OVERLAP)
    logger.info("  chunk_id=0: %s...", from_prev[:OVERLAP_PREVIEW])
    logger.info("  chunk_id=1: %s...", from_next[:OVERLAP_PREVIEW])
    logger.info("  overlap text matches: %s", from_prev == from_next)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    main()
