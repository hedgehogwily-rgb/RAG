import logging
from pathlib import Path

from src.chunker import CHUNK_SIZE, OVERLAP, chunk_documents
from src.document_loader import load_documents

logger = logging.getLogger(__name__)

PREVIEW_CHUNKS = 3   # сколько чанков показать в примерах


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

    # --- Примеры чанков ---
    logger.info("")
    logger.info("=== Examples (%d chunks) ===", PREVIEW_CHUNKS)
    for chunk in chunks[:PREVIEW_CHUNKS]:
        logger.info(
            "[doc_id=%d | source=%s | chunk_id=%d | len=%d]",
            chunk.document_id,
            chunk.source_name,
            chunk.chunk_id,
            chunk.length,
        )
        logger.info("%s", chunk.text)
        logger.info("-" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    main()
