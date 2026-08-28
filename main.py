import logging
from pathlib import Path

from src.document_loader import load_documents

logger = logging.getLogger(__name__)


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
    logger.info("Ready for the next step: chunking.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    main()
