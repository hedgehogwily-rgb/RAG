from dataclasses import dataclass, field

from src.document_loader import Document

CHUNK_SIZE = 500
OVERLAP = 100


@dataclass
class Chunk:
    document_id: int
    source_name: str
    chunk_id: int
    text: str

    @property
    def length(self) -> int:
        return len(self.text)


def split_into_chunks(
    document: Document,
    document_id: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    text = document.content
    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    document_id=document_id,
                    source_name=document.filename,
                    chunk_id=chunk_id,
                    text=chunk_text,
                )
            )
            chunk_id += 1

        if end == len(text):
            break
        start = end - overlap

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc_id, document in enumerate(documents):
        all_chunks.extend(
            split_into_chunks(document, doc_id, chunk_size, overlap)
        )
    return all_chunks
