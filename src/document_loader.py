from dataclasses import dataclass
from pathlib import Path

from src.text_cleaner import clean_text


@dataclass
class Document:
    filename: str
    path: Path
    content: str

    @property
    def length(self) -> int:
        return len(self.content)


def load_documents(knowledge_base_dir: Path) -> list[Document]:
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {knowledge_base_dir}")

    documents: list[Document] = []
    for file_path in sorted(knowledge_base_dir.glob("*.txt")):
        raw_text = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)
        documents.append(
            Document(
                filename=file_path.name,
                path=file_path,
                content=cleaned_text,
            )
        )

    return documents
