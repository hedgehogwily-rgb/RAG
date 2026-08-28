import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line.strip())
        if stripped:
            lines.append(stripped)

    return "\n".join(lines).strip()
