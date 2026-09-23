"""Chunk the knowledge base, embed chunks, and build a FAISS index."""
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "docs" / "knowledge_base"
INDEX_DIR = ROOT / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 200    # ~260 tokens
OVERLAP_WORDS = 40

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _model


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")) + sorted(KB_DIR.glob("*.txt")):
        for i, text in enumerate(chunk_text(path.read_text())):
            chunks.append({"id": f"{path.stem}#{i}", "source": path.name, "text": text})
    return chunks


def build_index() -> None:
    chunks = load_chunks()
    vecs = get_embedder().encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on unit vectors = cosine
    index.add(vecs)
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "kb.faiss"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=1))
    print(f"Indexed {len(chunks)} chunks from {KB_DIR}")


if __name__ == "__main__":
    build_index()
