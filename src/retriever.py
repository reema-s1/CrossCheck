"""Top-k cosine-similarity retrieval over the FAISS index."""
import json

import faiss

from src.indexing import INDEX_DIR, get_embedder


class Retriever:
    def __init__(self, k: int = 3):
        self.k = k
        self.index = faiss.read_index(str(INDEX_DIR / "kb.faiss"))
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text())

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        qv = get_embedder().encode([question], normalize_embeddings=True)
        scores, ids = self.index.search(qv, k or self.k)
        return [{**self.chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How much does a phone screen repair cost?"
    for hit in Retriever().retrieve(q):
        print(f"{hit['score']:.3f}  {hit['id']}  {hit['text'][:90]}...")
