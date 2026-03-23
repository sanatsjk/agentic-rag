import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[dict]):
        """Build FAISS index from chunks."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        self.index.add(embeddings)

        # Save index and chunks
        DATA_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(DATA_DIR / "faiss.index"))
        with open(DATA_DIR / "chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2)

    def load_index(self):
        """Load existing FAISS index and chunks from disk."""
        index_path = DATA_DIR / "faiss.index"
        chunks_path = DATA_DIR / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("Index not found. Run build_index first.")

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path) as f:
            self.chunks = json.load(f)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index and return top-k matching chunks with scores."""
        if self.index is None:
            self.load_index()

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def search_by_paper(self, paper_id: str, query: str, top_k: int = 3) -> list[dict]:
        """Search within a specific paper's chunks."""
        if self.index is None:
            self.load_index()

        # Filter chunks for this paper
        paper_chunks = [
            (i, c) for i, c in enumerate(self.chunks) if c["paper_id"] == paper_id
        ]

        if not paper_chunks:
            return []

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)

        # Get embeddings for paper chunks and search
        indices_list = [i for i, _ in paper_chunks]
        texts = [c["text"] for _, c in paper_chunks]
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        scores = np.dot(embeddings, query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = paper_chunks[idx][1].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results
