"""Semantic note search with FAISS + Sentence Transformers and a TF-IDF fallback.

The FAISS index is local and persisted under vector_store/. If optional AI packages
or the embedding model are unavailable, retrieval falls back to deterministic TF-IDF.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _note_documents(notes: pd.DataFrame) -> list[str]:
    return [
        f"Note ID: {row.note_id}. Date: {row.date}. Applies to: {row.applies_to}. Event: {row.note}"
        for row in notes.itertuples(index=False)
    ]


class NoteVectorStore:
    """Index contextual notes and return a score for every note in the index."""

    def __init__(self, notes: pd.DataFrame, index_dir: str | Path | None = None,
                 model_name: str = DEFAULT_MODEL):
        self.notes = notes.reset_index(drop=True).copy()
        self.documents = _note_documents(self.notes)
        self.index_dir = Path(index_dir or ROOT / "vector_store")
        self.model_name = model_name
        self.backend = "tfidf"
        self.warning: str | None = None
        self._faiss = None
        self._model = None
        self._index = None
        self._tfidf = None
        self._matrix = None
        self._initialise()

    def _fingerprint(self) -> str:
        payload = self.model_name + "\n" + "\n".join(self.documents)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _initialise(self) -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            from sklearn.preprocessing import normalize

            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._faiss = faiss
            self._model = SentenceTransformer(self.model_name)
            index_path = self.index_dir / "context_notes.faiss"
            metadata_path = self.index_dir / "index_metadata.json"
            fingerprint = self._fingerprint()
            cached = False
            if index_path.exists() and metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("fingerprint") == fingerprint and metadata.get("note_count") == len(self.documents):
                        self._index = faiss.read_index(str(index_path))
                        cached = self._index.ntotal == len(self.documents)
                except (OSError, ValueError, RuntimeError):
                    cached = False
            if not cached:
                vectors = self._model.encode(self.documents, convert_to_numpy=True,
                                             normalize_embeddings=True, show_progress_bar=False)
                vectors = vectors.astype("float32")
                self._index = faiss.IndexFlatIP(vectors.shape[1])
                self._index.add(vectors)
                faiss.write_index(self._index, str(index_path))
                metadata_path.write_text(json.dumps({
                    "fingerprint": fingerprint,
                    "model_name": self.model_name,
                    "note_count": len(self.documents),
                    "note_ids": self.notes["note_id"].astype(str).tolist(),
                }, indent=2) + "\n", encoding="utf-8")
            self.backend = "faiss+sentence-transformers"
        except Exception as exc:  # Optional dependency/model download failure must not block core pipeline.
            self.warning = f"FAISS/embedding backend unavailable; using TF-IDF fallback ({type(exc).__name__}: {exc})"
            self._faiss = None
            self._model = None
            self._index = None
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self._matrix = self._tfidf.fit_transform(self.documents) if self.documents else None
            self.backend = "tfidf-fallback"

    def scores(self, query: str) -> dict[int, float]:
        """Return similarity scores keyed by original row index."""
        if not self.documents:
            return {}
        if self.backend == "faiss+sentence-transformers":
            vector = self._model.encode([query], convert_to_numpy=True,
                                        normalize_embeddings=True, show_progress_bar=False).astype("float32")
            scores, indices = self._index.search(vector, len(self.documents))
            return {int(idx): float(score) for idx, score in zip(indices[0], scores[0]) if idx >= 0}
        from sklearn.metrics.pairwise import cosine_similarity
        q = self._tfidf.transform([query])
        values = cosine_similarity(q, self._matrix).ravel()
        return {int(i): float(score) for i, score in enumerate(values)}
