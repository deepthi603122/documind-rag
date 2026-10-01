import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class VectorStore:
    """
    Persistent FAISS vector store.

    Uses IndexFlatIP with normalized embeddings,
    which gives cosine-similarity-style retrieval.
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = None
        self.metadata: List[Dict] = []

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ):
        """Build a new FAISS index."""

        if len(embeddings) == 0:
            raise ValueError(
                "Cannot build vector store with no embeddings."
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings must match metadata."
            )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(
            np.ascontiguousarray(
                embeddings.astype("float32")
            )
        )

        self.metadata = metadata

    def save(self):
        """Persist FAISS index and metadata to disk."""

        if self.index is None:
            raise ValueError("Vector index has not been built.")

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def load(self):
        """Load a previously persisted vector store."""

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Tuple[float, Dict]]:
        """Search for the most similar chunks."""

        if self.index is None:
            raise ValueError(
                "Vector store has not been loaded."
            )

        query = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        scores, indices = self.index.search(
            query,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            results.append(
                (
                    float(score),
                    self.metadata[index],
                )
            )

        return results