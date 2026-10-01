from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)


class EmbeddingModel:
    """Wrapper around the Sentence Transformers embedding model."""

    def __init__(self):
        print(
            f"\nLoading embedding model: {EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def encode_documents(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Encode all document chunks in batches.

        This is intentionally a single batched embedding
        operation rather than one API/model call per chunk.
        """

        if not texts:
            return np.empty((0, 0), dtype="float32")

        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query using the same embedding model.
        """

        embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.astype("float32")[0]