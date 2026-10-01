from typing import Dict, List

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    MIN_SIMILARITY,
    TOP_K,
)
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    """Retrieve relevant document chunks for a user question."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore(
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATA_PATH,
        )

        self.vector_store.load()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Retrieve top-k chunks.

        Results below the minimum similarity threshold
        are removed to reduce unsupported answers.
        """

        query_embedding = (
            self.embedding_model.encode_query(query)
        )

        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        results = []

        for score, metadata in raw_results:
            if score < MIN_SIMILARITY:
                continue

            result = dict(metadata)
            result["score"] = round(score, 4)

            results.append(result)

        return results