from src.chunking import create_chunks
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VECTORSTORE_DIR,
    validate_config,
)
from src.embeddings import EmbeddingModel
from src.ingestion import load_documents
from src.vector_store import VectorStore


def main():
    print("=" * 70)
    print("DOCUMIND - DOCUMENT INDEXING")
    print("=" * 70)

    # API key is not needed for indexing, but configuration
    # validation also checks it, so we only validate directories here.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load documents
    # ---------------------------------------------------------

    page_records = load_documents(DATA_DIR)

    # ---------------------------------------------------------
    # 2. Create chunks
    # ---------------------------------------------------------

    print("\nCreating chunks...")

    chunks = create_chunks(
        page_records=page_records,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError(
            "No chunks were created from the documents."
        )

    print(
        f"[OK] Created {len(chunks)} chunks"
    )

    print(
        f"[INFO] Chunk size: {CHUNK_SIZE} characters"
    )

    print(
        f"[INFO] Chunk overlap: {CHUNK_OVERLAP} characters"
    )

    # ---------------------------------------------------------
    # 3. Generate embeddings in batches
    # ---------------------------------------------------------

    print("\nGenerating embeddings...")

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode_documents(
        texts
    )

    print(
        f"[OK] Generated embeddings with dimension "
        f"{embeddings.shape[1]}"
    )

    # ---------------------------------------------------------
    # 4. Build FAISS vector store
    # ---------------------------------------------------------

    print("\nBuilding FAISS vector database...")

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    vector_store = VectorStore(
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    vector_store.build(
        embeddings=embeddings,
        metadata=metadata,
    )

    # ---------------------------------------------------------
    # 5. Persist vector store
    # ---------------------------------------------------------

    vector_store.save()

    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)

    print(
        f"Documents loaded : {len(set(c['source'] for c in chunks))}"
    )

    print(
        f"Chunks created   : {len(chunks)}"
    )

    print(
        f"Vector dimension : {embeddings.shape[1]}"
    )

    print(
        f"FAISS index      : {FAISS_INDEX_PATH}"
    )

    print(
        f"Metadata         : {METADATA_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()