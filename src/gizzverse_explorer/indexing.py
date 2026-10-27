import json
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient, models
from gizzverse_explorer.config import (
    CHUNKS_PATH, COLLECTION_NAME, DENSE_MODEL_NAME, QDRANT_PATH, SPARSE_MODEL_NAME
)
from gizzverse_explorer.model_loader import (
    embed_documents, load_dense, load_sparse, report_truncation, to_sparse_vector, build_document_text
)

def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if not chunks:
        print("No chunks found")

        return

    print(f"Loading embedding model {DENSE_MODEL_NAME} and {SPARSE_MODEL_NAME}")

    dense_model = load_dense()
    sparse_model = load_sparse()

    texts = [build_document_text(chunk) for chunk in chunks]
    report_truncation(dense_model, texts)

    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    dense_embeddings = embed_documents(dense_model, texts)
    sparse_embeddings = list(
        sparse_model.embed(texts, batch_size=1)
    )

    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_embeddings.shape[1],
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )

    points = []

    for chunk, dense_emb, sparse_emb in zip(chunks, dense_embeddings, sparse_embeddings):
        point_id = str(uuid5(NAMESPACE_URL, chunk["id"]))

        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_emb.tolist(),
                    "sparse": to_sparse_vector(sparse_emb)
                },
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            )
        )

    print("Saving embeddings to Qdrant...")

    for start in range(0, len(points), 100):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + 100]
        )

    print(f"Indexed {len(points)} chunks successfully.")

    client.close()

if __name__ == "__main__":
    main()