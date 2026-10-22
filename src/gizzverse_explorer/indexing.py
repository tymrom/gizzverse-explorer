import json
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import torch

CHUNKS_PATH = Path("data/processed/chunks.json")
QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "kglw_lyrics"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if not chunks:
        print("No chunks found")

        return

    print(f"Loading embedding model {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"Generating embeddings for {len(chunks)} chunks...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=model.get_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid5(NAMESPACE_URL, chunk["id"]))

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
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

    print(f"Indexed {len(points)} chunks succesfully.")

    client.close()

if __name__ == "__main__":
    main()