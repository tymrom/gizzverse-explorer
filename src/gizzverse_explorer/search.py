from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
from gizzverse_explorer.indexing import (
    COLLECTION_NAME,
    MODEL_NAME,
    QDRANT_PATH
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    client = QdrantClient(path=QDRANT_PATH)

    question = input(str("Question: "))

    query_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
        with_payload=True
    ).points

    print(f"\nQuestion: {question}\n")

    for result in results:
        payload = result.payload or {}

        print(f"Title: {payload.get('title')}")
        print(f"Album: {payload.get('album')}")
        print(f"Score: {result.score:.3f}")
        print("-" * 40)

    client.close()

if __name__ == "__main__":
    main()