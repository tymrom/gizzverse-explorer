from qdrant_client import QdrantClient
from gizzverse_explorer.config import QDRANT_PATH
from gizzverse_explorer.retrieval import load_models, retrieve_chunks

def main():
    dense_model, sparse_model, reranker = load_models()
    client = QdrantClient(path=QDRANT_PATH)

    try:
        question = input("Question: ").strip()

        chunks = retrieve_chunks(question, dense_model, sparse_model, reranker, client)

        print(f"\nQuestion: {question}\n")

        for chunk in chunks:
            print(f"Title: {chunk['title']}")
            print(f"Album: {chunk['album']}")
            print(f"Fusion score: {chunk['fusion_score']:.3f}")
            print(f"Rerank score: {chunk['rerank_score']:.3f}")
            print("-" * 40)

    finally:
        client.close()

if __name__ == "__main__":
    main()