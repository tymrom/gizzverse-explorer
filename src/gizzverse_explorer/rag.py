from qdrant_client import QdrantClient
from gizzverse_explorer.config import QDRANT_PATH
from gizzverse_explorer.llm import generate_answer
from gizzverse_explorer.retrieval import retrieve_chunks, load_models

def answer_question(question, dense_model, sparse_model, reranker, client):
    chunks = retrieve_chunks(question, dense_model, sparse_model, reranker, client)

    if not chunks:
        return None, []

    return generate_answer(question, chunks), chunks

def main():
    print("Loading embedding models and reranker...")
    dense_model, sparse_model, reranker = load_models()
    client = QdrantClient(path=QDRANT_PATH)

    print("Explorer ready.")

    try:
        while True:
            question = input("Ask a question (or 'exit'): ").strip()

            if question.lower() == "exit":
                break

            if not question:
                continue

            print("\nGenerating answer...\n")
            answer, chunks = answer_question(
                question, dense_model, sparse_model, reranker, client
            )

            if not chunks:
                print("No relevant songs found.\n")
                continue

            print(answer)
            print("\nSources:")

            seen_songs = set()

            for chunk in chunks:
                song_key = (chunk["title"], chunk["album"])

                if song_key in seen_songs:
                    continue

                seen_songs.add(song_key)
                print(f"- {chunk['title']} ({chunk['album']})")

            print()

    finally:
        client.close()

if __name__ == "__main__":
    main()