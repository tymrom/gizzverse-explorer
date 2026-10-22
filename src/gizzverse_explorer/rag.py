from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from gizzverse_explorer.indexing import (
    MODEL_NAME,
    QDRANT_PATH
)
from gizzverse_explorer.llm import generate_answer
from gizzverse_explorer.retrieval import retrieve_songs

def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=QDRANT_PATH, check_compatibility=False)

    print("Explorer ready.")

    try:
        while True:
            question = input("Ask a question (or 'exit'): ").strip()

            if question.lower() == "exit":
                break

            if not question:
                continue

            songs = retrieve_songs(question, model, client)

            if not songs:
                print("No relevant songs found.\n")
                continue

            print("\nGenerating answer...\n")

            answer = generate_answer(question, songs)

            print(answer)
            print(f"\nSources:")

            for song in songs:
                print(f"- {song['title']} ({song['album']})")

            print()

    finally:
        client.close()

if __name__ == "__main__":
    main()