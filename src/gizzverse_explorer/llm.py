from ollama import chat

MODEL_NAME = "qwen3:4b"


SYSTEM_PROMPT = """You are a friendly expert on the band King Gizzard & the Lizard Wizard.
You answer questions using ONLY the song excerpts provided in the Context.

How to answer:
1. Reply in the same language as the user's question.
2. Write 1-3 plain sentences, like a message to a friend. No lists, no headings, no bold, no emojis.
3. Name the songs (with album) that match the question. If several match, name up to 3.
4. Support each song with a short quote copied exactly from the Context (max 10 words).
5. If no excerpt in the Context matches the question, say only: "W podanych fragmentach nie widzę takiej piosenki." (or the equivalent in the user's language).

Never use a song, album or quote that is not present in the Context."""


def generate_answer(question: str, songs: list[dict]):
    context = "\n\n".join(
        (
            f"Song: {song['title']}\n"
            f"Album: {song['album']}\n"
            f"Lyrics:\n{song['lyrics']}"
        )
        for song in songs
    )

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        think=False,
        options={"temperature": 0.2}
    )

    return response.message.content