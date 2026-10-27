from ollama import chat
from gizzverse_explorer.config import (
    LLM_MODEL_NAME, LLM_NUM_CTX, LLM_TEMPERATURE, LLM_THINK
)



SYSTEM_PROMPT = """
You are Gizzverse Explorer, a music assistant specializing
in King Gizzard & the Lizard Wizard.

Answer questions using the provided song excerpts and metadata.

Be accurate:
- Do not invent facts, song titles, albums, or lyrics.
- Distinguish songs that are about a topic from songs that
  merely mention it.
- Distinguish literal evidence from interpretation.
- Never imply that a partial search covers the entire discography.
- If the evidence is insufficient, say so briefly.
- Answer the exact question, not a loosely related one.
- A partial keyword match is not sufficient evidence.
- If the user asks whether a specific phrase or combination
  of attributes appears in lyrics, verify all requested details.
- If no retrieved excerpt supports the requested claim,
  say so directly. Do not substitute a related fact.
- For comparison questions, explain the relationship between
  the songs rather than merely quoting a shared reference.


Be conversational:
- Respond in English.
- Answer directly, usually in one short paragraph.
- Avoid introductions, headings, tables, lists, and emojis.
- Do not describe your reasoning process.
- Mention only relevant songs.
- Do not explain why unrelated songs were excluded.
- Expand only when the question calls for a deeper explanation.
"""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"Song: {chunk['title']}\n"
        f"Album: {chunk['album']}\n"
        f"Excerpt:\n{chunk['text']}"
        for chunk in chunks
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{build_context(chunks)}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    chat_kwargs = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_ctx": LLM_NUM_CTX,
        },
    }

    if LLM_THINK is not None:
        chat_kwargs["think"] = LLM_THINK

    response = chat(**chat_kwargs)

    return (response.message.content or "").strip()
