import json
from gizzverse_explorer.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, LYRICS_DIR
)


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    current: list[str] = []
    current_len = 0

    for line in text.splitlines():
        line_len = len(line) + 1

        if current and current_len + line_len > chunk_size:
            chunks.append("\n".join(current).strip())

            tail: list[str] = []
            tail_len = 0

            for previous in reversed(current):
                if tail_len + len(previous) + 1 > overlap:
                    break

                tail.insert(0, previous)
                tail_len += len(previous) + 1

            current, current_len = tail, tail_len

        current.append(line)
        current_len += line_len

    if any(line.strip() for line in current):
        chunks.append("\n".join(current).strip())

    return chunks

def main():
    chunks = []

    for path in sorted(LYRICS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        lyrics = data.get("lyrics", {})

        if lyrics.get("status") != "found":
            continue

        text = lyrics.get("plain_text")

        if not text or not text.strip():
            continue

        track_id = data.get("recording_id") or data["track_id"]

        for index, chunk in enumerate(split_text(text)):
            chunks.append({
                "id": f"{track_id}_{index}",
                "text": chunk,
                "metadata": {
                    "track_id": track_id,
                    "title": data["title"],
                    "artist": data["artist"],
                    "album": data["album"],
                    "release_date": data.get("release_date"),
                    "chunk_index": index
                }
            })

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {CHUNKS_PATH}")

if __name__ == "__main__":
    main()