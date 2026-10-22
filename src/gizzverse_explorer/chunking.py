import json
from pathlib import Path

LYRICS_DIR = Path("data/raw/lyrics")
OUTPUT_PATH = Path("data/processed/chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def split_text(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks = []
    current = ""

    for line in lines:
        if current and len(current) + len(line) > CHUNK_SIZE:
            chunks.append(current.strip())
            current = current[-CHUNK_OVERLAP:] + line
        else:
            current += line

    if current.strip():
        chunks.append(current.strip())

    return chunks

def main():
    chunks = []

    for path in LYRICS_DIR.glob("*.json"):
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()