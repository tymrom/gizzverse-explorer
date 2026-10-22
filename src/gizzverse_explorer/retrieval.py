import json
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from gizzverse_explorer.indexing import COLLECTION_NAME

LYRICS_DIR = Path("data/raw/lyrics")

def retrieve_songs(question: str, model: SentenceTransformer, client: QdrantClient, limit: int = 6) -> list[dict]:
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    results = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=20, with_payload=True).points

    songs = []
    seen_tracks = set()

    for result in results:
        payload = result.payload or {}
        recording_id = payload.get("track_id")

        if not recording_id or recording_id in seen_tracks:
            continue

        path = LYRICS_DIR / f"{recording_id}.json"

        if not path.exists():
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        lyrics = data.get("lyrics", {}).get("plain_text")

        if not lyrics:
            continue

        songs.append({
            "title": data["title"],
            "album": data["album"],
            "lyrics": lyrics,
            "score": result.score
        })

        seen_tracks.add(recording_id)

        if len(songs) >= limit:
            break

    return songs