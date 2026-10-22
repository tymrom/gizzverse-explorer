import json
import re
import time
from pathlib import Path
import requests

ARTIST = "King Gizzard & the Lizard Wizard"
DISCOGRAPHY_PATH = Path("data/raw/discography.json")
LYRICS_DIR = Path("data/raw/lyrics")
SEARCH_URL = "https://lrclib.net/api/search"

HEADERS = {
    "User-Agent": "GizzverseExplorer/0.1 (https://github.com/tymrom/gizzverse-explorer)"
}

def normalize_title(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def search_lyrics(title: str) -> list[dict]:
    max_attempts = 20

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                SEARCH_URL,
                params={"q": f"king gizzard {title}"},
                headers=HEADERS,
                timeout=20,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == max_attempts:
                    response.raise_for_status()

                retry_after = response.headers.get("Retry-After")

                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    wait_time = min(2 ** attempt, 60)

                print(
                    f"  HTTP {response.status_code} "
                    f"for '{title}' "
                    f"(attempt {attempt}/{max_attempts}). "
                    f"Retry in {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == max_attempts:
                raise

            wait_time = min(2 ** attempt, 60)
            print(f"  Connection error: {error}. Retry in {wait_time}s...")
            time.sleep(wait_time)

    return []



def find_best_match(
    candidates: list[dict],
    title: str,
    album: str,
) -> dict | None:
    normalized_title = normalize_title(title)
    normalized_album = normalize_title(album)

    for candidate in candidates:
        artist = normalize_title(candidate.get("artistName") or "")
        candidate_title = normalize_title(candidate.get("trackName") or "")
        candidate_album = normalize_title(candidate.get("albumName") or "")

        if "king gizzard" not in artist:
            continue

        if not candidate_title or candidate_title not in normalized_title:
            continue

        if not candidate.get("plainLyrics"):
            continue

        album_matches = (
            candidate_album in normalized_album
            or normalized_album in candidate_album
        ) if candidate_album else False

        if album_matches:
            print(f"    Album matched: {candidate.get('albumName')}")

        return candidate

    return None


def save_track(track: dict, album: dict, match: dict | None) -> None:
    track_id = track.get("recording_id") or track["id"]

    output = {
        "recording_id": track.get("recording_id"),
        "track_id": track["id"],
        "title": track["title"],
        "artist": ARTIST,
        "album": album["title"],
        "release_date": album.get("release_date"),
        "track_number": track.get("track_number"),
        "disc_number": track.get("disc_number"),
        "duration_ms": track.get("duration_ms"),
        "lyrics": {
            "status": "found" if match else "missing",
            "plain_text": match.get("plainLyrics") if match else None,
            "source": "lrclib",
            "source_id": match.get("id") if match else None
        }
    }

    LYRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = LYRICS_DIR / f"{track_id}.json"

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def main():
    albums = json.loads(
        DISCOGRAPHY_PATH.read_text(encoding="utf-8")
    )

    total = sum(len(album["tracks"]) for album in albums)
    processed = 0
    found = 0

    print(f"Found {total} track entries")

    for album in albums:
        print(f"\nAlbum: {album["title"]}")

        for track in album["tracks"]:
            track_id = track.get("recording_id") or track["id"]
            output_path = LYRICS_DIR / f"{track_id}.json"

            if output_path.exists():
                saved = json.loads(output_path.read_text(encoding="utf-8"))

                if saved.get("lyrics", {}).get("status") == "found":
                    print(f"    SKIP: {track['title']}")
                    continue

            try:
                candidates = search_lyrics(track["title"])
                match = find_best_match(candidates, track["title"], album["title"])
                save_track(track, album, match)

                if match:
                    found += 1
                    status = "FOUND"
                else:
                    status = "MISSING"

                print(f"    {status}: {track["title"]}")

            except requests.RequestException as error:
                print(f"    ERROR: {track["title"]} - {error}")

            processed += 1

            time.sleep(1.0)

    print("\nFinished")
    print(f"Newly processed: {processed}")
    print(f"New lyrics found: {found}")

if __name__ == "__main__":
    main()