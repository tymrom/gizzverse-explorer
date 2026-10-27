import json
from collections import Counter
from gizzverse_explorer.config import DISCOGRAPHY_PATH, LYRICS_DIR


def main() -> None:
    albums = json.loads(
        DISCOGRAPHY_PATH.read_text(encoding="utf-8")
    )

    expected_tracks = sum(len(album["tracks"]) for album in albums)

    found = 0
    missing = []
    empty = []
    invalid = []
    track_ids = []

    files = list(LYRICS_DIR.glob("*.json"))

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid.append(path.name)
            continue

        track_ids.append(data.get("track_id"))

        lyrics = data.get("lyrics", {})
        status = lyrics.get("status")
        text = lyrics.get("plain_text")

        if status == "found":
            if isinstance(text, str) and text.strip():
                found += 1
            else:
                empty.append(data.get("title", path.name))

        elif status == "missing":
            missing.append(
                f"{data.get('album')} - {data.get('title')}"
            )

        else:
            invalid.append(path.name)

    duplicates = [
        track_id
        for track_id, count in Counter(track_ids).items()
        if track_id is not None and count > 1
    ]

    print("\n=== GIZZVERSE DATA VALIDATION ===")
    print(f"Albums: {len(albums)}")
    print(f"Expected track entries: {expected_tracks}")
    print(f"JSON files: {len(files)}")
    print(f"Found lyrics: {found}")
    print(f"Missing lyrics: {len(missing)}")
    print(f"Empty lyrics: {len(empty)}")
    print(f"Invalid files: {len(invalid)}")
    print(f"Duplicate track IDs: {len(duplicates)}")

    if missing:
        print("\n=== MISSING LYRICS ===")
        for title in missing:
            print(f"  - {title}")

    if empty:
        print("\n=== EMPTY LYRICS ===")
        for title in empty:
            print(f"  - {title}")

    if invalid:
        print("\n=== INVALID FILES ===")
        for filename in invalid:
            print(f"  - {filename}")

    if duplicates:
        print("\n=== DUPLICATE TRACK IDS ===")
        for track_id in duplicates:
            print(f"  - {track_id}")


if __name__ == "__main__":
    main()
