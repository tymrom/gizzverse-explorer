import json
import time
from pathlib import Path
import requests

ARTIST = "King Gizzard & the Lizard Wizard"
BASE_URL = "https://musicbrainz.org/ws/2/"

HEADERS = {
    "User-Agent": "GizzverseExplorer/0.1 (your-email@example.com)"
}

OUTPUT = Path("data/raw/discography.json")

def get_json(endpoint, params=None):
    time.sleep(1.1)

    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params={**(params or {}), "fmt": "json"},
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    return response.json()

def find_artist():
    data = get_json(
        "artist",
        {"query": f'artist:"{ARTIST}"', "limit": 10}
    )

    matches = [artist for artist in data["artists"] if artist["name"].casefold() == ARTIST.casefold()]

    if not matches:
        raise RuntimeError("Artist not found")

    return matches[0]["id"]

def get_albums(artist_id):
    albums = []
    offset = 0

    while True:
        data = get_json(
            "release-group",
            {
                "artist": artist_id,
                "type": "album",
                "limit": 100,
                "offset": offset
            }
        )

        batch = data.get("release-groups", [])
        albums.extend(batch)

        offset += len(batch)
        if not batch or offset >= data.get("release-group-count", 0):
            break

    return [album for album in albums if album.get("primary-type") == "Album" and not album.get("secondary-types")]

def get_tracks(release_group_id):
    releases = get_json(
        f"release-group/{release_group_id}",
        {"inc": "releases"}
    ).get("releases", [])

    official = [release for release in releases if release.get("status") == "official"]

    if not official:
        return []

    release_id = official[0]["id"]

    data = get_json(
        f"release/{release_id}",
        {"inc": "recordings"}
    )

    tracks = []

    for medium in data.get("media", []):
        for track in medium.get("tracks", []):
            tracks.append({
                "id": track["id"],
                "recording_id": track.get("recording", {}).get("id"),
                "title": track["title"],
                "track_number": track.get("number"),
                "disc_number": medium.get("position"),
                "duration_ms": track.get("length")
            })

    return tracks

def main():
    artist_id = find_artist()
    print(f"Artist Id: {artist_id}")

    albums = get_albums(artist_id)
    print(f"Found {len(albums)} albums")

    result = []

    for album in albums:
        print(f"Fetching: {album['title']}")

        result.append({
            "id": album["id"],
            "title": album["title"],
            "release_date": album.get("first-release-date"),
            "tracks": get_tracks(album["id"])
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    total_tracks = sum(len(a["tracks"]) for a in result)

    print(f"Saved {len(result)} albums")
    print(f"Saved total of {total_tracks} track entries")

if __name__ == "__main__":
    main()

    