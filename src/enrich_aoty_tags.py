from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "aoty_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "aoty_enriched.csv"
REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "aoty_tags_for_review.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "lastfm_album_tags_cache.json"

LASTFM_API_ROOT = "https://ws.audioscrobbler.com/2.0/"
REQUEST_DELAY_SECONDS = 0.25
MAX_TAGS = 3


IGNORED_TAGS = {
    "albums i own",
    "album",
    "albums",
    "favorite",
    "favorites",
    "favourite",
    "favourites",
    "seen live",
    "spotify",
    "lastfm",
    "my albums",
    "owned",
    "want to listen",
    "check out",
    "recommendations",
}


def get_api_key() -> str:
    api_key = os.getenv("LASTFM_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Missing LASTFM_API_KEY environment variable.\n"
            "Run this in your terminal first:\n"
            'export LASTFM_API_KEY="your_api_key_here"'
        )

    return api_key


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, ensure_ascii=False)


def make_cache_key(artist: str, album: str) -> str:
    return f"{artist.strip().lower()}|||{album.strip().lower()}"


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def is_useful_tag(tag: str) -> bool:
    cleaned_tag = normalize_tag(tag)

    if not cleaned_tag:
        return False

    if cleaned_tag in IGNORED_TAGS:
        return False

    if len(cleaned_tag) <= 1:
        return False

    return True


def request_lastfm_album_tags(
    artist: str,
    album: str,
    api_key: str,
) -> list[dict[str, Any]]:
    params = {
        "method": "album.getTopTags",
        "artist": artist,
        "album": album,
        "api_key": api_key,
        "format": "json",
        "autocorrect": "1",
    }

    url = f"{LASTFM_API_ROOT}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    if "error" in data:
        error_code = data.get("error")
        message = data.get("message", "Unknown Last.fm error")

        print(f"Last.fm error for {artist} - {album}: {error_code} {message}")
        return []

    raw_tags = data.get("toptags", {}).get("tag", [])

    if isinstance(raw_tags, dict):
        raw_tags = [raw_tags]

    return raw_tags


def extract_top_tags(raw_tags: list[dict[str, Any]]) -> list[str]:
    tags = []

    for tag_data in raw_tags:
        tag_name = tag_data.get("name", "")
        cleaned_tag = normalize_tag(tag_name)

        if is_useful_tag(cleaned_tag):
            tags.append(cleaned_tag)

    return tags[:MAX_TAGS]


def get_album_tags(
    artist: str,
    album: str,
    api_key: str,
    cache: dict[str, Any],
) -> list[str]:
    cache_key = make_cache_key(artist, album)

    if cache_key in cache:
        return cache[cache_key]

    raw_tags = request_lastfm_album_tags(artist, album, api_key)
    tags = extract_top_tags(raw_tags)

    cache[cache_key] = tags
    time.sleep(REQUEST_DELAY_SECONDS)

    return tags


def add_tag_columns(df: pd.DataFrame, api_key: str, cache: dict[str, Any]) -> pd.DataFrame:
    enriched_df = df.copy()

    tag_1_values = []
    tag_2_values = []
    tag_3_values = []
    tag_source_values = []

    total_rows = len(enriched_df)

    for row_number, (_, row) in enumerate(enriched_df.iterrows(), start=1):
        artist = str(row["artist"])
        album = str(row["album"])

        print(f"[{row_number}/{total_rows}] Fetching tags for {artist} - {album}")

        tags = get_album_tags(
            artist=artist,
            album=album,
            api_key=api_key,
            cache=cache,
        )

        tag_1_values.append(tags[0] if len(tags) > 0 else None)
        tag_2_values.append(tags[1] if len(tags) > 1 else None)
        tag_3_values.append(tags[2] if len(tags) > 2 else None)
        tag_source_values.append("lastfm" if tags else "missing")

        if row_number % 25 == 0:
            save_cache(cache, CACHE_PATH)

    enriched_df["genre_1"] = tag_1_values
    enriched_df["genre_2"] = tag_2_values
    enriched_df["genre_3"] = tag_3_values
    enriched_df["tag_source"] = tag_source_values

    return enriched_df


def save_review_file(df: pd.DataFrame) -> None:
    review_columns = [
        "artist",
        "album",
        "release_year",
        "format",
        "score",
        "genre_1",
        "genre_2",
        "genre_3",
        "tag_source",
    ]

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[review_columns].to_csv(REVIEW_PATH, index=False)


def print_summary(df: pd.DataFrame) -> None:
    total_rows = int(len(df))
    tagged_rows = int((df["tag_source"] == "lastfm").sum())
    missing_rows = int((df["tag_source"] == "missing").sum())
    tagged_percentage = (tagged_rows / total_rows) * 100 if total_rows else 0

    print()
    print("Last.fm tag enrichment complete")
    print("-------------------------------")
    print(f"Total rows: {total_rows}")
    print(f"Tagged rows: {tagged_rows}")
    print(f"Missing rows: {missing_rows}")
    print(f"Tagged percentage: {tagged_percentage:.2f}%")
    print(f"Saved enriched data to: {OUTPUT_PATH}")
    print(f"Saved review file to: {REVIEW_PATH}")
    print(f"Saved cache to: {CACHE_PATH}")


def main() -> None:
    api_key = get_api_key()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find cleaned AOTY data at: {INPUT_PATH}\n"
            "Run src/import_aoty.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    cache = load_cache(CACHE_PATH)

    enriched_df = add_tag_columns(df, api_key, cache)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(OUTPUT_PATH, index=False)

    save_review_file(enriched_df)
    save_cache(cache, CACHE_PATH)
    print_summary(enriched_df)


if __name__ == "__main__":
    main()