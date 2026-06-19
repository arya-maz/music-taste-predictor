from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            cleaned_line = line.strip()

            if not cleaned_line or cleaned_line.startswith("#"):
                continue

            if "=" not in cleaned_line:
                continue

            key, value = cleaned_line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(PROJECT_ROOT / ".env")

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "aoty_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "aoty_enriched.csv"
REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "aoty_tags_for_review.csv"
DISCOGS_GENRE_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "discogs_album_genres_cache.json"
METADATA_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "lastfm_album_metadata_cache.json"

LASTFM_API_ROOT = "https://ws.audioscrobbler.com/2.0/"
DISCOGS_API_ROOT = "https://api.discogs.com"
REQUEST_DELAY_SECONDS = 0.25
DISCOGS_REQUEST_DELAY_SECONDS = 1.0
DISCOGS_MAX_RETRIES = 3

FINAL_OUTPUT_COLUMNS = [
    "Artist",
    "Album",
    "Year",
    "Number of tracks",
    "Runtime",
    "Genres",
    "Score",
]


GENRE_TAG_ALIASES = {
    "hip hop": "hip-hop",
    "hiphop": "hip-hop",
    "hip-hop": "hip-hop",
    "r&b": "rnb",
    "rhythm and blues": "rnb",
    "contemporary randb": "contemporary rnb",
    "neo soul": "neo-soul",
    "hardcore hip hop": "hardcore hip-hop",
    "abstract hip hop": "abstract hip-hop",
    "southern hip hop": "southern hip-hop",
    "east coast hip hop": "east coast hip-hop",
    "west coast hip hop": "west coast hip-hop",
    "instrumental hip hop": "instrumental hip-hop",
    "alt pop": "alt-pop",
}

IGNORED_TAGS = {
    "album",
    "albums",
    "albums i own",
    "american",
    "aoty",
    "auto-tagged",
    "best",
    "check",
    "classic",
    "comfort album",
    "favorite",
    "favorites",
    "favourite",
    "favourite albums",
    "favourites",
    "finished",
    "grammy",
    "male vocalists",
    "female vocalists",
    "female vocalist",
    "listen list",
    "lp",
    "masterpiece",
    "mid",
    "my albums",
    "owned",
    "randomvalue",
    "seen live",
    "skipless albums",
    "spotify",
    "title is declarative",
    "usa",
    "vinyl",
    "want to listen",
    "website",
    "wikipedia",
}

IGNORED_TAG_PHRASES = {
    "best of",
    "album of the year",
    "albums of",
    "favorite albums",
    "favourite albums",
    "number one",
    "out of 5",
    "out of five",
}

ACCEPTED_GENRE_TAGS = {
    "alternative metal",
    "alternative rock",
    "ambient",
    "art pop",
    "black metal",
    "boom bap",
    "cloud rap",
    "conscious hip-hop",
    "contemporary rnb",
    "death metal",
    "doom metal",
    "dream pop",
    "east coast hip-hop",
    "electronic",
    "emo",
    "experimental",
    "experimental hip-hop",
    "folk",
    "funk",
    "garage rock",
    "glitch pop",
    "grunge",
    "hard rock",
    "hardcore hip-hop",
    "hardcore punk",
    "heavy metal",
    "hip-hop",
    "indie pop",
    "indie rock",
    "industrial",
    "industrial hip-hop",
    "jazz",
    "jazz rap",
    "metal",
    "metalcore",
    "neo-soul",
    "noise rock",
    "nu metal",
    "pop",
    "pop rap",
    "post-hardcore",
    "post-punk",
    "progressive metal",
    "progressive rock",
    "psychedelic rock",
    "punk",
    "punk rock",
    "rap",
    "rnb",
    "shoegaze",
    "singer-songwriter",
    "sludge metal",
    "soul",
    "southern hip-hop",
    "synthpop",
    "trap",
    "west coast hip-hop",
}

ACCEPTED_TAG_PHRASES = {
    "ambient",
    "blues",
    "core",
    "country",
    "dance",
    "disco",
    "doom",
    "dream",
    "drone",
    "dub",
    "electro",
    "electronic",
    "emo",
    "folk",
    "funk",
    "garage",
    "gaze",
    "grind",
    "grunge",
    "hardcore",
    "hip-hop",
    "hop",
    "house",
    "industrial",
    "jazz",
    "metal",
    "noise",
    "pop",
    "post-",
    "prog",
    "progressive",
    "punk",
    "rap",
    "rnb",
    "rock",
    "shoegaze",
    "soul",
    "synth",
    "techno",
    "trap",
    "wave",
}



def get_lastfm_api_key() -> str:
    api_key = os.getenv("LASTFM_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Missing LASTFM_API_KEY environment variable.\n"
            "Run this in your terminal first:\n"
            'export LASTFM_API_KEY="your_api_key_here"'
        )

    return api_key


def get_discogs_user_token() -> str:
    user_token = os.getenv("DISCOGS_USER_TOKEN")

    if not user_token:
        raise EnvironmentError(
            "Missing DISCOGS_USER_TOKEN environment variable.\n"
            "Run this in your terminal first:\n"
            'export DISCOGS_USER_TOKEN="your_discogs_token_here"'
        )

    return user_token


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

# Helper to check if a genre cache entry exists and is missing (empty)
def is_cached_missing_genre(
    cache: dict[str, Any],
    artist: str,
    album: str,
) -> bool:
    cache_key = make_cache_key(artist, album)
    return cache_key in cache and not cache[cache_key]


def make_discogs_headers() -> dict[str, str]:
    return {
        "User-Agent": "MusicTastePredictionApp/1.0",
    }


# Helper for Discogs rate limit retry delay
def get_discogs_retry_delay(error: urllib.error.HTTPError, attempt_number: int) -> float:
    retry_after = error.headers.get("Retry-After")

    if retry_after and retry_after.isdigit():
        return float(retry_after) + 1.0

    return DISCOGS_REQUEST_DELAY_SECONDS * attempt_number


def normalize_tag(tag: str) -> str:
    cleaned_tag = tag.strip().lower()
    cleaned_tag = re.sub(r"\s+", " ", cleaned_tag)
    return GENRE_TAG_ALIASES.get(cleaned_tag, cleaned_tag)


def compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def is_year_or_decade_tag(tag: str) -> bool:
    if re.fullmatch(r"\d{4}", tag):
        return True

    if re.fullmatch(r"\d{2}s", tag):
        return True

    if re.fullmatch(r"\d{4}s", tag):
        return True

    if re.fullmatch(r"\d+", tag):
        return True

    return False


def contains_ignored_phrase(tag: str) -> bool:
    return any(phrase in tag for phrase in IGNORED_TAG_PHRASES)


def matches_artist_or_album_name(tag: str, artist: str, album: str) -> bool:
    compact_tag = compact_text(tag)
    compact_artist = compact_text(artist)
    compact_album = compact_text(album)

    if compact_tag == compact_artist:
        return True

    if compact_tag == compact_album:
        return True

    if compact_tag and compact_tag in compact_artist and len(compact_tag) >= 5:
        return True

    if compact_tag and compact_tag in compact_album and len(compact_tag) >= 5:
        return True

    return False

def is_accepted_music_tag(tag: str) -> bool:
    if tag in ACCEPTED_GENRE_TAGS:
        return True

    compact_tag = compact_text(tag)

    for phrase in ACCEPTED_TAG_PHRASES:
        compact_phrase = compact_text(phrase)

        if compact_phrase and compact_phrase in compact_tag:
            return True

    return False


def is_useful_tag(tag: str, artist: str, album: str) -> bool:
    cleaned_tag = normalize_tag(tag)

    if not cleaned_tag:
        return False

    if len(cleaned_tag) <= 1:
        return False

    if cleaned_tag in IGNORED_TAGS:
        return False

    if is_year_or_decade_tag(cleaned_tag):
        return False

    if contains_ignored_phrase(cleaned_tag):
        return False

    if matches_artist_or_album_name(cleaned_tag, artist, album):
        return False

    if not is_accepted_music_tag(cleaned_tag):
        return False

    return True



def request_discogs_database_search(
    artist: str,
    album: str,
    user_token: str,
    result_type: str,
) -> list[dict[str, Any]] | None:
    params = {
        "artist": artist,
        "release_title": album,
        "type": result_type,
        "token": user_token,
    }

    url = f"{DISCOGS_API_ROOT}/database/search?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=make_discogs_headers())

    for attempt_number in range(1, DISCOGS_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))

            time.sleep(DISCOGS_REQUEST_DELAY_SECONDS)
            return data.get("results", [])
        except urllib.error.HTTPError as error:
            if error.code == 429:
                retry_delay = get_discogs_retry_delay(error, attempt_number)
                print(
                    f"Discogs rate limit hit for {artist} - {album}. "
                    f"Waiting {retry_delay:.1f} seconds before retrying."
                )
                time.sleep(retry_delay)
                continue

            print(
                f"Discogs {result_type} search HTTP error for "
                f"{artist} - {album}: {error.code}"
            )
            time.sleep(DISCOGS_REQUEST_DELAY_SECONDS)
            return []
        except urllib.error.URLError as error:
            print(
                f"Discogs {result_type} search URL error for "
                f"{artist} - {album}: {error.reason}"
            )
            time.sleep(DISCOGS_REQUEST_DELAY_SECONDS)
            return []

    print(
        f"Discogs {result_type} search skipped for {artist} - {album} "
        "after repeated rate-limit responses."
    )
    return None


def request_discogs_resource(
    resource_url: str,
    user_token: str,
) -> dict[str, Any] | None:
    separator = "&" if "?" in resource_url else "?"
    url = f"{resource_url}{separator}{urllib.parse.urlencode({'token': user_token})}"
    request = urllib.request.Request(url, headers=make_discogs_headers())

    for attempt_number in range(1, DISCOGS_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))

            time.sleep(DISCOGS_REQUEST_DELAY_SECONDS)
            return data
        except urllib.error.HTTPError as error:
            if error.code == 429:
                retry_delay = get_discogs_retry_delay(error, attempt_number)
                print(
                    f"Discogs rate limit hit while fetching {resource_url}. "
                    f"Waiting {retry_delay:.1f} seconds before retrying."
                )
                time.sleep(retry_delay)
                continue

            print(f"Discogs resource HTTP error for {resource_url}: {error.code}")
            time.sleep(DISCOGS_REQUEST_DELAY_SECONDS)
            return {}
        except urllib.error.URLError as error:
            print(f"Discogs resource URL error for {resource_url}: {error.reason}")
            time.sleep(DISCOGS_REQUEST_DELAY_SECONDS)
            return {}

    print(
        f"Discogs resource skipped for {resource_url} "
        "after repeated rate-limit responses."
    )
    return None


def request_lastfm_album_info(
    artist: str,
    album: str,
    api_key: str,
) -> dict[str, Any]:
    params = {
        "method": "album.getInfo",
        "artist": artist,
        "album": album,
        "api_key": api_key,
        "format": "json",
        "autocorrect": "1",
    }

    url = f"{LASTFM_API_ROOT}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"Last.fm metadata HTTP error for {artist} - {album}: {error.code}")
        return {}
    except urllib.error.URLError as error:
        print(f"Last.fm metadata URL error for {artist} - {album}: {error.reason}")
        return {}

    if "error" in data:
        error_code = data.get("error")
        message = data.get("message", "Unknown Last.fm error")

        print(f"Last.fm metadata error for {artist} - {album}: {error_code} {message}")
        return {}

    return data.get("album", {})


def extract_album_metadata(album_info: dict[str, Any]) -> dict[str, Any]:
    tracks_data = album_info.get("tracks", {}).get("track", [])

    if isinstance(tracks_data, dict):
        tracks = [tracks_data]
    elif isinstance(tracks_data, list):
        tracks = tracks_data
    else:
        tracks = []

    track_count = len(tracks)
    total_seconds = 0

    for track in tracks:
        if not isinstance(track, dict):
            continue

        duration = track.get("duration")

        if isinstance(duration, int):
            total_seconds += duration
            continue

        if isinstance(duration, str) and duration.isdigit():
            total_seconds += int(duration)

    runtime_minutes = round(total_seconds / 60, 2) if total_seconds else None

    return {
        "track_count": track_count if track_count else None,
        "runtime_minutes": runtime_minutes,
        "metadata_source": "lastfm" if track_count or runtime_minutes else "missing",
    }



def normalize_discogs_genre_name(value: str) -> str:
    cleaned_value = value.strip()
    cleaned_value = re.sub(r"\s+", " ", cleaned_value)
    return cleaned_value


def extract_discogs_genres(release_data: dict[str, Any]) -> list[str]:
    genres = []

    for field_name in ["genres", "styles"]:
        field_values = release_data.get(field_name, [])

        if not isinstance(field_values, list):
            continue

        for value in field_values:
            if not isinstance(value, str):
                continue

            genre_name = normalize_discogs_genre_name(value)

            if genre_name and genre_name not in genres:
                genres.append(genre_name)

    return genres


# Extract genres from a Discogs search result (result from /database/search)
def extract_discogs_genres_from_search_result(
    search_result: dict[str, Any],
) -> list[str]:
    genres = []

    for field_name in ["genre", "style", "genres", "styles"]:
        field_values = search_result.get(field_name, [])

        if isinstance(field_values, str):
            field_values = [field_values]

        if not isinstance(field_values, list):
            continue

        for value in field_values:
            if not isinstance(value, str):
                continue

            genre_name = normalize_discogs_genre_name(value)

            if genre_name and genre_name not in genres:
                genres.append(genre_name)

    return genres



def get_album_genres(
    artist: str,
    album: str,
    user_token: str,
    cache: dict[str, Any],
    refresh_missing: bool = False,
) -> list[str]:
    cache_key = make_cache_key(artist, album)

    if cache_key in cache and (cache[cache_key] or not refresh_missing):
        return cache[cache_key]

    search_results = request_discogs_database_search(
        artist=artist,
        album=album,
        user_token=user_token,
        result_type="master",
    )

    if search_results is None:
        return []

    if not search_results:
        search_results = request_discogs_database_search(
            artist=artist,
            album=album,
            user_token=user_token,
            result_type="release",
        )

    if search_results is None:
        return []

    if not search_results:
        cache[cache_key] = []
        time.sleep(REQUEST_DELAY_SECONDS)
        return []

    best_search_result = search_results[0]
    genres = extract_discogs_genres_from_search_result(best_search_result)

    if genres:
        cache[cache_key] = genres
        time.sleep(REQUEST_DELAY_SECONDS)
        return genres

    resource_url = best_search_result.get("resource_url")

    if not resource_url:
        cache[cache_key] = []
        time.sleep(REQUEST_DELAY_SECONDS)
        return []

    release_data = request_discogs_resource(
        resource_url=resource_url,
        user_token=user_token,
    )

    if release_data is None:
        return []

    genres = extract_discogs_genres(release_data)

    cache[cache_key] = genres
    time.sleep(REQUEST_DELAY_SECONDS)

    return genres


def get_album_metadata(
    artist: str,
    album: str,
    api_key: str,
    cache: dict[str, Any],
) -> dict[str, Any]:
    cache_key = make_cache_key(artist, album)

    if cache_key in cache:
        return cache[cache_key]

    album_info = request_lastfm_album_info(artist, album, api_key)
    metadata = extract_album_metadata(album_info)

    cache[cache_key] = metadata
    time.sleep(REQUEST_DELAY_SECONDS)

    return metadata



def add_enrichment_columns(
    df: pd.DataFrame,
    lastfm_api_key: str,
    discogs_user_token: str,
    genre_cache: dict[str, Any],
    metadata_cache: dict[str, Any],
) -> pd.DataFrame:
    enriched_df = df.copy()

    genre_values_by_index = {}
    genre_source_values_by_index = {}
    track_count_values_by_index = {}
    runtime_values_by_index = {}
    metadata_source_values_by_index = {}

    total_rows = len(enriched_df)
    missing_genre_indices = []
    remaining_indices = []

    for row_index, row in enriched_df.iterrows():
        artist = str(row["artist"])
        album = str(row["album"])

        if is_cached_missing_genre(genre_cache, artist, album):
            missing_genre_indices.append(row_index)
        else:
            remaining_indices.append(row_index)

    row_order = missing_genre_indices + remaining_indices

    if missing_genre_indices:
        print(
            f"Retrying {len(missing_genre_indices)} previously missing "
            "Discogs genre rows first."
        )

    for row_number, row_index in enumerate(row_order, start=1):
        row = enriched_df.loc[row_index]
        artist = str(row["artist"])
        album = str(row["album"])
        refresh_missing_genre = row_index in missing_genre_indices

        print(f"[{row_number}/{total_rows}] Fetching data for {artist} - {album}")

        genres = get_album_genres(
            artist=artist,
            album=album,
            user_token=discogs_user_token,
            cache=genre_cache,
            refresh_missing=refresh_missing_genre,
        )

        metadata = get_album_metadata(
            artist=artist,
            album=album,
            api_key=lastfm_api_key,
            cache=metadata_cache,
        )

        genre_values_by_index[row_index] = ", ".join(genres) if genres else None
        genre_source_values_by_index[row_index] = "discogs" if genres else "missing"
        track_count_values_by_index[row_index] = metadata.get("track_count")
        runtime_values_by_index[row_index] = metadata.get("runtime_minutes")
        metadata_source_values_by_index[row_index] = metadata.get(
            "metadata_source",
            "missing",
        )

        if row_number % 25 == 0:
            save_cache(genre_cache, DISCOGS_GENRE_CACHE_PATH)
            save_cache(metadata_cache, METADATA_CACHE_PATH)

    enriched_df["genres"] = enriched_df.index.map(
        lambda row_index: genre_values_by_index.get(row_index)
    )
    enriched_df["genre_source"] = enriched_df.index.map(
        lambda row_index: genre_source_values_by_index.get(row_index)
    )
    enriched_df["track_count"] = enriched_df.index.map(
        lambda row_index: track_count_values_by_index.get(row_index)
    )
    enriched_df["runtime_minutes"] = enriched_df.index.map(
        lambda row_index: runtime_values_by_index.get(row_index)
    )
    enriched_df["metadata_source"] = enriched_df.index.map(
        lambda row_index: metadata_source_values_by_index.get(row_index)
    )

    return enriched_df


def format_final_output(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.rename(
        columns={
            "artist": "Artist",
            "album": "Album",
            "release_year": "Year",
            "track_count": "Number of tracks",
            "runtime_minutes": "Runtime",
            "genres": "Genres",
            "score": "Score",
        }
    )

    output_df["Number of tracks"] = pd.to_numeric(
        output_df["Number of tracks"],
        errors="coerce",
    ).astype("Int64")

    remaining_columns = [
        column for column in output_df.columns if column not in FINAL_OUTPUT_COLUMNS
    ]

    return output_df[FINAL_OUTPUT_COLUMNS + remaining_columns]


def save_review_file(df: pd.DataFrame) -> None:
    review_columns = [
        "artist",
        "album",
        "release_year",
        "format",
        "score",
        "track_count",
        "runtime_minutes",
        "genres",
        "genre_source",
        "metadata_source",
    ]

    review_df = df[review_columns].copy()
    review_df["track_count"] = pd.to_numeric(
        review_df["track_count"],
        errors="coerce",
    ).astype("Int64")

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(REVIEW_PATH, index=False)


def print_summary(df: pd.DataFrame) -> None:
    total_rows = int(len(df))
    genre_rows = int((df["genre_source"] == "discogs").sum())
    missing_genre_rows = int((df["genre_source"] == "missing").sum())
    genre_percentage = (genre_rows / total_rows) * 100 if total_rows else 0
    metadata_rows = int((df["metadata_source"] == "lastfm").sum())
    metadata_percentage = (metadata_rows / total_rows) * 100 if total_rows else 0

    print()
    print("AOTY enrichment complete")
    print("------------------------")
    print(f"Total rows: {total_rows}")
    print(f"Discogs genre rows: {genre_rows}")
    print(f"Missing genre rows: {missing_genre_rows}")
    print(f"Discogs genre percentage: {genre_percentage:.2f}%")
    print(f"Metadata rows: {metadata_rows}")
    print(f"Metadata percentage: {metadata_percentage:.2f}%")
    print(f"Saved enriched data to: {OUTPUT_PATH}")
    print(f"Saved review file to: {REVIEW_PATH}")
    print(f"Saved Discogs genre cache to: {DISCOGS_GENRE_CACHE_PATH}")
    print(f"Saved metadata cache to: {METADATA_CACHE_PATH}")


def main() -> None:
    lastfm_api_key = get_lastfm_api_key()
    discogs_user_token = get_discogs_user_token()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find cleaned AOTY data at: {INPUT_PATH}\n"
            "Run src/import_aoty.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    genre_cache = load_cache(DISCOGS_GENRE_CACHE_PATH)
    metadata_cache = load_cache(METADATA_CACHE_PATH)

    enriched_df = add_enrichment_columns(
        df=df,
        lastfm_api_key=lastfm_api_key,
        discogs_user_token=discogs_user_token,
        genre_cache=genre_cache,
        metadata_cache=metadata_cache,
    )
    final_output_df = format_final_output(enriched_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_output_df.to_csv(OUTPUT_PATH, index=False)

    save_review_file(enriched_df)
    save_cache(genre_cache, DISCOGS_GENRE_CACHE_PATH)
    save_cache(metadata_cache, METADATA_CACHE_PATH)
    print_summary(enriched_df)


if __name__ == "__main__":
    main()