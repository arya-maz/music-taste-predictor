from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/albums_2025_features.csv")

BINARY_FEATURES = [
    "LONG_TRACKLIST",
    "FEW_TRACKS",
    "SHORT_ALBUM",
    "LONG_ALBUM",
    "SHORT_TRACK_DENSE",
    "MODERN_ALBUM",
]

# Define the order to avoid default alphabetical ordering
SCORE_TIER_ORDER = [
    "love",
    "like",
    "moderately_like",
    "neutral",
    "dislike",
]

LIKE_LABEL_ORDER = [
    "like",
    "neutral",
    "dislike",
]

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(" " * ((80 - len(title)) // 2) + title)
    print("=" * 80)

def summarize_score_distribution(df: pd.DataFrame) -> None:
    print_section("Score Distribution")

    print(df["SCORE"].describe().round(2))

    print("\nScore tier counts:")
    print(df["SCORE_TIER"].value_counts().reindex(SCORE_TIER_ORDER, fill_value=0))

    print("\nThree-label counts:")
    print(df["LIKE_LABEL"].value_counts().reindex(LIKE_LABEL_ORDER, fill_value=0))


def summarize_binary_features(df: pd.DataFrame) -> None:
    print_section("Binary Feature Score Averages")

    for feature in BINARY_FEATURES:
        print("\n")
        summary = (
            df.groupby(feature)["SCORE"]
            .agg(["count", "mean", "median", "min", "max"])
            .round(2)
        )
        print(summary)


def summarize_genres(df: pd.DataFrame) -> None:
    print_section("Top Genre_1 Score Averages")

    genre_summary = (
        df.groupby("GENRE_1")["SCORE"]
        .agg(["count", "mean", "median"])
        .sort_values(by=["mean", "count"], ascending=[False, False])
        .round(2)
    )

    print(genre_summary.head(20))


def summarize_decades(df: pd.DataFrame) -> None:
    print_section("Decade Score Averages")

    decade_summary = (
        df.groupby("DECADE")["SCORE"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_index()
        .round(2)
    )

    print(decade_summary)


def summarize_track_structure(df: pd.DataFrame) -> None:
    print_section("Track Structure")

    print("\nAverage score by track count:")
    track_summary = (
        df.groupby("TRACKS")["SCORE"]
        .agg(["count", "mean", "median"])
        .sort_index()
        .round(2)
    )
    print(track_summary)

    print("\nLowest average track lengths:")
    print(
        df[
            [
                "ARTIST",
                "ALBUM",
                "TRACKS",
                "RUNTIME_MIN",
                "AVG_TRACK_LENGTH",
                "SCORE",
            ]
        ]
        .sort_values("AVG_TRACK_LENGTH")
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest average track lengths:")
    print(
        df[
            [
                "ARTIST",
                "ALBUM",
                "TRACKS",
                "RUNTIME_MIN",
                "AVG_TRACK_LENGTH",
                "SCORE",
            ]
        ]
        .sort_values("AVG_TRACK_LENGTH", ascending=False)
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find processed data at {DATA_PATH}. "
            "Run build_features.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded processed dataset: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    summarize_score_distribution(df)
    summarize_binary_features(df)
    summarize_genres(df)
    summarize_decades(df)
    summarize_track_structure(df)


if __name__ == "__main__":
    main()