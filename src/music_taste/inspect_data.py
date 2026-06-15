from pathlib import Path
import pandas as pd


CSV_PATH = Path("data/raw/albums_2025.csv")

REQUIRED_COLUMNS = [
    "ORDER",
    "ARTIST",
    "ALBUM",
    "YEAR",
    "TRACKS",
    "RUNTIME_MIN",
    "AVG_TRACK_LENGTH",
    "GENRE_1",
    "GENRE_2",
    "GENRE_3",
    "GENRE_RAW",
    "SCORE",
]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find CSV at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print("\nCSV loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        print("\nMissing required columns:")
        for col in missing_columns:
            print(f"- {col}")
    else:
        print("\nAll required columns are present.")

    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")

    print("\nMissing values by column:")
    print(df.isna().sum())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()