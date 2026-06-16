from pathlib import Path

import pandas as pd

from music_taste.features import add_features


INPUT_PATH = Path("data/raw/albums_2025.csv")
OUTPUT_PATH = Path("data/processed/albums_2025_features.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input CSV at: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH).loc[:, "ARTIST":]
    df_features = add_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering complete.")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(df_features)}")
    print(f"Output columns: {len(df_features.columns)}")
    print(f"Saved processed data to: {OUTPUT_PATH}")

    print("\nNew feature columns:")
    new_columns = [col for col in df_features.columns if col not in df.columns]
    for col in new_columns:
        print(f"- {col}")


if __name__ == "__main__":
    main()