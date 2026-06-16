from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/albums_2025_features.csv")
MODEL_OUTPUT_PATH = Path("models/catboost_score_model.cbm")


FEATURE_COLUMNS = [
    "ARTIST",
    "YEAR",
    "TRACKS",
    "RUNTIME_MIN",
    "AVG_TRACK_LENGTH",
    "GENRE_1",
    "GENRE_2",
    "GENRE_3",
    "ALBUM_AGE",
    "DECADE",
    "LONG_TRACKLIST",
    "FEW_TRACKS",
    "SHORT_ALBUM",
    "LONG_ALBUM",
    "SHORT_TRACK_DENSE",
    "MODERN_ALBUM",
]

CATEGORICAL_COLUMNS = [
    "ARTIST",
    "GENRE_1",
    "GENRE_2",
    "GENRE_3",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find processed data at {DATA_PATH}. "
            "Run build_features.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["SCORE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_COLUMNS,
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model training complete.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Mean Absolute Error: {mae:.2f} points")
    print(f"R² Score: {r2:.3f}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUTPUT_PATH)

    print(f"Saved model to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()