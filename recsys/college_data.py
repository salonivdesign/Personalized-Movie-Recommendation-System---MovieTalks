"""Loader and conservative cleaner for the college-provided movie ratings CSV."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = ["user_id", "movie_title", "genre", "language", "release_year", "rating", "timestamp"]


def load_college_dataset(path: str | Path, cleaned_path: str | Path | None = None):
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    report = {"rows_input": int(len(df)), "missing_required_values_removed": 0,
              "invalid_ratings_removed": 0, "duplicate_user_title_rows_removed": 0,
              "metadata_inconsistencies": {}}
    df = df[REQUIRED].copy()
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["movie_title", "genre", "language"]:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace("", pd.NA)
    before = len(df)
    df = df.dropna(subset=["user_id", "movie_title", "rating", "timestamp"])
    report["missing_required_values_removed"] = int(before - len(df))
    before = len(df)
    df = df[df["rating"].between(1, 5)]
    report["invalid_ratings_removed"] = int(before - len(df))
    df["user_id"] = df["user_id"].astype(int)
    df["release_year"] = df["release_year"].astype("Int64")
    # Stable numeric IDs derived from sorted unique titles; original titles remain intact.
    titles = sorted(df["movie_title"].unique(), key=str.casefold)
    title_to_id = {title: i + 1 for i, title in enumerate(titles)}
    df["movie_id"] = df["movie_title"].map(title_to_id).astype(int)
    # Keep the latest timestamp for repeated user/title pairs; ties resolve to last input row.
    df["_row_order"] = range(len(df))
    before = len(df)
    df = df.sort_values(["timestamp", "_row_order"]).drop_duplicates(["user_id", "movie_id"], keep="last")
    report["duplicate_user_title_rows_removed"] = int(before - len(df))
    # Record inconsistencies; don't guess a film's real metadata or silently rewrite it.
    for col in ["genre", "language", "release_year"]:
        counts = df.groupby("movie_title")[col].nunique(dropna=True)
        report["metadata_inconsistencies"][col] = int((counts > 1).sum())
    df = df.sort_values(["user_id", "timestamp", "movie_id"]).drop(columns="_row_order").reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["users"] = int(df["user_id"].nunique())
    report["movies"] = int(df["movie_id"].nunique())
    report["ratings"] = int(len(df))
    report["rating_min"] = float(df["rating"].min()) if len(df) else None
    report["rating_max"] = float(df["rating"].max()) if len(df) else None
    if cleaned_path:
        out = Path(cleaned_path); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, date_format="%Y-%m-%d")
    return df, report
