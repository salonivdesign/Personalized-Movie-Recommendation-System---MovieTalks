"""Train and save the existing API's model bundle from the college dataset.

Usage: python scripts/train_college_api.py
The supplied dataset has inconsistent metadata; for serving only, the modal
(non-null) genre/year per title are used. This is a convenience for the demo,
not a claim that those values are verified movie facts.
"""
from __future__ import annotations
import json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import scipy, sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg
from recsys.college_data import load_college_dataset
from recsys.pipeline import fit_models, _atomic_write


def main():
    source = ROOT / "data" / "raw" / "dataset.csv"
    if not source.exists():
        raise SystemExit(f"College dataset not found: {source}\nPut your dataset.csv at data/raw/dataset.csv")

    # Clean using the same policy as the project evaluation scripts.
    cleaned_path = ROOT / "data" / "processed" / "dataset.csv"
    df, clean_report = load_college_dataset(source, cleaned_path)
    if df.empty or df.movie_id.nunique() < 2:
        raise SystemExit("Need at least two distinct movies to train the demo model.")

    # Convert to the MovieLens-like internal schema used by the existing service.
    titles = sorted(df.movie_title.unique(), key=str.casefold)
    movie_ids = {title: i + 1 for i, title in enumerate(titles)}
    movies_rows = []
    for title in titles:
        group = df[df.movie_title == title]
        def mode_or_default(series, default):
            values = series.dropna()
            if values.empty:
                return default
            counts = values.value_counts()
            # deterministic tie break by string form
            max_count = counts.max()
            return sorted([v for v, n in counts.items() if n == max_count], key=lambda x: str(x))[0]
        genre = mode_or_default(group.genre, "Unknown")
        year = mode_or_default(group.release_year, np.nan)
        genres = [] if str(genre).strip().lower() in {"unknown", "nan", "<na>", ""} else [str(genre)]
        movies_rows.append({"movieId": movie_ids[title], "title": title, "display_title": title,
                            "year": float(year) if pd.notna(year) else np.nan,
                            "genre_list": genres, "imdbId": np.nan})
    movies = pd.DataFrame(movies_rows).sort_values("movieId").reset_index(drop=True)

    ratings = df[["user_id", "movie_title", "rating", "timestamp"]].copy()
    ratings["userId"] = ratings.user_id.astype(int)
    ratings["movieId"] = ratings.movie_title.map(movie_ids).astype(int)
    ratings["rating"] = ratings.rating.astype(float)
    ratings["timestamp"] = pd.to_datetime(ratings.timestamp).astype("int64") // 10**9
    lookup = pd.Series(np.arange(len(movies)), index=movies.movieId)
    ratings["iidx"] = ratings.movieId.map(lookup).astype(int)
    ratings = ratings[["userId", "movieId", "rating", "timestamp", "iidx"]].reset_index(drop=True)

    tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    pop, content, cf = fit_models(ratings, movies, tags, dict(cfg.CF_PARAMS), cfg.SEED)
    from recsys.hybrid import HybridRecommender
    hybrid = HybridRecommender(movies, pop, content, cf)

    bundle = {
        "movies": movies,
        "hybrid": hybrid,
        "ratings": {"userId": ratings.userId.to_numpy(), "iidx": ratings.iidx.to_numpy(),
                    "rating": ratings.rating.to_numpy(dtype=np.float32),
                    "timestamp": ratings.timestamp.to_numpy()},
        "meta": {
            "bundle_version": 1,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": cfg.SEED,
            "dataset": {"name": "college-provided dataset.csv", "synthetic": False,
                        "note": "Serving demo trained on college dataset; metadata is inconsistent and modal values are used."},
            "rating_scale": [cfg.RATING_MIN, cfg.RATING_MAX, cfg.RATING_STEP],
            "cf_params": dict(cfg.CF_PARAMS), "hybrid_weights": dict(cfg.DEFAULT_HYBRID_WEIGHTS),
            "cold_start_weights": dict(cfg.COLD_START_WEIGHTS),
            "min_ratings_for_cf": cfg.MIN_RATINGS_FOR_CF,
            "n_users": int(ratings.userId.nunique()), "n_movies": int(len(movies)),
            "n_ratings": int(len(ratings)),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        },
    }
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(cfg.ARTIFACT_DIR / cfg.BUNDLE_FILENAME,
                  lambda p: joblib.dump(bundle, p, compress=3))
    metrics = {"dataset": "college-provided dataset.csv", "note": "See docs/evaluation_results.csv for held-out evaluation. Metadata inconsistencies limit content recommendations.",
               "training_rows": int(len(ratings)), "users": int(ratings.userId.nunique()), "movies": int(len(movies)),
               "trained_at_utc": bundle["meta"]["trained_at_utc"]}
    _atomic_write(cfg.ARTIFACT_DIR / cfg.METRICS_FILENAME,
                  lambda p: p.write_text(json.dumps(metrics, indent=2), encoding="utf-8"))
    print(f"Saved API model bundle: {cfg.ARTIFACT_DIR / cfg.BUNDLE_FILENAME}")
    print(f"Trained on {len(ratings)} ratings, {ratings.userId.nunique()} users, {len(movies)} movies.")
    print("Next: uvicorn api.main:app --reload")

if __name__ == "__main__":
    main()
