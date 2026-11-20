"""Exploratory data analysis: summary statistics + figures.

Everything is computed from the cleaned data, so the numbers in
``eda_summary.json`` and the PNGs always reflect the dataset you actually have.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT = "#4f6df5"


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_eda(ratings: pd.DataFrame, movies: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_users, n_movies_rated = ratings["userId"].nunique(), ratings["movieId"].nunique()
    per_user = ratings.groupby("userId").size()
    per_movie = ratings.groupby("movieId").size().sort_values(ascending=False)
    top_decile = per_movie.head(max(1, len(per_movie) // 10)).sum() / per_movie.sum()

    exploded = movies.explode("genre_list").dropna(subset=["genre_list"])
    genre_counts = exploded["genre_list"].value_counts()
    rated = ratings.merge(exploded[["movieId", "genre_list"]], on="movieId")
    genre_mean = rated.groupby("genre_list")["rating"].agg(["mean", "size"])

    summary = {
        "n_ratings": int(len(ratings)), "n_users": int(n_users),
        "n_movies_in_catalog": int(len(movies)), "n_movies_rated": int(n_movies_rated),
        "sparsity": float(1 - len(ratings) / (n_users * n_movies_rated)),
        "mean_rating": float(ratings["rating"].mean()),
        "rating_value_counts": {str(k): int(v) for k, v in
                                ratings["rating"].value_counts().sort_index().items()},
        "ratings_per_user": {"min": int(per_user.min()), "median": float(per_user.median()),
                             "max": int(per_user.max())},
        "ratings_per_movie": {"median": float(per_movie.median()), "max": int(per_movie.max())},
        "share_movies_with_lt5_ratings": float((per_movie < 5).mean()),
        "share_of_ratings_on_top_10pct_movies": float(top_decile),
        "movies_without_genres": int((movies["genre_list"].str.len() == 0).sum()),
        "movies_without_year": int(movies["year"].isna().sum()),
        "rating_date_range": [
            pd.to_datetime(ratings["timestamp"].min(), unit="s").strftime("%Y-%m-%d"),
            pd.to_datetime(ratings["timestamp"].max(), unit="s").strftime("%Y-%m-%d")],
        "genre_counts": {k: int(v) for k, v in genre_counts.items()},
        "genre_mean_rating": {k: round(float(v), 4) for k, v in genre_mean["mean"].items()},
    }

    # 1. rating value distribution
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vc = ratings["rating"].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color=ACCENT)
    ax.set(title="Rating distribution", xlabel="Stars", ylabel="Number of ratings")
    _save(fig, out_dir / "01_rating_distribution.png")

    # 2. ratings per user
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(per_user.values, bins=np.logspace(np.log10(per_user.min()), np.log10(per_user.max()), 40),
            color=ACCENT)
    ax.set(xscale="log", title="Ratings per user (log scale)", xlabel="Ratings given",
           ylabel="Users")
    _save(fig, out_dir / "02_ratings_per_user.png")

    # 3. long tail of movies
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(np.arange(1, len(per_movie) + 1), per_movie.values, color=ACCENT)
    ax.set(xscale="log", yscale="log", title="Long tail: ratings per movie (by popularity rank)",
           xlabel="Movie rank", ylabel="Ratings received")
    _save(fig, out_dir / "03_long_tail.png")

    # 4. genres
    fig, ax = plt.subplots(figsize=(6, 4.6))
    gc = genre_counts.sort_values()
    ax.barh(gc.index, gc.values, color=ACCENT)
    ax.set(title="Movies per genre", xlabel="Movies")
    _save(fig, out_dir / "04_genres.png")

    # 5. mean rating per genre
    fig, ax = plt.subplots(figsize=(6, 4.6))
    gm = genre_mean["mean"].sort_values()
    ax.barh(gm.index, gm.values, color=ACCENT)
    ax.set(title="Mean rating per genre", xlabel="Mean stars",
           xlim=(max(0, gm.min() - 0.3), min(5, gm.max() + 0.2)))
    _save(fig, out_dir / "05_genre_mean_rating.png")

    # 6. release decades
    fig, ax = plt.subplots(figsize=(6, 3.6))
    dec = (movies["year"].dropna() // 10 * 10).astype(int).value_counts().sort_index()
    ax.bar(dec.index.astype(str), dec.values, color=ACCENT)
    ax.set(title="Movies by release decade", xlabel="Decade", ylabel="Movies")
    ax.tick_params(axis="x", rotation=60)
    _save(fig, out_dir / "06_release_decades.png")

    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
