"""Data loading, cleaning and leakage-safe splitting.

Key design decision: the split is *per user and chronological*. For every user
the oldest ratings are used for training, the next slice for validation and the
newest slice for testing. We never train on a rating that happened after a
rating we evaluate on for the same user.
"""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
ARTICLE_RE = re.compile(r"^(?P<main>.+?), (?P<art>The|A|An)(?P<rest>\s*\(.*\))?$")
# A file with this name in the data folder marks the dataset as synthetic test data.
SYNTHETIC_MARKER = "SYNTHETIC_DATA.txt"


class DataError(ValueError):
    """The dataset is missing, malformed or empty. The message says how to fix it."""


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; found {list(df.columns)}")


def load_raw(data_dir: str | Path):
    """Read MovieLens CSVs. ratings.csv and movies.csv are required."""
    d = Path(data_dir)
    missing = [f for f in ("ratings.csv", "movies.csv") if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {d}. Run: python scripts/download_data.py"
        )
    ratings = _read_csv(d / "ratings.csv")
    movies = _read_csv(d / "movies.csv")
    _require_columns(ratings, ["userId", "movieId", "rating", "timestamp"], "ratings.csv")
    _require_columns(movies, ["movieId", "title", "genres"], "movies.csv")

    # Optional files: if present but unusable, warn and carry on without them.
    tags = _read_optional(d / "tags.csv", ["userId", "movieId", "tag"],
                          ["userId", "movieId", "tag", "timestamp"])
    links = _read_optional(d / "links.csv", ["movieId", "imdbId"],
                           ["movieId", "imdbId", "tmdbId"])
    return ratings, movies, tags, links


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataError(f"{path.name} is empty or not valid CSV: {exc}") from exc


def _read_optional(path: Path, required: list[str], empty_cols: list[str]) -> pd.DataFrame:
    empty = pd.DataFrame(columns=empty_cols)
    if not path.exists():
        return empty
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        warnings.warn(f"{path.name} ignored (unreadable: {exc})", stacklevel=3)
        return empty
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(f"{path.name} ignored: missing columns {missing}", stacklevel=3)
        return empty
    return df


def dataset_fingerprint(data_dir: str | Path) -> dict:
    """Identify the dataset a model was trained on (name, file sizes, SHA-256).

    ``synthetic`` is True when the folder holds the SYNTHETIC_MARKER file that the
    test-fixture generator writes. It is a safety net, not proof of authenticity.
    """
    d = Path(data_dir)
    files = {}
    for name in ("ratings.csv", "movies.csv", "tags.csv", "links.csv"):
        p = d / name
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            files[name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    return {"name": d.name, "files": files, "synthetic": (d / SYNTHETIC_MARKER).exists()}


def _fix_article(title: str) -> str:
    """'Matrix, The' -> 'The Matrix' (MovieLens stores articles at the end)."""
    m = ARTICLE_RE.match(title)
    if not m:
        return title
    return f"{m['art']} {m['main']}{m['rest'] or ''}"


def clean_movies(movies: pd.DataFrame, links: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a tidy catalog sorted by movieId.

    Position in this table is the *item index* (``iidx``) used by every model.
    """
    m = movies.dropna(subset=["movieId", "title"]).copy()
    if m.empty:
        raise DataError("movies.csv has no usable rows (need movieId and title).")
    m["movieId"] = m["movieId"].astype("int64")
    m = m.drop_duplicates("movieId")
    m["title"] = m["title"].astype(str).str.strip()
    m["year"] = pd.to_numeric(m["title"].str.extract(YEAR_RE)[0], errors="coerce")
    bare = m["title"].str.replace(YEAR_RE, "", regex=True).str.strip()
    m["display_title"] = bare.map(_fix_article)
    genres = m["genres"].fillna("").astype(str)
    m["genre_list"] = genres.map(
        lambda s: [] if s in ("", cfg.NO_GENRES_LABEL) else s.split(cfg.GENRE_SEPARATOR))

    m["imdbId"] = np.nan
    if links is not None and len(links):
        imdb = links.drop_duplicates("movieId").set_index("movieId")["imdbId"]
        m["imdbId"] = m["movieId"].map(imdb)

    m = m.sort_values("movieId").reset_index(drop=True)
    return m[["movieId", "title", "display_title", "year", "genre_list", "imdbId"]]


def clean_ratings(ratings: pd.DataFrame, valid_movie_ids) -> tuple[pd.DataFrame, dict]:
    """Drop invalid rows. Returns (clean_df, report_of_what_was_removed)."""
    report = {"rows_raw": int(len(ratings))}
    r = ratings.copy()
    for col in ("userId", "movieId", "rating", "timestamp"):
        r[col] = pd.to_numeric(r[col], errors="coerce")      # junk text -> NaN -> dropped
    r = r.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    report["dropped_null"] = report["rows_raw"] - len(r)

    n = len(r)
    r = r[(r["rating"] >= cfg.RATING_MIN) & (r["rating"] <= cfg.RATING_MAX)]
    report["dropped_out_of_range"] = n - len(r)

    n = len(r)
    r = r[r["movieId"].isin(set(valid_movie_ids))]
    report["dropped_unknown_movie"] = n - len(r)

    n = len(r)
    # Keep the latest rating if a user rated the same movie more than once.
    r = r.sort_values(["timestamp", "userId", "movieId"]).drop_duplicates(
        ["userId", "movieId"], keep="last")
    report["dropped_duplicates"] = n - len(r)

    r = r.astype({"userId": "int64", "movieId": "int64", "rating": "float64",
                  "timestamp": "int64"}).reset_index(drop=True)
    report["rows_clean"] = int(len(r))
    if r.empty:
        raise DataError(f"No valid ratings remain after cleaning: {report}. Check that ratings "
                        f"lie in [{cfg.RATING_MIN}, {cfg.RATING_MAX}] and movieIds match movies.csv.")
    return r, report


def add_item_index(ratings: pd.DataFrame, movie_ids: np.ndarray) -> pd.DataFrame:
    """Attach ``iidx`` = row position of the movie in the catalog."""
    lookup = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    out = ratings.copy()
    out["iidx"] = out["movieId"].map(lookup).astype("int64")
    return out


def temporal_user_split(ratings: pd.DataFrame, val_frac: float, test_frac: float,
                        min_ratings: int):
    """Chronological per-user split -> (train, val, test).

    Users with fewer than ``min_ratings`` ratings go entirely to train.
    Guarantee (checked in tests): for each user, max(train ts) <= min(val ts)
    <= max(val ts) <= min(test ts) when timestamps are distinct.
    """
    r = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    pos = r.groupby("userId").cumcount().to_numpy()
    n = r.groupby("userId")["movieId"].transform("size").to_numpy()
    eligible = n >= min_ratings
    n_test = np.where(eligible, np.maximum(1, np.round(n * test_frac)), 0).astype(int)
    n_val = np.where(eligible, np.maximum(1, np.round(n * val_frac)), 0).astype(int)

    test_mask = pos >= n - n_test
    val_mask = (pos >= n - n_test - n_val) & ~test_mask
    train = r[~(test_mask | val_mask)].reset_index(drop=True)
    val = r[val_mask].reset_index(drop=True)
    test = r[test_mask].reset_index(drop=True)
    return train, val, test


def build_tag_text(tags: pd.DataFrame, allowed_ratings: pd.DataFrame) -> pd.Series:
    """One text blob of tags per movie, using only tags by (user, movie) pairs
    that appear in ``allowed_ratings`` (the data the model is allowed to see).

    This stops a held-out interaction from leaking into the content features.
    """
    if tags is None or tags.empty:
        return pd.Series(dtype=object)
    t = tags.dropna(subset=["userId", "movieId", "tag"]).copy()
    t["userId"] = t["userId"].astype("int64")
    t["movieId"] = t["movieId"].astype("int64")
    pairs = allowed_ratings[["userId", "movieId"]].drop_duplicates()
    t = t.merge(pairs, on=["userId", "movieId"], how="inner")
    t["tag"] = t["tag"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)
    return t.groupby("movieId")["tag"].apply(lambda s: " ".join(s))
