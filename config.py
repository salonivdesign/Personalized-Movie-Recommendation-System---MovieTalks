"""Central configuration.

Every tunable value lives here so training is reproducible and the README can
point to a single place for "what are the hyperparameters?".

Dataset assumptions (MovieLens `ml-latest*` CSV format)
-------------------------------------------------------
* ratings.csv: userId, movieId, rating, timestamp (Unix seconds)  -- required
* movies.csv:  movieId, title, genres                             -- required
* tags.csv, links.csv                                             -- optional
* ratings lie on RATING_MIN..RATING_MAX in steps of RATING_STEP
* titles end in "(YYYY)" and move articles to the end ("Matrix, The")
* genres are separated by GENRE_SEPARATOR; NO_GENRES_LABEL means "none"
To retrain on another dataset in this format: point RECSYS_DATA_DIR (or
``--data-dir``) at its folder, adjust the scale below if needed, and re-run
``python scripts/train.py``.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Paths -----------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
DATASET_NAME = "ml-latest-small"
DATASET_URL = f"https://files.grouplens.org/datasets/movielens/{DATASET_NAME}.zip"
# Folder training reads from. Override with RECSYS_DATA_DIR to retrain on another
# dataset without editing code.
DEFAULT_DATA_DIR = Path(os.getenv("RECSYS_DATA_DIR", RAW_DIR / DATASET_NAME))
# RECSYS_ARTIFACT_DIR lets tests / deployments redirect where artifacts live.
ARTIFACT_DIR = Path(os.getenv("RECSYS_ARTIFACT_DIR", ROOT / "artifacts"))
FRONTEND_DIR = ROOT / "frontend"
BUNDLE_FILENAME = "model_bundle.joblib"
METRICS_FILENAME = "metrics.json"

# ---- Dataset schema (MovieLens-specific; change here for another dataset) --
RATING_MIN, RATING_MAX, RATING_STEP = 0.5, 5.0, 0.5
GENRE_SEPARATOR = "|"
NO_GENRES_LABEL = "(no genres listed)"

# ---- Reproducibility -------------------------------------------------------
SEED = 42

# ---- Splitting (per-user, chronological) -----------------------------------
# For every user with >= MIN_RATINGS_FOR_SPLIT ratings, the oldest ratings go to
# train, the next slice to validation and the newest slice to test.
MIN_RATINGS_FOR_SPLIT = 5
VAL_FRACTION = 0.10
TEST_FRACTION = 0.20

# ---- Evaluation ------------------------------------------------------------
RELEVANCE_THRESHOLD = 4.0      # a held-out rating >= 4.0 counts as "relevant"
EVAL_KS = (5, 10, 20)
PRIMARY_K = 10
COLD_START_HISTORY_SIZES = (0, 1, 3, 5, 10, 20)

# ---- Models ----------------------------------------------------------------
POP_MIN_VOTES_QUANTILE = 0.60  # "m" in the weighted-rating formula
CF_PARAMS = dict(n_factors=32, reg=0.1, n_iters=12, user_bias_reg=15.0, item_bias_reg=10.0)
CF_TUNING_GRID = dict(n_factors=[16, 32, 64], reg=[0.05, 0.1, 0.2])
CF_MIN_ITEM_RATINGS = 3        # items with fewer training ratings get no CF score

# ---- Hybrid ----------------------------------------------------------------
MIN_RATINGS_FOR_CF = 5         # below this many ratings the hybrid skips CF
COLD_START_WEIGHTS = {"content": 0.6, "pop": 0.4}
DEFAULT_HYBRID_WEIGHTS = {"cf": 0.5, "content": 0.3, "pop": 0.2}  # replaced by validation tuning
HYBRID_WEIGHT_STEP = 0.1
SIMILAR_WEIGHTS = {"content": 0.5, "cf": 0.5}
