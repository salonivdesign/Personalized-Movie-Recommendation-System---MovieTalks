"""Run exploratory data analysis. Writes figures + eda_summary.json to docs/figures.

Usage:  python scripts/eda.py [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
from recsys import data as D  # noqa: E402
from recsys.eda import run_eda  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cfg.DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=cfg.ROOT / "docs" / "figures")
    args = ap.parse_args()

    ratings, movies, _, links = D.load_raw(args.data_dir)
    movies = D.clean_movies(movies, links)
    ratings, report = D.clean_ratings(ratings, movies["movieId"])
    print("Cleaning report:", report)
    summary = run_eda(ratings, movies, args.out_dir)
    for key in ("n_users", "n_movies_rated", "n_ratings", "sparsity", "mean_rating",
                "share_movies_with_lt5_ratings", "share_of_ratings_on_top_10pct_movies"):
        print(f"{key:40s} {summary[key]}")
    print(f"Figures + eda_summary.json written to {args.out_dir}")


if __name__ == "__main__":
    main()
