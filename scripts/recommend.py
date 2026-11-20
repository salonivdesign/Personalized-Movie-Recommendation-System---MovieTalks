"""Command-line top-N recommendations (no web server needed).

Uses the saved model from `python scripts/train.py`.

  python scripts/recommend.py --user 12 --n 10          # an existing user in the training data
  python scripts/recommend.py --popular --n 10          # popularity baseline
  python scripts/recommend.py --similar 1 --n 10        # movies similar to movieId 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from recsys.service import RecommenderService, ServiceError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--user", type=int, help="existing user id from the training data")
    g.add_argument("--popular", action="store_true", help="popularity baseline list")
    g.add_argument("--similar", type=int, metavar="MOVIE_ID", help="movies similar to this movieId")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--artifact-dir", type=Path, default=cfg.ARTIFACT_DIR)
    a = ap.parse_args()
    try:
        svc = RecommenderService.load(a.artifact_dir)
        if a.user is not None:
            out = svc.recommend_for_user(a.user, a.n)
        elif a.popular:
            out = svc.popular(a.n)
        else:
            out = svc.similar(a.similar, a.n)
    except (FileNotFoundError, ServiceError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    h = svc.health()
    if h.get("synthetic_data"):
        print("NOTE: model was trained on SYNTHETIC test data, not MovieLens.\n")
    print(f"{out['mode']}: {out['mode_description']}")
    for k, m in enumerate(out["items"], 1):
        pr = f"  pred {m['predicted_rating']:.2f}" if m.get("predicted_rating") is not None else ""
        print(f"{k:>2}. {m['title']}  [{', '.join(m['genres'])}]{pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
