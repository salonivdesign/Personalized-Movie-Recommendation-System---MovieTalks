"""Train, evaluate and save all model artifacts. Run this BEFORE starting the API.

Usage:  python scripts/train.py [--data-dir DIR] [--artifact-dir DIR] [--no-tune] [--seed N]

Outputs (artifacts/): model_bundle.joblib, metrics.json, RESULTS.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
from recsys.pipeline import run_training  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cfg.DEFAULT_DATA_DIR)
    ap.add_argument("--artifact-dir", type=Path, default=cfg.ARTIFACT_DIR)
    ap.add_argument("--no-tune", action="store_true",
                    help="skip the CF hyperparameter grid search (faster)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()
    run_training(args.data_dir, args.artifact_dir, tune_cf=not args.no_tune, seed=args.seed)
    print(f"\nNext: uvicorn api.main:app --reload   (then open http://127.0.0.1:8000)")


if __name__ == "__main__":
    main()
