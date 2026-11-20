"""Copy the measured result tables from artifacts/RESULTS.md into README.md.

The README contains a block delimited by RESULTS markers. This script replaces
only that block, so numbers in the README always come from a real training run.

Usage:  python scripts/update_readme.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-other-dataset", action="store_true",
                    help=f"allow artifacts trained on a dataset other than '{cfg.DATASET_NAME}'")
    args = ap.parse_args()

    results = cfg.ARTIFACT_DIR / "RESULTS.md"
    metrics = cfg.ARTIFACT_DIR / cfg.METRICS_FILENAME
    readme = cfg.ROOT / "README.md"
    if not results.exists() or not metrics.exists():
        sys.exit("artifacts/RESULTS.md or metrics.json not found. Run: python scripts/train.py")

    # Guard: never copy synthetic-test numbers into the README.
    dataset = json.loads(metrics.read_text(encoding="utf-8")).get("meta", {}).get("dataset", {})
    if dataset.get("synthetic"):
        sys.exit("Refusing: these artifacts were trained on SYNTHETIC test data. "
                 "Train on real MovieLens first (python scripts/train.py).")
    if dataset.get("name") != cfg.DATASET_NAME and not args.allow_other_dataset:
        sys.exit(f"Refusing: artifacts were trained on '{dataset.get('name')}', not "
                 f"'{cfg.DATASET_NAME}'. Re-run with --allow-other-dataset if that is intended.")
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit("README.md has no RESULTS markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{results.read_text(encoding='utf-8')}\n{END}{tail}"
    readme.write_text(new, encoding="utf-8")
    print("README.md results block updated from artifacts/RESULTS.md")


if __name__ == "__main__":
    main()
