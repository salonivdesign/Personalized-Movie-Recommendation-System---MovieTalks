"""Download and unpack MovieLens (ml-latest-small, ~1 MB) from GroupLens.

Usage:  python scripts/download_data.py
        python scripts/download_data.py --zip path\\to\\ml-latest-small.zip   (already downloaded)
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

REQUIRED = ("ratings.csv", "movies.csv")


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract while refusing paths that would escape ``dest`` (zip-slip)."""
    dest = dest.resolve()
    for member in zf.namelist():
        target = (dest / member).resolve()
        if dest not in target.parents and target != dest:
            raise RuntimeError(f"Unsafe path in zip: {member}")
    zf.extractall(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="use a local zip instead of downloading")
    ap.add_argument("--force", action="store_true", help="re-download even if data exists")
    args = ap.parse_args()

    target = cfg.RAW_DIR / cfg.DATASET_NAME   # where the GroupLens zip unpacks to
    if all((target / f).exists() for f in REQUIRED) and not args.force:
        print(f"Data already present in {target}. Use --force to re-download.")
        return

    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = args.zip or (cfg.RAW_DIR / f"{cfg.DATASET_NAME}.zip")
    if not args.zip:
        print(f"Downloading {cfg.DATASET_URL} ...")
        try:
            urllib.request.urlretrieve(cfg.DATASET_URL, zip_path)
        except Exception as exc:  # network errors vary widely
            sys.exit(f"Download failed: {exc}\nDownload the zip manually from "
                     f"{cfg.DATASET_URL} and run: python scripts/download_data.py --zip <file>")
    with zipfile.ZipFile(zip_path) as zf:
        safe_extract(zf, cfg.RAW_DIR)
    if not args.zip:
        zip_path.unlink(missing_ok=True)

    missing = [f for f in REQUIRED if not (target / f).exists()]
    if missing:
        sys.exit(f"Extraction finished but {missing} not found in {target}")
    print(f"Ready: {target}")
    print("Files:", ", ".join(sorted(p.name for p in target.glob('*.csv'))))


if __name__ == "__main__":
    main()
