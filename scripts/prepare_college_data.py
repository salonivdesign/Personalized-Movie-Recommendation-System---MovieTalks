"""Clean the supplied college dataset without changing the original file."""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.college_data import load_college_dataset
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "raw" / "dataset.csv"
out = ROOT / "data" / "processed" / "dataset.csv"
cleaned, report = load_college_dataset(source, out)
(ROOT / "docs" / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"Cleaned dataset: {out}")
