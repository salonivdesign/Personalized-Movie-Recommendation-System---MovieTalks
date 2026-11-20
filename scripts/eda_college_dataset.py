"""EDA for the college dataset. Creates at least 3 PNG charts."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.college_data import load_college_dataset
ROOT = Path(__file__).resolve().parents[1]
df, report = load_college_dataset(ROOT / "data/raw/dataset.csv")
out = ROOT / "docs/figures/college_eda"; out.mkdir(parents=True, exist_ok=True)

def save(name):
    plt.tight_layout(); plt.savefig(out / name, dpi=140); plt.close()

ax = df["rating"].value_counts().sort_index().plot(kind="bar")
ax.set(title="Rating Distribution", xlabel="Rating", ylabel="Count"); save("01_rating_distribution.png")
ax = df.groupby("user_id").size().plot(kind="hist", bins=15)
ax.set(title="Ratings per User", xlabel="Number of ratings", ylabel="Users"); save("02_ratings_per_user.png")
ax = df.groupby("movie_title").size().sort_values(ascending=False).plot(kind="bar")
ax.set(title="Ratings per Movie Title", xlabel="Movie title", ylabel="Count"); plt.xticks(rotation=35, ha="right"); save("03_ratings_per_movie.png")
ax = df["genre"].fillna("Unknown").value_counts().plot(kind="bar")
ax.set(title="Genre Values in Supplied Rows (metadata is inconsistent)", xlabel="Genre value", ylabel="Rows"); plt.xticks(rotation=35, ha="right"); save("04_genre_values.png")
ax = df.assign(month=df["timestamp"].dt.to_period("M").astype(str)).groupby("month").size().plot(kind="line", marker="o")
ax.set(title="Ratings Over Time", xlabel="Month", ylabel="Rows"); plt.xticks(rotation=45, ha="right"); save("05_ratings_over_time.png")
summary = {"source": "college-provided dataset.csv", "rows_after_deduplication": len(df), "users": int(df.user_id.nunique()), "movies": int(df.movie_id.nunique()), "mean_rating": round(float(df.rating.mean()), 4), "rating_min": float(df.rating.min()), "rating_max": float(df.rating.max()), "note": "Genre, language and release_year vary for the same title in the supplied data; plots reflect recorded values and should not be interpreted as verified movie metadata.", "cleaning_report": report}
import json
(out / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Saved {len(list(out.glob('*.png')))} visualizations to {out}")
print(json.dumps({k:v for k,v in summary.items() if k != 'cleaning_report'}, indent=2))
