"""Training pipeline: clean -> split -> tune -> evaluate -> refit -> save artifacts.

Three model fits happen, each on a different slice, to keep evaluation honest:

  Stage A  fit on TRAIN            tune CF hyperparameters + hybrid weights on VAL
  Stage B  fit on TRAIN+VAL        report final numbers on TEST (never used to tune)
  Stage C  fit on ALL ratings      this is the model that gets saved and served
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

import config as cfg
from recsys import data as D
from recsys import evaluate as E
from recsys.collaborative import MatrixFactorizationCF
from recsys.content import ContentModel
from recsys.hybrid import HybridRecommender
from recsys.popularity import PopularityRecommender

BUNDLE_VERSION = 1


def _atomic_write(path: Path, writer) -> None:
    """Write via a temp file, then rename. A crash mid-write leaves the old file intact."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def fit_models(ratings: pd.DataFrame, movies: pd.DataFrame, tags: pd.DataFrame,
               cf_params: dict, seed: int):
    """Fit popularity, content and CF models on exactly the given ratings."""
    n_items = len(movies)
    pop = PopularityRecommender(cfg.POP_MIN_VOTES_QUANTILE).fit(ratings, n_items)
    tag_text = D.build_tag_text(tags, ratings)       # tags limited to visible interactions
    content = ContentModel().fit(movies, tag_text)
    cf = MatrixFactorizationCF(seed=seed, **cf_params).fit(ratings, n_items)
    return pop, content, cf


def _clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items() if not str(k).startswith("_users")}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_training(data_dir: Path = cfg.DEFAULT_DATA_DIR, artifact_dir: Path = cfg.ARTIFACT_DIR,
                 tune_cf: bool = True, seed: int = cfg.SEED) -> dict:
    t0 = time.time()
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # ---- 1. load + clean -------------------------------------------------------
    raw_ratings, raw_movies, tags, links = D.load_raw(data_dir)
    dataset = D.dataset_fingerprint(data_dir)      # recorded in the artifacts
    movies = D.clean_movies(raw_movies, links)
    ratings, clean_report = D.clean_ratings(raw_ratings, movies["movieId"])
    ratings = D.add_item_index(ratings, movies["movieId"].to_numpy())
    n_items = len(movies)
    _log(f"{len(ratings):,} ratings | {ratings.userId.nunique():,} users | {n_items:,} movies "
         f"(cleaning report: {clean_report})")

    # ---- 2. split ----------------------------------------------------------------
    train, val, test = D.temporal_user_split(
        ratings, cfg.VAL_FRACTION, cfg.TEST_FRACTION, cfg.MIN_RATINGS_FOR_SPLIT)
    _log(f"split -> train {len(train):,} | val {len(val):,} | test {len(test):,}")
    if train.empty:
        raise D.DataError("The training split is empty; check ratings.csv.")
    if val.empty or test.empty:
        _log(f"WARNING: no user has >= {cfg.MIN_RATINGS_FOR_SPLIT} ratings, so validation/test "
             "are empty. Tuning and evaluation are SKIPPED and no metrics will be produced.")
    trainval = pd.concat([train, val], ignore_index=True)

    # ---- 3. Stage A: tune on validation ---------------------------------------------
    cf_params = dict(cfg.CF_PARAMS)
    cf_tuning = []
    if tune_cf and len(val):
        _log("tuning CF hyperparameters on validation RMSE ...")
        cf_params, cf_tuning = E.tune_cf(train, val, n_items, cf_params, cfg.CF_TUNING_GRID, seed)
        _log(f"best CF params: {cf_params}")
    pop, content, cf = fit_models(train, movies, tags, cf_params, seed)
    hybrid = HybridRecommender(movies, pop, content, cf)
    weights, weight_table = dict(cfg.DEFAULT_HYBRID_WEIGHTS), []
    if len(val):
        _log("tuning hybrid blend weights on validation NDCG@10 ...")
        weights, weight_table = E.tune_hybrid_weights(
            hybrid, E.build_histories(train), E.build_relevant(val, cfg.RELEVANCE_THRESHOLD))
        _log(f"best hybrid weights: {weights}")

    # ---- 4. Stage B: final evaluation on test -----------------------------------------
    _log("fitting on train+val and evaluating on test ...")
    pop, content, cf = fit_models(trainval, movies, tags, cf_params, seed)
    hybrid = HybridRecommender(movies, pop, content, cf, weights=weights)
    metrics: dict = {}
    if len(test):
        metrics["rating_prediction"] = E.evaluate_rating_prediction(cf, test)
        hist = E.build_histories(trainval)
        rel = E.build_relevant(test, cfg.RELEVANCE_THRESHOLD)
        n_users_tv = trainval["userId"].nunique()
        rankers = E.build_rankers(hybrid, seed, max(cfg.EVAL_KS))
        summary, per_user = E.evaluate_rankers(rankers, hist, rel, hybrid, pop.count_, n_users_tv)
        users = summary["_users"]
        # hybrid NDCG split by which routing branch each user would take
        modes = np.array([hybrid.route(hist[u].n)[0] for u in users])
        by_mode = {m: {"n_users": int((modes == m).sum()),
                       "hybrid_ndcg@10": float(per_user["hybrid"][modes == m].mean())}
                   for m in np.unique(modes)}
        metrics["ranking"] = {k: v for k, v in summary.items() if k != "_users"}
        metrics["hybrid_by_routing_mode"] = by_mode
        _log("simulating cold-start users ...")
        metrics["cold_start_simulation"] = E.cold_start_curve(
            hybrid, hist, rel, pop.count_, n_users_tv, seed)

    # ---- 5. Stage C: final fit on everything -> serve ---------------------------------------
    _log("fitting final models on all ratings ...")
    pop, content, cf = fit_models(ratings, movies, tags, cf_params, seed)
    hybrid = HybridRecommender(movies, pop, content, cf, weights=weights)

    meta = {
        "bundle_version": BUNDLE_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "dataset": dataset,
        "rating_scale": [cfg.RATING_MIN, cfg.RATING_MAX, cfg.RATING_STEP],
        "cf_params": cf_params,
        "hybrid_weights": weights,
        "cold_start_weights": hybrid.cold_weights,
        "min_ratings_for_cf": hybrid.min_ratings_cf,
        "n_users": int(ratings["userId"].nunique()),
        "n_movies": int(n_items),
        "n_ratings": int(len(ratings)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    histories = {"userId": ratings["userId"].to_numpy(), "iidx": ratings["iidx"].to_numpy(),
                 "rating": ratings["rating"].to_numpy(dtype=np.float32),
                 "timestamp": ratings["timestamp"].to_numpy()}
    bundle = {"movies": movies, "hybrid": hybrid, "ratings": histories, "meta": meta}
    _atomic_write(artifact_dir / cfg.BUNDLE_FILENAME,
                  lambda p: joblib.dump(bundle, p, compress=3))

    report = {
        "meta": meta,
        "data_cleaning": clean_report,
        "split": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test)),
                  "relevance_threshold": cfg.RELEVANCE_THRESHOLD},
        "cf_tuning_grid": cf_tuning,
        "hybrid_weight_tuning_top10": weight_table,
        **metrics,
        "training_seconds": round(time.time() - t0, 1),
    }
    report = _clean_for_json(report)
    _atomic_write(artifact_dir / cfg.METRICS_FILENAME,
                  lambda p: p.write_text(json.dumps(report, indent=2), encoding="utf-8"))
    _atomic_write(artifact_dir / "RESULTS.md",
                  lambda p: p.write_text(render_results_markdown(report), encoding="utf-8"))
    _log(f"done in {report['training_seconds']}s -> {artifact_dir}")
    return report


def render_results_markdown(report: dict) -> str:
    """Turn metrics.json into README-ready tables (numbers are copied, never typed)."""
    k = cfg.PRIMARY_K
    lines = []
    ds = report.get("meta", {}).get("dataset", {})
    if ds.get("synthetic"):
        lines += ["> **SYNTHETIC TEST DATA, not MovieLens. These numbers say nothing about "
                  "real-data quality. Do not quote them.**", ""]
    elif ds:
        sha = ds.get("files", {}).get("ratings.csv", {}).get("sha256", "?")[:12]
        lines += [f"_Dataset: `{ds.get('name')}` (ratings.csv sha256 {sha}...)_", ""]
    if "rating_prediction" in report:
        rp = report["rating_prediction"]
        lines += ["#### Rating prediction (test set; lower is better)", "",
                  "| Model | RMSE | MAE |", "|---|---|---|"]
        for name in ("global_mean", "item_bias", "user+item_bias", "matrix_factorization"):
            lines.append(f"| {name} | {rp[name]['rmse']:.4f} | {rp[name]['mae']:.4f} |")
        lines.append("")
    if "ranking" in report:
        rk = report["ranking"]
        lines += [f"#### Top-{k} ranking (test set; {rk['_n_users_evaluated']} users; "
                  f"relevant = held-out rating >= {report['split']['relevance_threshold']})", "",
                  f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | Coverage@{k} | Diversity@{k} | Novelty@{k} |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in rk.items():
            if name.startswith("_"):
                continue
            lines.append(f"| {name} | {r[f'precision@{k}']:.4f} | {r[f'recall@{k}']:.4f} | "
                         f"{r[f'ndcg@{k}']:.4f} | {r[f'coverage@{k}']:.4f} | "
                         f"{r[f'diversity@{k}']:.4f} | {r[f'novelty@{k}']:.2f} |")
        lines.append("")
    if "cold_start_simulation" in report:
        cs = report["cold_start_simulation"]
        names = [n for n in next(iter(cs.values())) if not n.startswith("_")]
        lines += [f"#### Cold-start simulation: NDCG@{k} when only the first *m* ratings are known", "",
                  "| m ratings | hybrid routing | " + " | ".join(names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for m, row in cs.items():
            lines.append(f"| {m} | {row['_mode_used_by_hybrid']} | " +
                         " | ".join(f"{row[n][f'ndcg@{k}']:.4f}" for n in names) + " |")
        lines.append("")
    if "meta" in report:
        m = report["meta"]
        lines += [f"_Trained {m['trained_at_utc']} (UTC), seed {m['seed']}, "
                  f"{m['n_ratings']:,} ratings, {m['n_users']:,} users, {m['n_movies']:,} movies. "
                  f"CF params: {m['cf_params']}. Hybrid weights: {m['hybrid_weights']}._"]
    return "\n".join(lines) + "\n"
