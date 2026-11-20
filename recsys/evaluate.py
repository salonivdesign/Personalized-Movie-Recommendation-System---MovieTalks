"""Offline evaluation + validation-based tuning.

Leakage rules enforced here
* Models are fit on TRAIN only while tuning on VALIDATION.
* Final test numbers come from models fit on TRAIN+VAL and scored on TEST.
* A user's candidate list never contains movies already in their visible history.
* Hybrid weights and CF hyperparameters are chosen on validation, never on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from recsys import metrics as M
from recsys.collaborative import MatrixFactorizationCF
from recsys.hybrid import HybridRecommender, UserProfile, top_n_indices


# ---- helpers ---------------------------------------------------------------
def build_histories(df: pd.DataFrame) -> dict[int, UserProfile]:
    """userId -> UserProfile with ratings in chronological order."""
    out = {}
    for uid, g in df.groupby("userId", sort=True):
        g = g.sort_values(["timestamp", "movieId"])
        out[int(uid)] = UserProfile(g["iidx"].to_numpy(), g["rating"].to_numpy(dtype=float))
    return out


def build_relevant(df: pd.DataFrame, threshold: float) -> dict[int, set]:
    """userId -> set of held-out item indices rated >= threshold."""
    hot = df[df["rating"] >= threshold]
    return {int(uid): set(g["iidx"].tolist()) for uid, g in hot.groupby("userId", sort=True)}


# ---- rating prediction -------------------------------------------------------
def evaluate_rating_prediction(cf: MatrixFactorizationCF, test: pd.DataFrame) -> dict:
    uid, iidx, y = test["userId"].to_numpy(), test["iidx"].to_numpy(), test["rating"].to_numpy()
    out = {}
    for name, level in [("global_mean", "global"), ("item_bias", "item"),
                        ("user+item_bias", "bias"), ("matrix_factorization", "full")]:
        pred = cf.predict_pairs(uid, iidx, level)
        out[name] = {"rmse": M.rmse(y, pred), "mae": M.mae(y, pred)}
    out["n_test_ratings"] = int(len(test))
    out["share_test_ratings_on_unseen_movies"] = float(np.mean(cf.item_count_[iidx] == 0))
    return out


def tune_cf(train: pd.DataFrame, val: pd.DataFrame, n_items: int, base_params: dict,
            grid: dict, seed: int) -> tuple[dict, list]:
    """Grid-search CF hyperparameters by validation RMSE."""
    rows, best = [], None
    for k in grid["n_factors"]:
        for reg in grid["reg"]:
            params = {**base_params, "n_factors": k, "reg": reg}
            model = MatrixFactorizationCF(seed=seed, **params).fit(train, n_items)
            pred = model.predict_pairs(val["userId"].to_numpy(), val["iidx"].to_numpy())
            score = M.rmse(val["rating"].to_numpy(), pred)
            rows.append({"n_factors": k, "reg": reg, "val_rmse": score})
            if best is None or score < best[0]:
                best = (score, params)
    return best[1], rows


# ---- ranking -------------------------------------------------------------------
def build_rankers(hybrid: HybridRecommender, seed: int, kmax: int) -> dict:
    """name -> function(UserProfile) -> array of top item indices."""
    pop, n_items = hybrid.pop, hybrid.n_items
    rng = np.random.default_rng(seed)

    def hidden(p: UserProfile) -> np.ndarray:
        mask = np.zeros(n_items, dtype=bool)
        mask[p.item_idx] = True
        if p.exclude is not None:
            mask[p.exclude] = True
        return mask

    def by_scores(raw: np.ndarray):
        def rank(p: UserProfile):
            s = raw.astype(float).copy()
            s[hidden(p)] = -np.inf
            return top_n_indices(s, kmax)
        return rank

    def random_rank(p: UserProfile):
        perm = rng.permutation(n_items)
        return perm[~hidden(p)[perm]][:kmax]

    return {
        "random": random_rank,
        "popularity (weighted rating)": by_scores(pop.score_),
        "popularity (rating count)": by_scores(pop.count_),
        "content-based": lambda p: hybrid.recommend(p, kmax, weights={"content": 1.0}).item_idx,
        "collaborative (MF)": lambda p: hybrid.recommend(p, kmax, weights={"cf": 1.0}).item_idx,
        "hybrid": lambda p: hybrid.recommend(p, kmax).item_idx,
    }


def evaluate_rankers(rankers: dict, histories: dict, relevant: dict, hybrid: HybridRecommender,
                     item_counts: np.ndarray, n_users_train: int,
                     ks=cfg.EVAL_KS, primary_k: int = cfg.PRIMARY_K):
    """Evaluate each ranker over users that have >= 1 relevant held-out item.

    Returns (summary, per_user_ndcg) where per_user_ndcg[name] is an array in the
    order of ``users``; ``users`` is returned inside summary['_users'].
    """
    kmax = max(ks)
    users = [u for u in sorted(relevant) if u in histories and relevant[u]]
    summary, per_user = {}, {}
    for name, fn in rankers.items():
        acc = {f"{m}@{k}": [] for m in ("precision", "recall", "ndcg") for k in ks}
        lists, ild, nov = [], [], []
        for u in users:
            rec = np.asarray(fn(histories[u]))[:kmax]
            rel = relevant[u]
            for k in ks:
                acc[f"precision@{k}"].append(M.precision_at_k(rec, rel, k))
                acc[f"recall@{k}"].append(M.recall_at_k(rec, rel, k))
                acc[f"ndcg@{k}"].append(M.ndcg_at_k(rec, rel, k))
            top = rec[:primary_k]
            lists.append(top)
            if len(top):
                ild.append(hybrid.content.diversity(top))
                nov.append(M.novelty(top, item_counts, n_users_train))
        res = {key: float(np.mean(v)) for key, v in acc.items()}
        res[f"coverage@{primary_k}"] = M.coverage(lists, hybrid.n_items)
        res[f"diversity@{primary_k}"] = float(np.mean(ild)) if ild else 0.0
        res[f"novelty@{primary_k}"] = float(np.mean(nov)) if nov else 0.0
        summary[name] = res
        per_user[name] = np.array(acc[f"ndcg@{primary_k}"])
    summary["_n_users_evaluated"] = len(users)
    summary["_users"] = users
    return summary, per_user


# ---- hybrid weight tuning --------------------------------------------------------
def weight_grid(step: float) -> list[dict]:
    n = int(round(1 / step))
    return [{"cf": a / n, "content": b / n, "pop": (n - a - b) / n}
            for a in range(n + 1) for b in range(n + 1 - a)]


def tune_hybrid_weights(hybrid: HybridRecommender, histories: dict, relevant: dict,
                        step: float = cfg.HYBRID_WEIGHT_STEP, k: int = cfg.PRIMARY_K):
    """Pick blend weights maximising mean NDCG@k on the validation split.

    Component rank-scores are computed once per user; each weight combo is then
    just a cheap weighted average, so the whole grid is fast.
    """
    users = [u for u in sorted(relevant) if relevant[u] and u in histories
             and histories[u].n >= hybrid.min_ratings_cf]
    if not users:
        return dict(hybrid.weights), []
    grid = weight_grid(step)
    totals = np.zeros(len(grid))
    for u in users:
        prof = histories[u]
        cand = np.ones(hybrid.n_items, dtype=bool)
        cand[prof.item_idx] = False
        norm, _ = hybrid.components(prof, cand, ["cf", "content", "pop"])
        for gi, w in enumerate(grid):
            s = hybrid.blend(norm, w, hybrid.n_items)
            s[~cand] = -np.inf
            totals[gi] += M.ndcg_at_k(top_n_indices(s, k), relevant[u], k)
    totals /= len(users)
    order = np.argsort(-totals, kind="stable")
    table = [{**grid[i], f"val_ndcg@{k}": float(totals[i])} for i in order[:10]]
    return grid[int(order[0])], table + [{"n_users_used": len(users)}]


# ---- cold-start simulation ------------------------------------------------------------
def cold_start_curve(hybrid: HybridRecommender, histories: dict, relevant: dict,
                     item_counts: np.ndarray, n_users_train: int, seed: int,
                     sizes=cfg.COLD_START_HISTORY_SIZES, k: int = cfg.PRIMARY_K) -> dict:
    """Pretend each test user is new: reveal only their first m training ratings.

    Everything else they rated stays hidden from the candidate list (``exclude``)
    so results are comparable across m.
    """
    rankers_all = build_rankers(hybrid, seed, k)
    keep = {n: rankers_all[n] for n in
            ("popularity (weighted rating)", "content-based", "collaborative (MF)", "hybrid")}
    out = {}
    for m in sizes:
        trunc = {u: UserProfile(h.item_idx[:m], h.ratings[:m], exclude=h.item_idx[m:])
                 for u, h in histories.items()}
        summary, _ = evaluate_rankers(keep, trunc, relevant, hybrid, item_counts,
                                      n_users_train, ks=(k,), primary_k=k)
        out[str(m)] = {name: {f"ndcg@{k}": s[f"ndcg@{k}"], f"recall@{k}": s[f"recall@{k}"]}
                       for name, s in summary.items() if not name.startswith("_")}
        out[str(m)]["_mode_used_by_hybrid"] = hybrid.route(m)[0]
    return out
