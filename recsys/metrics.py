"""Evaluation metrics (all hand-written so they can be explained line by line).

Rating prediction (lower is better)
    RMSE = sqrt(mean((y - y_hat)^2))   punishes big mistakes more
    MAE  = mean(|y - y_hat|)           average star error

Top-K ranking (higher is better). A held-out movie is "relevant" if the user
rated it >= 4 stars. K is the length of the recommendation list.
    Precision@K = relevant items in top-K / K
    Recall@K    = relevant items in top-K / all relevant items
    NDCG@K      = DCG / ideal DCG, DCG = sum 1/log2(rank+1) over hits.
                  Rewards putting hits near the top of the list.

Beyond accuracy
    Coverage  = distinct movies ever recommended / catalog size
    Diversity = mean intra-list distance (1 - cosine) of content vectors
    Novelty   = mean -log2(popularity) of recommended movies (higher = less obvious)
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(d)))


def precision_at_k(recommended, relevant: set, k: int) -> float:
    hits = sum(1 for i in list(recommended)[:k] if i in relevant)
    return hits / k


def recall_at_k(recommended, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for i in list(recommended)[:k] if i in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 2) for rank, i in enumerate(list(recommended)[:k])
              if i in relevant)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return float(dcg / ideal)


def coverage(all_recommended_lists, n_items: int) -> float:
    seen = set()
    for lst in all_recommended_lists:
        seen.update(int(i) for i in lst)
    return len(seen) / n_items


def novelty(recommended, item_counts: np.ndarray, n_users: int) -> float:
    p = np.maximum(item_counts[np.asarray(recommended, dtype=int)], 1) / max(n_users, 1)
    return float(np.mean(-np.log2(np.minimum(p, 1.0))))
