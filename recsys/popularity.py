"""Popularity baseline: Bayesian weighted rating (the IMDb "Top 250" formula).

    score = v/(v+m) * R + m/(v+m) * C

    R = movie's mean rating      v = number of ratings
    C = global mean rating       m = minimum votes (a quantile of rating counts)

Why: a raw mean rewards a movie with one 5-star rating. Shrinking toward the
global mean C when v is small gives a fair, stable ranking. Items with no
ratings score exactly C, which makes this a safe fallback for new movies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityRecommender:
    def __init__(self, min_votes_quantile: float = 0.6):
        self.min_votes_quantile = min_votes_quantile

    def fit(self, ratings: pd.DataFrame, n_items: int) -> "PopularityRecommender":
        """ratings needs columns ``iidx`` and ``rating`` (training data only)."""
        iidx = ratings["iidx"].to_numpy()
        r = ratings["rating"].to_numpy(dtype=float)
        self.count_ = np.bincount(iidx, minlength=n_items).astype(np.int64)
        sums = np.bincount(iidx, weights=r, minlength=n_items)
        self.mean_ = np.divide(sums, self.count_, out=np.zeros(n_items),
                               where=self.count_ > 0)
        self.global_mean_ = float(r.mean())
        rated = self.count_[self.count_ > 0]
        self.min_votes_ = max(1.0, float(np.quantile(rated, self.min_votes_quantile)))
        v, m = self.count_.astype(float), self.min_votes_
        self.score_ = (v / (v + m)) * self.mean_ + (m / (v + m)) * self.global_mean_
        return self
