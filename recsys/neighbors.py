"""Simple user-based and item-based neighborhood collaborative filtering."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class NeighborhoodCF:
    def __init__(self, mode: str = "user", min_common: int = 1):
        if mode not in {"user", "item"}:
            raise ValueError("mode must be 'user' or 'item'")
        self.mode, self.min_common = mode, min_common

    def fit(self, ratings: pd.DataFrame):
        self.users_ = np.sort(ratings.user_id.unique())
        self.items_ = np.sort(ratings.movie_id.unique())
        self.user_index_ = {v:i for i,v in enumerate(self.users_)}
        self.item_index_ = {v:i for i,v in enumerate(self.items_)}
        self.matrix_ = np.full((len(self.users_), len(self.items_)), np.nan, dtype=float)
        for row in ratings.itertuples():
            self.matrix_[self.user_index_[row.user_id], self.item_index_[row.movie_id]] = float(row.rating)
        self.user_means_ = np.nanmean(self.matrix_, axis=1)
        self.global_mean_ = float(np.nanmean(self.matrix_))
        self.user_means_ = np.where(np.isfinite(self.user_means_), self.user_means_, self.global_mean_)
        # Center observed ratings; unobserved entries remain zero for similarity calculations.
        centered = np.where(np.isfinite(self.matrix_), self.matrix_ - self.user_means_[:, None], 0.0)
        if self.mode == "user":
            self.similarity_ = cosine_similarity(centered)
            np.fill_diagonal(self.similarity_, 0)
        else:
            self.similarity_ = cosine_similarity(centered.T)
            np.fill_diagonal(self.similarity_, 0)
        self.centered_ = centered
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        ui = self.user_index_.get(user_id); ii = self.item_index_.get(movie_id)
        if ui is None or ii is None: return self.global_mean_
        if self.mode == "user":
            raters = np.flatnonzero(np.isfinite(self.matrix_[:, ii]))
            raters = raters[raters != ui]
            if not len(raters): return self.user_means_[ui]
            sims = self.similarity_[ui, raters]
            order = np.argsort(np.abs(sims))[::-1]
            raters, sims = raters[order], sims[order]
            valid = np.abs(sims) > 1e-12
            raters, sims = raters[valid], sims[valid]
            if not len(sims): return self.user_means_[ui]
            residuals = self.matrix_[raters, ii] - self.user_means_[raters]
            pred = self.user_means_[ui] + float(np.dot(sims, residuals) / np.sum(np.abs(sims)))
        else:
            rated_items = np.flatnonzero(np.isfinite(self.matrix_[ui]))
            rated_items = rated_items[rated_items != ii]
            if not len(rated_items): return self.user_means_[ui]
            sims = self.similarity_[ii, rated_items]
            order = np.argsort(np.abs(sims))[::-1]
            rated_items, sims = rated_items[order], sims[order]
            valid = np.abs(sims) > 1e-12
            rated_items, sims = rated_items[valid], sims[valid]
            if not len(sims): return self.user_means_[ui]
            residuals = self.matrix_[ui, rated_items] - self.user_means_[ui]
            pred = self.user_means_[ui] + float(np.dot(sims, residuals) / np.sum(np.abs(sims)))
        return float(np.clip(pred, 1, 5))

    def predict_pairs(self, users, items):
        return np.array([self.predict(int(u), int(i)) for u, i in zip(users, items)], dtype=float)

    def score_items(self, user_id: int):
        return np.array([self.predict(user_id, int(item)) for item in self.items_])

class UserBasedCF(NeighborhoodCF):
    def __init__(self): super().__init__(mode="user")

class ItemBasedCF(NeighborhoodCF):
    def __init__(self): super().__init__(mode="item")
