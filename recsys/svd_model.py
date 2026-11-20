"""Explicit truncated SVD model for the college project (separate from ALS)."""
from __future__ import annotations
import numpy as np
import pandas as pd

class SVDRecommender:
    """SVD on a user-mean-centred matrix; unknown ratings are represented as zero residual."""
    def __init__(self, n_factors: int = 3): self.n_factors = n_factors
    def fit(self, ratings: pd.DataFrame):
        self.users_ = np.sort(ratings.user_id.unique()); self.items_ = np.sort(ratings.movie_id.unique())
        self.user_index_ = {v:i for i,v in enumerate(self.users_)}; self.item_index_ = {v:i for i,v in enumerate(self.items_)}
        self.matrix_ = np.full((len(self.users_), len(self.items_)), np.nan)
        for r in ratings.itertuples(): self.matrix_[self.user_index_[r.user_id], self.item_index_[r.movie_id]] = float(r.rating)
        self.global_mean_ = float(np.nanmean(self.matrix_))
        self.user_means_ = np.nanmean(self.matrix_, axis=1)
        self.user_means_ = np.where(np.isfinite(self.user_means_), self.user_means_, self.global_mean_)
        residual = np.where(np.isfinite(self.matrix_), self.matrix_ - self.user_means_[:, None], 0.0)
        # Direct dense SVD is appropriate for this small college dataset and is genuine SVD.
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        rank = max(1, min(int(self.n_factors), len(s)))
        self.reconstruction_ = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        self.rank_ = rank
        return self
    def predict(self, user_id, movie_id):
        ui=self.user_index_.get(user_id); ii=self.item_index_.get(movie_id)
        if ui is None or ii is None: return self.global_mean_
        return float(np.clip(self.user_means_[ui] + self.reconstruction_[ui, ii], 1, 5))
    def predict_pairs(self, users, items): return np.array([self.predict(int(u), int(i)) for u,i in zip(users,items)])
    def score_items(self, user_id):
        ui=self.user_index_.get(user_id)
        if ui is None: return np.full(len(self.items_), self.global_mean_)
        return np.clip(self.user_means_[ui] + self.reconstruction_[ui], 1, 5)
