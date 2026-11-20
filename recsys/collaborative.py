"""Collaborative filtering: biased matrix factorization trained with ALS.

Prediction for user u and movie i:

    r_hat(u, i) = mu + b_i + b_u + p_u . q_i

    mu   global mean rating
    b_i  item bias (how much better/worse than average the movie is)
    b_u  user bias (how generous the user is)
    p_u, q_i  k-dimensional latent "taste" and "movie" vectors

Training steps
1. Compute damped biases (shrunk toward 0 when a user/item has few ratings).
2. Factorise the *residual* ratings  r - mu - b_i - b_u  with Alternating Least
   Squares: fix Q and solve a small ridge regression for every user's p_u, then
   fix P and solve for every q_i. Repeat. Each sub-problem has a closed form.

Why ALS instead of sklearn's TruncatedSVD: plain SVD treats the ~98% missing
ratings as zeros, which biases the model. ALS only fits the *observed* ratings,
which is what Funk-SVD / "SVD" in the Netflix Prize actually does.

New users (e.g. someone rating movies in the web UI) are handled by ``fold_in``:
keep Q fixed and solve the same ridge problem for just that user. No retraining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import config as cfg


def _als_step(R: sparse.csr_matrix, fixed: np.ndarray, reg: float) -> np.ndarray:
    """Solve for every row factor given the fixed opposite-side factors.

    Uses "weighted-lambda" regularisation (reg * number_of_ratings) so heavy
    users/items and light ones are regularised comparably.
    """
    n_rows, k = R.shape[0], fixed.shape[1]
    out = np.zeros((n_rows, k))
    eye = np.eye(k)
    indptr, indices, data = R.indptr, R.indices, R.data
    for row in range(n_rows):
        s, e = indptr[row], indptr[row + 1]
        if s == e:
            continue  # no ratings -> factor stays 0 (prediction falls back to biases)
        y = fixed[indices[s:e]]
        a = y.T @ y + reg * (e - s) * eye
        out[row] = np.linalg.solve(a, y.T @ data[s:e])
    return out


class MatrixFactorizationCF:
    def __init__(self, n_factors: int = 32, reg: float = 0.1, n_iters: int = 12,
                 user_bias_reg: float = 15.0, item_bias_reg: float = 10.0, seed: int = 42):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iters = n_iters
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg
        self.seed = seed

    # ---- training ----------------------------------------------------------
    def fit(self, ratings: pd.DataFrame, n_items: int) -> "MatrixFactorizationCF":
        """ratings needs columns userId, iidx, rating (training data only)."""
        self.n_items = n_items
        self.user_ids_ = np.sort(ratings["userId"].unique())
        n_users = len(self.user_ids_)
        uidx = np.searchsorted(self.user_ids_, ratings["userId"].to_numpy())
        iidx = ratings["iidx"].to_numpy()
        r = ratings["rating"].to_numpy(dtype=float)

        self.mu_ = float(r.mean())
        self.item_count_ = np.bincount(iidx, minlength=n_items).astype(np.int64)
        user_count = np.bincount(uidx, minlength=n_users)

        # damped biases: sum(residual) / (n + lambda)
        b_i = np.bincount(iidx, weights=r - self.mu_, minlength=n_items) / (
            self.item_count_ + self.item_bias_reg)
        resid = r - self.mu_ - b_i[iidx]
        b_u = np.bincount(uidx, weights=resid, minlength=n_users) / (
            user_count + self.user_bias_reg)
        resid = resid - b_u[uidx]

        r_user = sparse.csr_matrix((resid, (uidx, iidx)), shape=(n_users, n_items))
        r_item = r_user.T.tocsr()

        rng = np.random.default_rng(self.seed)
        p = rng.normal(0, 0.1, (n_users, self.n_factors))
        q = rng.normal(0, 0.1, (n_items, self.n_factors))
        q[self.item_count_ == 0] = 0.0
        for _ in range(self.n_iters):
            p = _als_step(r_user, q, self.reg)
            q = _als_step(r_item, p, self.reg)

        self.b_i_ = b_i.astype(np.float32)
        self.b_u_ = b_u.astype(np.float32)
        self.P_ = p.astype(np.float32)
        self.Q_ = q.astype(np.float32)
        norms = np.linalg.norm(self.Q_, axis=1, keepdims=True)
        self.Qn_ = self.Q_ / np.maximum(norms, 1e-12)
        return self

    # ---- rating prediction (used for RMSE / MAE) ----------------------------
    def predict_pairs(self, user_ids: np.ndarray, item_idx: np.ndarray,
                      level: str = "full") -> np.ndarray:
        """Vectorised predictions. ``level`` selects an ablation:
        'global' (mu), 'item' (mu+b_i), 'bias' (mu+b_i+b_u), 'full' (+ p_u.q_i).
        """
        user_ids = np.asarray(user_ids)
        item_idx = np.asarray(item_idx)
        pred = np.full(len(item_idx), self.mu_, dtype=float)
        if level == "global":
            return pred
        pred += self.b_i_[item_idx]
        if level == "item":
            return np.clip(pred, cfg.RATING_MIN, cfg.RATING_MAX)
        pos = np.clip(np.searchsorted(self.user_ids_, user_ids), 0, len(self.user_ids_) - 1)
        known = self.user_ids_[pos] == user_ids
        pred += np.where(known, self.b_u_[pos], 0.0)
        if level == "full":
            dot = np.einsum("ij,ij->i", self.P_[pos], self.Q_[item_idx])
            pred += np.where(known, dot, 0.0)
        return np.clip(pred, cfg.RATING_MIN, cfg.RATING_MAX)

    # ---- serving: new users --------------------------------------------------
    def fold_in(self, item_idx: np.ndarray, ratings: np.ndarray) -> tuple[float, np.ndarray]:
        """Estimate (b_u, p_u) for a user from their ratings, keeping Q fixed."""
        item_idx = np.asarray(item_idx)
        r = np.asarray(ratings, dtype=float)
        resid = r - self.mu_ - self.b_i_[item_idx]
        b_u = float(resid.sum() / (len(r) + self.user_bias_reg))
        resid = resid - b_u
        y = self.Q_[item_idx].astype(float)
        a = y.T @ y + self.reg * len(r) * np.eye(self.n_factors)
        p_u = np.linalg.solve(a, y.T @ resid)
        return b_u, p_u

    def score_items(self, b_u: float, p_u: np.ndarray, min_item_ratings: int = 0) -> np.ndarray:
        """Predicted rating for every movie. NaN where the movie has too few
        training ratings for CF to be trusted (the hybrid then falls back)."""
        scores = self.mu_ + self.b_i_ + b_u + self.Q_ @ p_u.astype(np.float32)
        scores = scores.astype(float)
        scores[self.item_count_ < max(1, min_item_ratings)] = np.nan
        return scores

    def item_similarity(self, item_idx: int, min_item_ratings: int = 0) -> np.ndarray | None:
        """Cosine similarity between latent item vectors (NaN where unsupported).
        Returns None if the query movie itself has no CF signal."""
        if self.item_count_[item_idx] < max(1, min_item_ratings):
            return None
        sims = (self.Qn_ @ self.Qn_[item_idx]).astype(float)
        sims[self.item_count_ < max(1, min_item_ratings)] = np.nan
        return sims
