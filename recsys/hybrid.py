"""Hybrid recommender: blends popularity, content-based and collaborative scores.

How the blend works
-------------------
The three models score on different scales (predicted stars, cosine, weighted
rating), so each model's scores are converted to *percentile ranks* (0..1) over
the candidate movies and then combined with a weighted average:

    final(i) = sum_m w_m * rank_m(i)  /  sum of w_m over models that scored i

Dividing by the weights of the models that actually scored the movie is what
makes item cold-start work: a brand-new movie has no CF score, so it is judged
by content + popularity only instead of being dropped.

Cold-start routing (user side)
------------------------------
    0 ratings        -> popularity only (optionally filtered by chosen genres)
    1..4 ratings     -> content + popularity (CF needs more signal)
    >=5 ratings      -> full hybrid with weights tuned on a validation split
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config as cfg
from recsys.collaborative import MatrixFactorizationCF
from recsys.content import ContentModel
from recsys.popularity import PopularityRecommender

REASON_TEXT = {
    "cf": "Viewers with similar taste rated this highly",
    "content": "Matches the genres and tags you like",
    "pop": "Highly rated in this dataset",
}


@dataclass
class UserProfile:
    """A user's known ratings, as catalog item indices (``iidx``) + stars."""
    item_idx: np.ndarray
    ratings: np.ndarray
    # Extra movies to hide from the results without using them as taste signal
    # (used by the cold-start simulation, where the user "hasn't told us" about them).
    exclude: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.item_idx)


@dataclass
class RecResult:
    item_idx: np.ndarray
    scores: np.ndarray
    mode: str
    weights: dict
    reasons: list = field(default_factory=list)       # "cf" | "content" | "pop" per item
    predicted: np.ndarray | None = None                # CF predicted stars (NaN if n/a)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map values to percentile ranks in [0, 1]; NaN stays NaN."""
    out = np.full(x.shape, np.nan)
    valid = ~np.isnan(x)
    n = int(valid.sum())
    if n == 0:
        return out
    if n == 1:
        out[valid] = 1.0
        return out
    out[valid] = (rankdata(x[valid], method="average") - 1.0) / (n - 1.0)
    return out


def top_n_indices(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n largest scores, best first (O(N) partition + small sort)."""
    n = min(n, scores.size)
    if n <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(-scores, n - 1)[:n]
    return part[np.argsort(-scores[part], kind="stable")]


def centered_weights(ratings: np.ndarray) -> np.ndarray:
    """Mean-centred ratings; if a user rates everything the same, treat all as positive."""
    w = ratings - ratings.mean()
    return w if np.abs(w).max() > 1e-9 else np.ones_like(ratings)


class HybridRecommender:
    def __init__(self, movies: pd.DataFrame, pop: PopularityRecommender,
                 content: ContentModel, cf: MatrixFactorizationCF,
                 weights: dict | None = None, cold_weights: dict | None = None,
                 min_ratings_cf: int = cfg.MIN_RATINGS_FOR_CF,
                 cf_min_item_ratings: int = cfg.CF_MIN_ITEM_RATINGS):
        self.pop, self.content, self.cf = pop, content, cf
        self.weights = dict(weights or cfg.DEFAULT_HYBRID_WEIGHTS)
        self.cold_weights = dict(cold_weights or cfg.COLD_START_WEIGHTS)
        self.min_ratings_cf = min_ratings_cf
        self.cf_min_item_ratings = cf_min_item_ratings
        self.n_items = len(movies)
        self.genres = sorted({g for gl in movies["genre_list"] for g in gl})
        self._genre_masks = {
            g: np.array([g in gl for gl in movies["genre_list"]]) for g in self.genres}

    # ---- routing ------------------------------------------------------------
    def route(self, n_ratings: int) -> tuple[str, dict]:
        if n_ratings == 0:
            return "popular", {"pop": 1.0}
        if n_ratings < self.min_ratings_cf:
            return "cold_start", self.cold_weights
        return "hybrid", self.weights

    def genre_mask(self, genres: list[str] | None) -> np.ndarray | None:
        """Boolean mask of movies having ANY of the genres (None = no filter)."""
        if not genres:
            return None
        mask = np.zeros(self.n_items, dtype=bool)
        for g in genres:
            mask |= self._genre_masks[g]
        return mask

    # ---- scoring ------------------------------------------------------------
    def components(self, profile: UserProfile, cand: np.ndarray, names) -> tuple[dict, np.ndarray | None]:
        """Rank-normalised score arrays per model (None/missing = no signal)."""
        norm: dict = {}
        cf_raw = None
        if "cf" in names and profile.n > 0:
            b_u, p_u = self.cf.fold_in(profile.item_idx, profile.ratings)
            cf_raw = self.cf.score_items(b_u, p_u, self.cf_min_item_ratings)
            norm["cf"] = rank_normalize(np.where(cand, cf_raw, np.nan))
        if "content" in names and profile.n > 0:
            raw = self.content.profile_scores(profile.item_idx, centered_weights(profile.ratings))
            norm["content"] = None if raw is None else rank_normalize(np.where(cand, raw, np.nan))
        if "pop" in names:
            norm["pop"] = rank_normalize(np.where(cand, self.pop.score_, np.nan))
        return norm, cf_raw

    @staticmethod
    def blend(norm: dict, weights: dict, n_items: int) -> np.ndarray:
        """Weighted average of rank scores; -inf where no model produced a score."""
        num = np.zeros(n_items)
        den = np.zeros(n_items)
        for name, w in weights.items():
            arr = norm.get(name)
            if w <= 0 or arr is None:
                continue
            ok = ~np.isnan(arr)
            num[ok] += w * arr[ok]
            den[ok] += w
        out = np.full(n_items, -np.inf)
        np.divide(num, den, out=out, where=den > 0)
        return out

    def _candidates(self, profile: UserProfile, genres, extra_exclude) -> np.ndarray:
        cand = np.ones(self.n_items, dtype=bool)
        cand[profile.item_idx] = False                 # never re-recommend what they rated
        if profile.exclude is not None and len(profile.exclude):
            cand[profile.exclude] = False
        if extra_exclude is not None and len(extra_exclude):
            cand[np.asarray(extra_exclude, dtype=int)] = False
        gm = self.genre_mask(genres)
        if gm is not None:
            cand &= gm
        return cand

    def recommend(self, profile: UserProfile, n: int = 10, genres: list[str] | None = None,
                  weights: dict | None = None, extra_exclude=None) -> RecResult:
        """Top-n personalised recommendations. ``weights`` overrides routing
        (used to evaluate single models through the exact same code path)."""
        mode, w = self.route(profile.n) if weights is None else ("custom", weights)
        cand = self._candidates(profile, genres, extra_exclude)
        if not cand.any():
            return RecResult(np.array([], dtype=int), np.array([]), mode, w, [], np.array([]))

        names = [k for k, v in w.items() if v > 0]
        norm, cf_raw = self.components(profile, cand, names)
        scores = self.blend(norm, w, self.n_items)
        scores[~cand] = -np.inf
        if not np.isfinite(scores[cand]).any():        # no model had a signal -> popularity
            scores = np.where(cand, rank_normalize(np.where(cand, self.pop.score_, np.nan)), -np.inf)
            norm = {"pop": rank_normalize(np.where(cand, self.pop.score_, np.nan))}
            w = {"pop": 1.0}

        top = top_n_indices(scores, n)
        top = top[np.isfinite(scores[top])]
        reasons = []
        for i in top:
            contrib = {m: w.get(m, 0) * norm[m][i] for m in norm
                       if norm[m] is not None and not np.isnan(norm[m][i])}
            reasons.append(max(contrib, key=contrib.get) if contrib else "pop")
        predicted = None
        if cf_raw is not None:
            predicted = np.clip(cf_raw[top], cfg.RATING_MIN, cfg.RATING_MAX)
        return RecResult(top, scores[top], mode, dict(w), reasons, predicted)

    def similar_items(self, item_idx: int, n: int = 10, genres: list[str] | None = None) -> RecResult:
        """Movies similar to one movie: blend of content cosine and CF latent cosine."""
        cand = np.ones(self.n_items, dtype=bool)
        cand[item_idx] = False
        gm = self.genre_mask(genres)
        if gm is not None:
            cand &= gm

        norm: dict = {}
        content = self.content.similar_scores(item_idx)
        if np.any(content != 0):
            norm["content"] = rank_normalize(np.where(cand, content, np.nan))
        cf_sims = self.cf.item_similarity(item_idx, self.cf_min_item_ratings)
        if cf_sims is not None:
            norm["cf"] = rank_normalize(np.where(cand, cf_sims, np.nan))

        w = {k: v for k, v in cfg.SIMILAR_WEIGHTS.items() if k in norm}
        if not w:                                     # nothing to compare -> popularity
            norm["pop"] = rank_normalize(np.where(cand, self.pop.score_, np.nan))
            w = {"pop": 1.0}
        scores = self.blend(norm, w, self.n_items)
        scores[~cand] = -np.inf
        top = top_n_indices(scores, n)
        top = top[np.isfinite(scores[top])]
        reasons = []
        for i in top:
            contrib = {m: w[m] * norm[m][i] for m in w if not np.isnan(norm[m][i])}
            reasons.append(max(contrib, key=contrib.get) if contrib else "pop")
        return RecResult(top, scores[top], "similar", w, reasons, None)

    def popular(self, n: int = 10, genres: list[str] | None = None) -> RecResult:
        """Top movies by Bayesian weighted rating (cold-start / browse view)."""
        empty = UserProfile(np.array([], dtype=int), np.array([]))
        return self.recommend(empty, n=n, genres=genres)
