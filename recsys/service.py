"""Serving layer: loads saved artifacts and answers questions in plain Python.

This module is deliberately free of web-framework code. FastAPI (api/main.py)
only translates HTTP <-> these method calls, so the ML side can be tested and
reused (CLI, notebook, another framework) without a server.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

import config as cfg
from recsys.hybrid import REASON_TEXT, RecResult, UserProfile

log = logging.getLogger("recsys.service")

REQUIRED_BUNDLE_KEYS = ("movies", "hybrid", "ratings", "meta")
MOVIE_SORTS = ("popular", "top_rated", "year", "title")

MODE_TEXT = {
    "popular": "Popular picks (no ratings yet). Rate a few movies to personalise.",
    "cold_start": "Content + popularity (rate a few more movies to unlock collaborative filtering).",
    "hybrid": "Hybrid: collaborative + content + popularity.",
    "similar": "Similar movies: content similarity blended with collaborative similarity.",
}


class ServiceError(Exception):
    status_code = 400


class NotFoundError(ServiceError):
    status_code = 404


class InvalidInputError(ServiceError):
    status_code = 422


class ArtifactError(RuntimeError):
    """Saved model files are unreadable, incomplete, or incompatible with config.py.

    Messages are safe to show to API clients (no file paths or internal details);
    the original exception is chained and logged by the caller.
    """


class RecommenderService:
    def __init__(self, bundle: dict, metrics: dict | None = None):
        self.movies = bundle["movies"]
        self.hybrid = bundle["hybrid"]
        self.meta = bundle["meta"]
        self.metrics = metrics
        self.n_items = len(self.movies)
        self.movie_ids = self.movies["movieId"].to_numpy()
        self._id_to_idx = {int(m): i for i, m in enumerate(self.movie_ids)}
        self._title_lc = np.array([t.lower() for t in self.movies["display_title"]])
        self._years = self.movies["year"].to_numpy()
        self._imdb = self.movies["imdbId"].to_numpy()
        self._genre_lists = list(self.movies["genre_list"])
        pop = self.hybrid.pop
        self._counts, self._means = pop.count_, pop.mean_

        # per-user rating history for the "demo user" feature
        r = bundle["ratings"]
        order = np.lexsort((r["timestamp"], r["userId"]))
        uid, iidx, rating = r["userId"][order], r["iidx"][order], r["rating"][order].astype(float)
        bounds = np.flatnonzero(np.diff(uid)) + 1
        self._user_hist = {
            int(u[0]): (i, rt) for u, i, rt in
            zip(np.split(uid, bounds), np.split(iidx, bounds), np.split(rating, bounds))}

    @classmethod
    def load(cls, artifact_dir: Path | str = cfg.ARTIFACT_DIR) -> "RecommenderService":
        artifact_dir = Path(artifact_dir)
        bundle_path = artifact_dir / cfg.BUNDLE_FILENAME
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"{bundle_path} not found. Train first: python scripts/train.py")
        retrain = "Retrain with: python scripts/train.py"
        try:
            bundle = joblib.load(bundle_path)
        except Exception as exc:
            raise ArtifactError(f"{cfg.BUNDLE_FILENAME} could not be read (corrupt, or built with "
                                f"different library versions). {retrain}") from exc
        if not isinstance(bundle, dict) or any(k not in bundle for k in REQUIRED_BUNDLE_KEYS):
            raise ArtifactError(f"{cfg.BUNDLE_FILENAME} is incomplete. {retrain}")

        # The API validates ratings against config.py, so the model must agree with it.
        trained_scale = bundle["meta"].get("rating_scale")
        current_scale = [cfg.RATING_MIN, cfg.RATING_MAX, cfg.RATING_STEP]
        if trained_scale is not None and [float(x) for x in trained_scale] != current_scale:
            raise ArtifactError(f"The model was trained with rating scale {list(trained_scale)} but "
                                f"config.py says {current_scale}. {retrain}")

        metrics = None
        metrics_path = artifact_dir / cfg.METRICS_FILENAME
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("%s is unreadable; /api/metrics will return 404", cfg.METRICS_FILENAME)
        try:
            return cls(bundle, metrics)
        except Exception as exc:
            raise ArtifactError(f"{cfg.BUNDLE_FILENAME} has an unexpected structure. {retrain}") from exc

    # ---- serialisation ------------------------------------------------------------
    def _movie(self, i: int) -> dict:
        year = None if np.isnan(self._years[i]) else int(self._years[i])
        imdb = self._imdb[i]
        n = int(self._counts[i])
        return {
            "movie_id": int(self.movie_ids[i]),
            "title": str(self.movies["display_title"].iloc[i]),
            "year": year,
            "genres": list(self._genre_lists[i]),
            "avg_rating": round(float(self._means[i]), 2) if n else None,
            "n_ratings": n,
            "imdb_url": None if np.isnan(imdb) else f"https://www.imdb.com/title/tt{int(imdb):07d}/",
        }

    def _items(self, res: RecResult, profile: UserProfile | None = None) -> list[dict]:
        liked = None
        if profile is not None and profile.n:
            liked = profile.item_idx[profile.ratings >= cfg.RELEVANCE_THRESHOLD]
        because = [None] * len(res.item_idx)
        if liked is not None and len(liked) and len(res.item_idx):
            m = self.hybrid.content.matrix_
            sims = np.asarray((m[res.item_idx] @ m[liked].T).todense())
            best = sims.argmax(axis=1)
            for j, reason in enumerate(res.reasons):
                if reason == "content" and sims[j, best[j]] > 0:
                    because[j] = str(self.movies["display_title"].iloc[liked[best[j]]])
        out = []
        for j, i in enumerate(res.item_idx):
            item = self._movie(int(i))
            item["reason_code"] = res.reasons[j]
            item["reason"] = REASON_TEXT[res.reasons[j]]
            item["because"] = because[j]
            pr = None
            if res.predicted is not None and not np.isnan(res.predicted[j]):
                pr = round(float(res.predicted[j]), 2)
            item["predicted_rating"] = pr
            item["score"] = round(float(res.scores[j]), 4)
            out.append(item)
        return out

    # ---- helpers --------------------------------------------------------------------
    def _idx(self, movie_id: int) -> int:
        try:
            return self._id_to_idx[int(movie_id)]
        except KeyError:
            raise NotFoundError(f"Movie {movie_id} not found") from None

    def _check_genres(self, genres: list[str] | None) -> list[str]:
        genres = genres or []
        bad = [g for g in genres if g not in self.hybrid.genres]
        if bad:
            raise InvalidInputError(f"Unknown genres: {bad}. Valid: {self.hybrid.genres}")
        return genres

    # ---- public API -----------------------------------------------------------------
    def health(self) -> dict:
        dataset = self.meta.get("dataset") or {}
        return {"status": "ok", "n_movies": self.n_items, "n_users": self.meta["n_users"],
                "n_ratings": self.meta["n_ratings"], "trained_at_utc": self.meta["trained_at_utc"],
                "dataset_name": dataset.get("name"), "synthetic_data": dataset.get("synthetic"),
                "rating_scale": {"min": cfg.RATING_MIN, "max": cfg.RATING_MAX,
                                 "step": cfg.RATING_STEP},
                "relevance_threshold": cfg.RELEVANCE_THRESHOLD}

    def genres(self) -> list[str]:
        return list(self.hybrid.genres)

    def get_movie(self, movie_id: int) -> dict:
        return self._movie(self._idx(movie_id))

    def list_movies(self, limit: int = 20, offset: int = 0, genre: str | None = None,
                    sort: str = "popular", min_ratings: int = 0) -> dict:
        """Paginated catalog listing. Sorting is stable: ties are broken by catalog order."""
        if sort not in MOVIE_SORTS:
            raise InvalidInputError(f"Unknown sort '{sort}'. Valid: {list(MOVIE_SORTS)}")
        mask = self._counts >= max(0, min_ratings)
        if genre:
            self._check_genres([genre])
            mask = mask & self.hybrid.genre_mask([genre])
        idx = np.flatnonzero(mask)                       # ascending catalog order
        if sort == "popular":                            # most ratings first
            idx = idx[np.lexsort((idx, -self._counts[idx]))]
        elif sort == "top_rated":                        # Bayesian weighted rating, best first
            idx = idx[np.lexsort((idx, -self.hybrid.pop.score_[idx]))]
        elif sort == "year":                             # newest first, unknown year last
            idx = idx[np.lexsort((idx, -np.nan_to_num(self._years[idx], nan=-1.0)))]
        else:                                            # title A-Z (stable)
            idx = idx[np.argsort(self._title_lc[idx], kind="stable")]
        page = idx[offset:offset + limit]
        return {"total": int(len(idx)), "limit": limit, "offset": offset, "sort": sort,
                "items": [self._movie(int(i)) for i in page]}

    def search(self, query: str, limit: int = 12, genre: str | None = None) -> list[dict]:
        q = query.strip().lower()
        if not q:
            raise InvalidInputError("Search query must not be empty")
        mask = np.ones(self.n_items, dtype=bool)
        for token in q.split():
            mask &= np.char.find(self._title_lc, token) >= 0
        if genre:
            self._check_genres([genre])
            mask &= self.hybrid.genre_mask([genre])
        idx = np.flatnonzero(mask)
        starts = np.char.startswith(self._title_lc[idx], q).astype(int)
        order = np.lexsort((-self._counts[idx], -starts))      # prefix match, then popularity
        return [self._movie(int(i)) for i in idx[order][:limit]]

    def popular(self, n: int = 20, genres: list[str] | None = None) -> dict:
        res = self.hybrid.popular(n, self._check_genres(genres))
        return {"mode": "popular", "mode_description": MODE_TEXT["popular"],
                "n_ratings_used": 0, "items": self._items(res)}

    def similar(self, movie_id: int, n: int = 12, genres: list[str] | None = None) -> dict:
        i = self._idx(movie_id)
        res = self.hybrid.similar_items(i, n, self._check_genres(genres))
        return {"mode": "similar", "mode_description": MODE_TEXT["similar"],
                "movie": self._movie(i), "items": self._items(res)}

    def recommend(self, ratings: dict[int, float], n: int = 12,
                  genres: list[str] | None = None) -> dict:
        """Personalised recommendations for an anonymous user described by ratings."""
        if not ratings:
            out = self.popular(n, genres)
            # same meaning as for any user: how many more ratings unlock collaborative filtering
            out["ratings_needed_for_cf"] = self.hybrid.min_ratings_cf
            return out
        idx = np.array([self._idx(m) for m in ratings], dtype=int)
        profile = UserProfile(idx, np.array(list(ratings.values()), dtype=float))
        return self._recommend_profile(profile, n, genres)

    def recommend_for_user(self, user_id: int, n: int = 12,
                           genres: list[str] | None = None) -> dict:
        """Recommendations for an existing MovieLens user (demo feature)."""
        if int(user_id) not in self._user_hist:
            raise NotFoundError(f"User {user_id} not found in the training data")
        i, r = self._user_hist[int(user_id)]
        return self._recommend_profile(UserProfile(i, r), n, genres)

    def _recommend_profile(self, profile: UserProfile, n, genres) -> dict:
        res = self.hybrid.recommend(profile, n, self._check_genres(genres))
        mode = res.mode
        return {"mode": mode, "mode_description": MODE_TEXT.get(mode, ""),
                "n_ratings_used": profile.n,
                "ratings_needed_for_cf": max(0, self.hybrid.min_ratings_cf - profile.n),
                "items": self._items(res, profile)}

    def get_metrics(self) -> dict:
        if not self.metrics:
            raise NotFoundError("metrics.json not found. Run: python scripts/train.py")
        return self.metrics
