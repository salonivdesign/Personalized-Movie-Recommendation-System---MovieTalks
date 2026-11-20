"""Content-based filtering with TF-IDF + cosine similarity.

Each movie becomes a short "document": genre tokens, a release-decade token and
user-applied tags. TF-IDF down-weights tokens that appear everywhere (e.g.
"drama") and up-weights distinctive ones (e.g. "cyberpunk"). Rows are
L2-normalised, so a dot product between two rows *is* their cosine similarity.

We never build the full item x item matrix (9k x 9k floats is ~300 MB). Instead
we compute one row of similarities on demand with a sparse matrix product.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def _genre_token(genre: str) -> str:
    # "Sci-Fi" -> "scifi", "Children's" -> "childrens", "Film-Noir" -> "filmnoir"
    return re.sub(r"[^a-z0-9]", "", genre.lower())


class ContentModel:
    def __init__(self, genre_repeat: int = 2):
        self.genre_repeat = genre_repeat

    def _document(self, genres: list[str], year: float, tag_text: str) -> str:
        toks = [_genre_token(g) for g in genres] * self.genre_repeat
        if not np.isnan(year):
            toks.append(f"decade{int(year) // 10 * 10}")
        return " ".join(toks) + " " + tag_text

    def fit(self, movies: pd.DataFrame, tag_text: pd.Series) -> "ContentModel":
        tag_lookup = tag_text.to_dict() if len(tag_text) else {}
        docs = [
            self._document(g, y, tag_lookup.get(mid, ""))
            for g, y, mid in zip(movies["genre_list"], movies["year"], movies["movieId"])
        ]
        # min_df=2 drops one-off tags (noise) but keeps all genre/decade tokens.
        # Tiny catalogs can lose every token that way, so retry with min_df=1.
        for min_df in (2, 1):
            vec = TfidfVectorizer(
                stop_words="english", sublinear_tf=True, min_df=min_df,
                token_pattern=r"(?u)\b[a-z0-9]{2,}\b")
            try:
                self.matrix_ = vec.fit_transform(docs).tocsr().astype(np.float32)
                self.vectorizer_ = vec
                return self
            except ValueError:        # "empty vocabulary"
                continue
        # Dataset without usable metadata (no genres, years or tags): the content
        # model degrades to "no signal" and the hybrid relies on CF + popularity.
        # Note: diversity() is not meaningful in this case.
        self.vectorizer_ = None
        self.matrix_ = sparse.csr_matrix((len(docs), 1), dtype=np.float32)
        return self

    # ---- scoring -----------------------------------------------------------
    def similar_scores(self, item_idx: int) -> np.ndarray:
        """Cosine similarity of every movie to movie ``item_idx`` (dense, length n_items)."""
        return np.asarray((self.matrix_ @ self.matrix_[item_idx].T).todense()).ravel()

    def profile_scores(self, item_idx: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
        """Score all movies against a user taste profile.

        profile = sum_i weights_i * tfidf_i  (weights are mean-centred ratings,
        so liked items pull the profile toward them and disliked items push it
        away). Returns None when the profile has no signal.
        """
        rows = self.matrix_[item_idx]
        profile = np.asarray(rows.T @ weights.astype(np.float32)).ravel()
        norm = float(np.linalg.norm(profile))
        if norm < 1e-9:
            return None
        return np.asarray(self.matrix_ @ (profile / norm)).ravel()

    def diversity(self, item_idx) -> float:
        """Intra-list diversity: mean (1 - cosine) over all pairs in the list."""
        idx = np.asarray(item_idx)
        n = len(idx)
        if n < 2:
            return 0.0
        rows = self.matrix_[idx]
        sim = np.asarray((rows @ rows.T).todense())
        off_diag_mean = (sim.sum() - np.trace(sim)) / (n * (n - 1))
        return float(1.0 - off_diag_mean)
