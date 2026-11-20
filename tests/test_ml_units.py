"""Unit tests for ML internals. All data here is SYNTHETIC (tests/make_fixture.py):
these tests check correctness properties (shapes, hand-computed values, no leakage,
fallbacks), not recommendation quality on real MovieLens."""
import numpy as np
import pandas as pd
import pytest

import config as cfg
from recsys import evaluate as E
from recsys import metrics as M
from recsys.content import ContentModel
from recsys.eda import run_eda
from recsys.hybrid import HybridRecommender, UserProfile
from recsys.pipeline import fit_models


# ---- collaborative filtering -----------------------------------------------------
def test_fold_in_zero_residual_gives_zero_user(ml_ctx):
    cf = ml_ctx.cf
    idx = np.arange(6)
    ratings = cf.mu_ + cf.b_i_[idx].astype(float)      # exactly what the model expects
    b_u, p_u = cf.fold_in(idx, ratings)
    assert b_u == pytest.approx(0.0, abs=1e-6)
    assert p_u.shape == (cf.n_factors,) and np.allclose(p_u, 0.0, atol=1e-6)


def test_fold_in_direction_matches_ratings(ml_ctx):
    cf = ml_ctx.cf
    top = np.argsort(-cf.item_count_)[:10]
    b_hi, p_hi = cf.fold_in(top, np.full(10, 5.0))
    b_lo, p_lo = cf.fold_in(top, np.full(10, 1.0))
    assert b_hi > 0 > b_lo
    assert np.nanmean(cf.score_items(b_hi, p_hi)) > np.nanmean(cf.score_items(b_lo, p_lo))


def test_score_items_masks_rarely_rated_items(ml_ctx):
    cf = ml_ctx.cf
    scores = cf.score_items(0.0, np.zeros(cf.n_factors), min_item_ratings=5)
    assert scores.shape == (ml_ctx.n_items,)
    assert np.array_equal(np.isnan(scores), cf.item_count_ < 5)


def test_item_similarity_properties(ml_ctx):
    cf = ml_ctx.cf
    i = int(np.argmax(cf.item_count_))
    sims = cf.item_similarity(i)
    assert sims.shape == (ml_ctx.n_items,)
    assert sims[i] == pytest.approx(1.0, abs=1e-4)
    assert np.nanmax(sims) <= 1 + 1e-4 and np.nanmin(sims) >= -1 - 1e-4
    masked = cf.item_similarity(i, min_item_ratings=10)
    assert np.array_equal(np.isnan(masked), cf.item_count_ < 10)


# ---- item cold start through the hybrid --------------------------------------------
@pytest.fixture(scope="module")
def cold_item(ml_ctx):
    """A hybrid fit WITHOUT any ratings of one movie, simulating a brand-new movie."""
    order = np.argsort(-ml_ctx.pop.count_)
    victim = next(int(i) for i in order[20:] if ml_ctx.movies["genre_list"].iloc[i])
    train = ml_ctx.train[ml_ctx.train["iidx"] != victim]
    pop, content, cf = fit_models(train, ml_ctx.movies, ml_ctx.tags, cfg.CF_PARAMS, seed=7)
    return victim, HybridRecommender(ml_ctx.movies, pop, content, cf)


def test_new_movie_has_no_cf_signal_but_is_still_recommendable(ml_ctx, cold_item):
    victim, hybrid = cold_item
    assert hybrid.cf.item_count_[victim] == 0
    assert hybrid.cf.item_similarity(victim) is None

    hist = E.build_histories(ml_ctx.train)
    # a user who has NOT rated the movie (rated movies are correctly excluded)
    uid = next(u for u, p in hist.items() if p.n >= 10 and victim not in p.item_idx)
    res = hybrid.recommend(hist[uid], n=hybrid.n_items)       # rank every candidate
    pos = np.flatnonzero(res.item_idx == victim)
    assert len(pos) == 1 and np.isfinite(res.scores[pos[0]])   # scored by content + popularity


def test_similar_items_for_new_movie_falls_back_to_content(cold_item):
    victim, hybrid = cold_item
    res = hybrid.similar_items(victim, n=8)
    assert set(res.weights) == {"content"}
    assert len(res.item_idx) == 8 and victim not in res.item_idx
    assert set(res.reasons) == {"content"}


def test_similar_items_basic_and_genre_filter(ml_ctx):
    hybrid = ml_ctx.hybrid
    i = int(np.argmax(ml_ctx.pop.count_))
    res = hybrid.similar_items(i, n=8)
    assert len(res.item_idx) == 8 and i not in res.item_idx
    assert np.all(np.diff(res.scores) <= 1e-12)                 # best first
    comedy = hybrid.similar_items(i, n=8, genres=["Comedy"])
    assert comedy.item_idx.size and all(
        "Comedy" in ml_ctx.movies["genre_list"].iloc[j] for j in comedy.item_idx)


# ---- content model --------------------------------------------------------------------
def _tiny_movies(genres, years=None):
    n = len(genres)
    return pd.DataFrame({"movieId": range(1, n + 1), "genre_list": genres,
                         "year": years if years is not None else [np.nan] * n})


def test_diversity_hand_computed():
    movies = _tiny_movies([["Action"], ["Action"], ["Comedy"], ["Comedy"]])
    model = ContentModel().fit(movies, pd.Series(dtype=object))
    assert model.diversity([0, 1]) == pytest.approx(0.0, abs=1e-6)    # identical vectors
    assert model.diversity([0, 2]) == pytest.approx(1.0, abs=1e-6)    # no shared tokens
    assert model.diversity([0]) == 0.0


def test_content_model_falls_back_for_tiny_or_metadata_free_catalogs():
    # every token unique -> min_df=2 would leave nothing; retry with min_df=1 keeps signal
    unique = ContentModel().fit(_tiny_movies([["Action"], ["Comedy"]]), pd.Series(dtype=object))
    assert unique.vectorizer_ is not None and unique.matrix_.nnz > 0
    # no genres/years/tags at all -> no features, but no crash
    empty = ContentModel().fit(_tiny_movies([[], [], []]), pd.Series(dtype=object))
    assert empty.vectorizer_ is None and empty.matrix_.shape == (3, 1)
    assert not np.any(empty.similar_scores(0))
    assert empty.profile_scores(np.array([0, 1]), np.array([1.0, -1.0])) is None


# ---- metrics ------------------------------------------------------------------------------
def test_novelty_hand_computed():
    counts = np.array([10, 5, 1])
    # popularity share: item0 = 10/10 = 1.0 -> 0 bits; item2 = 1/10 -> log2(10) bits
    assert M.novelty([0, 2], counts, n_users=10) == pytest.approx(np.log2(10) / 2)


def test_rating_prediction_ablation_matches_manual_baseline(ml_ctx):
    out = E.evaluate_rating_prediction(ml_ctx.cf, ml_ctx.test)
    assert set(out) >= {"global_mean", "item_bias", "user+item_bias", "matrix_factorization"}
    manual = M.rmse(ml_ctx.test["rating"], np.full(len(ml_ctx.test), ml_ctx.cf.mu_))
    assert out["global_mean"]["rmse"] == pytest.approx(manual)
    assert out["n_test_ratings"] == len(ml_ctx.test)


# ---- tuning and evaluation helpers --------------------------------------------------------------
def test_weight_grid_is_a_valid_simplex():
    grid = E.weight_grid(0.1)
    assert len(grid) == 66                                   # C(10+2, 2) combinations
    assert all(set(w) == {"cf", "content", "pop"} for w in grid)
    assert all(abs(sum(w.values()) - 1) < 1e-9 and min(w.values()) >= 0 for w in grid)
    assert len({tuple(w.values()) for w in grid}) == 66
    assert len(E.weight_grid(0.5)) == 6


def test_tune_hybrid_weights_picks_the_best_grid_point(ml_ctx):
    hybrid = ml_ctx.hybrid
    hist = E.build_histories(ml_ctx.train)
    rel = E.build_relevant(ml_ctx.val, cfg.RELEVANCE_THRESHOLD)
    weights, table = E.tune_hybrid_weights(hybrid, hist, rel, step=0.1, k=10)

    assert sum(weights.values()) == pytest.approx(1.0)
    *rows, info = table
    assert info["n_users_used"] > 0
    scores = [r["val_ndcg@10"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert {k: rows[0][k] for k in ("cf", "content", "pop")} == weights

    # independent recomputation through the public recommend() path
    users = [u for u in sorted(rel) if rel[u] and u in hist and hist[u].n >= hybrid.min_ratings_cf]
    assert len(users) == info["n_users_used"]
    mean_ndcg = np.mean([M.ndcg_at_k(hybrid.recommend(hist[u], 10, weights=weights).item_idx,
                                     rel[u], 10) for u in users])
    assert mean_ndcg == pytest.approx(rows[0]["val_ndcg@10"], abs=1e-9)


def test_cold_start_curve_structure_and_fallback(ml_ctx):
    hybrid = ml_ctx.hybrid
    hist = E.build_histories(ml_ctx.train)
    rel = E.build_relevant(ml_ctx.val, cfg.RELEVANCE_THRESHOLD)
    out = E.cold_start_curve(hybrid, hist, rel, ml_ctx.pop.count_, len(hist), seed=1,
                             sizes=(0, 1, 5), k=10)
    assert list(out) == ["0", "1", "5"]
    assert [out[m]["_mode_used_by_hybrid"] for m in out] == ["popular", "cold_start", "hybrid"]
    zero = [v["ndcg@10"] for name, v in out["0"].items() if not name.startswith("_")]
    assert len(set(zero)) == 1                       # with no history every model = popularity
    for m in out.values():
        for name, v in m.items():
            if not name.startswith("_"):
                assert 0.0 <= v["ndcg@10"] <= 1.0 and 0.0 <= v["recall@10"] <= 1.0


def test_hidden_history_is_never_recommended(ml_ctx):
    """Leakage guard for the cold-start simulation: items hidden via ``exclude`` stay hidden."""
    hist = E.build_histories(ml_ctx.train)
    uid = next(u for u, p in hist.items() if p.n >= 12)
    h = hist[uid]
    profile = UserProfile(h.item_idx[:3], h.ratings[:3], exclude=h.item_idx[3:])
    res = ml_ctx.hybrid.recommend(profile, n=60)
    assert not set(res.item_idx.tolist()) & set(h.item_idx.tolist())


# ---- EDA ------------------------------------------------------------------------------------------
def test_run_eda_outputs_and_consistency(ml_ctx, tmp_path):
    summary = run_eda(ml_ctx.ratings, ml_ctx.movies, tmp_path)
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert len(pngs) == 6 and all((tmp_path / p).stat().st_size > 1000 for p in pngs)
    assert (tmp_path / "eda_summary.json").exists()
    assert summary["n_ratings"] == len(ml_ctx.ratings)
    assert sum(summary["rating_value_counts"].values()) == len(ml_ctx.ratings)
    assert 0 < summary["sparsity"] < 1
    assert 0 <= summary["share_movies_with_lt5_ratings"] <= 1
    assert set(summary["genre_counts"]) <= set(ml_ctx.hybrid.genres)
