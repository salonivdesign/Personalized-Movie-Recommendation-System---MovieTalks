"""Correctness tests. They run on a SYNTHETIC fixture, so they check that the
code behaves (no leakage, valid shapes, reproducible, sensible API errors),
not how good the recommendations are on real data."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config as cfg
from api.main import create_app
from recsys import data as D
from recsys import metrics as M
from recsys.hybrid import rank_normalize, top_n_indices
from recsys.pipeline import run_training


# ---- data ---------------------------------------------------------------------
def test_cleaning_removes_bad_rows(trained):
    rep = trained["report"]["data_cleaning"]
    assert rep["dropped_out_of_range"] == 3
    assert rep["dropped_unknown_movie"] == 3
    assert rep["dropped_duplicates"] == 3


def test_split_is_chronological_and_disjoint(trained):
    raw, movies, _, links = D.load_raw(trained["data_dir"])
    movies = D.clean_movies(movies, links)
    ratings, _ = D.clean_ratings(raw, movies["movieId"])
    train, val, test = D.temporal_user_split(ratings, 0.1, 0.2, 5)
    assert len(train) + len(val) + len(test) == len(ratings)
    key = lambda d: set(zip(d.userId, d.movieId))
    assert not (key(train) & key(val)) and not (key(train) & key(test)) and not (key(val) & key(test))
    for uid, t in test.groupby("userId"):
        assert t.timestamp.min() >= train[train.userId == uid].timestamp.max()
        v = val[val.userId == uid]
        assert v.timestamp.max() <= t.timestamp.min()
        assert train[train.userId == uid].timestamp.max() <= v.timestamp.min()


def test_tags_only_from_allowed_pairs():
    tags = pd.DataFrame({"userId": [1, 2], "movieId": [10, 10], "tag": ["Funny", "Dark!"],
                         "timestamp": [0, 0]})
    allowed = pd.DataFrame({"userId": [1], "movieId": [10]})
    assert D.build_tag_text(tags, allowed).to_dict() == {10: "funny"}


def test_movie_title_cleaning():
    movies = pd.DataFrame({"movieId": [2, 1], "title": ["Matrix, The (1999)", "Heat (1995)"],
                           "genres": ["Action|Sci-Fi", "(no genres listed)"]})
    m = D.clean_movies(movies)
    assert list(m.movieId) == [1, 2]
    assert m.loc[1, "display_title"] == "The Matrix"
    assert m.loc[1, "year"] == 1999 and m.loc[0, "genre_list"] == []


# ---- metrics ------------------------------------------------------------------
def test_metrics_hand_computed():
    rec, rel = [5, 1, 9, 2], {1, 2, 7}
    assert M.precision_at_k(rec, rel, 4) == 0.5
    assert M.recall_at_k(rec, rel, 4) == pytest.approx(2 / 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert M.ndcg_at_k(rec, rel, 4) == pytest.approx(dcg / ideal)
    assert M.rmse([1, 3], [2, 5]) == pytest.approx(np.sqrt(2.5))
    assert M.mae([1, 3], [2, 5]) == pytest.approx(1.5)
    assert M.coverage([[1, 2], [2, 3]], 10) == 0.3


def test_rank_normalize_and_top_n():
    x = np.array([3.0, np.nan, 1.0, 2.0])
    r = rank_normalize(x)
    assert r[2] == 0 and r[0] == 1 and np.isnan(r[1])
    assert list(top_n_indices(np.array([0.1, 0.9, 0.5]), 2)) == [1, 2]


# ---- training artifacts ---------------------------------------------------------
def test_artifacts_written(trained):
    art = trained["artifacts"]
    assert (art / cfg.BUNDLE_FILENAME).exists()
    metrics = json.loads((art / cfg.METRICS_FILENAME).read_text())
    for key in ("rating_prediction", "ranking", "cold_start_simulation", "split"):
        assert key in metrics
    assert "#### Rating prediction" in (art / "RESULTS.md").read_text()


def test_training_is_reproducible(trained, tmp_path):
    run_training(trained["data_dir"], tmp_path, tune_cf=False, seed=7)
    a = joblib.load(trained["artifacts"] / cfg.BUNDLE_FILENAME)["hybrid"].cf.Q_
    b = joblib.load(tmp_path / cfg.BUNDLE_FILENAME)["hybrid"].cf.Q_
    np.testing.assert_allclose(a, b)


# ---- API ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app(trained["artifacts"])) as c:
        yield c


def test_health_and_genres(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Comedy" in client.get("/api/genres").json()


def test_search_and_movie_lookup(client):
    res = client.get("/api/movies/search", params={"q": "test movie 1"}).json()
    assert res and all("test movie 1" in m["title"].lower() for m in res)
    mid = res[0]["movie_id"]
    assert client.get(f"/api/movies/{mid}").json()["movie_id"] == mid
    assert client.get("/api/movies/123456789").status_code == 404
    assert client.get("/api/movies/search", params={"q": ""}).status_code == 422


def test_similar_excludes_self(client):
    mid = client.get("/api/movies/search", params={"q": "test movie"}).json()[0]["movie_id"]
    body = client.get(f"/api/movies/{mid}/similar", params={"n": 8}).json()
    assert len(body["items"]) == 8 and mid not in [m["movie_id"] for m in body["items"]]


def test_popular_with_genre_filter(client):
    body = client.get("/api/popular", params={"n": 5, "genres": "Comedy"}).json()
    assert body["items"] and all("Comedy" in m["genres"] for m in body["items"])
    assert client.get("/api/popular", params={"genres": "NotAGenre"}).status_code == 422


def test_recommend_routing_by_history_size(client):
    movies = client.get("/api/popular", params={"n": 12}).json()["items"]
    ids = [m["movie_id"] for m in movies]

    def call(k):
        return client.post("/api/recommend", json={
            "ratings": [{"movie_id": i, "rating": 5.0 if j % 2 else 3.5} for j, i in enumerate(ids[:k])],
            "n": 10}).json()

    assert call(0)["mode"] == "popular"
    cold = call(2)
    assert cold["mode"] == "cold_start" and cold["ratings_needed_for_cf"] == 3
    full = call(8)
    assert full["mode"] == "hybrid" and full["ratings_needed_for_cf"] == 0
    rated = set(ids[:8])
    assert len(full["items"]) == 10 and not rated & {m["movie_id"] for m in full["items"]}


def test_recommend_validation(client):
    bad = [
        {"ratings": [{"movie_id": 3, "rating": 6}]},
        {"ratings": [{"movie_id": 3, "rating": 3.3}]},
        {"ratings": [{"movie_id": 3, "rating": 3}, {"movie_id": 3, "rating": 4}]},
        {"ratings": [], "n": 0},
        {"ratings": [], "genres": ["<script>"]},
    ]
    for payload in bad:
        assert client.post("/api/recommend", json=payload).status_code == 422, payload
    unknown = client.post("/api/recommend", json={"ratings": [{"movie_id": 999999, "rating": 4}]})
    assert unknown.status_code == 404


def test_demo_user_and_metrics(client):
    ok = client.get("/api/users/1/recommend", params={"n": 5})
    assert ok.status_code == 200 and ok.json()["mode"] == "hybrid"
    assert client.get("/api/users/99999/recommend").status_code == 404
    assert "ranking" in client.get("/api/metrics").json()


def test_api_without_model_returns_503(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/api/health").json()["status"] == "no_model"
        assert c.get("/api/genres").status_code == 503


def test_unexpected_error_returns_clean_500(trained, monkeypatch):
    with TestClient(create_app(trained["artifacts"]), raise_server_exceptions=False) as c:
        def boom(*a, **k):
            raise RuntimeError("secret internal detail")
        monkeypatch.setattr(c.app.state.service, "search", boom)
        r = c.get("/api/movies/search", params={"q": "x"})
        assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
