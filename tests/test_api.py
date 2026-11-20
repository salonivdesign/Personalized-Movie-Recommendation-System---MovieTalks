"""Backend (FastAPI) tests.

!! All data here is SYNTHETIC (tests/make_fixture.py). These tests check that the API
!! behaves correctly: schemas, validation, error handling, pagination, ordering,
!! consistency with the ML service. They say NOTHING about recommendation quality on
!! real MovieLens data, which has not been evaluated yet.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config as cfg
from api.main import create_app
from api.schemas import HealthOut, MovieListOut, MovieOut, RecommendResponse, SimilarResponse
from make_fixture import make_fixture
from recsys.pipeline import run_training

EXPECTED_PATHS = {
    "/api/health", "/api/genres", "/api/movies", "/api/movies/search", "/api/movies/{movie_id}",
    "/api/movies/{movie_id}/similar", "/api/popular", "/api/recommend",
    "/api/users/{user_id}/recommend", "/api/metrics",
}
MODEL_ENDPOINTS = [   # (method, url, json) - everything that needs a trained model
    ("GET", "/api/genres", None), ("GET", "/api/movies", None),
    ("GET", "/api/movies/search?q=a", None), ("GET", "/api/movies/3", None),
    ("GET", "/api/movies/3/similar", None), ("GET", "/api/popular", None),
    ("POST", "/api/recommend", {"ratings": []}), ("GET", "/api/users/1/recommend", None),
    ("GET", "/api/metrics", None),
]


# ---- helpers ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app(trained["artifacts"])) as c:
        yield c


@pytest.fixture(scope="module")
def svc(client):
    return client.app.state.service


def popular_ids(client, n=12):
    return [m["movie_id"] for m in client.get("/api/popular", params={"n": n}).json()["items"]]


def rate(ids, value=4.0):
    return [{"movie_id": i, "rating": value} for i in ids]


def assert_error(r, status, contains=None):
    """Every error has the same shape: {"detail": "<readable message>"}."""
    assert r.status_code == status, r.text
    body = r.json()
    assert isinstance(body["detail"], str) and body["detail"]
    if contains:
        assert contains.lower() in r.text.lower(), r.text


def assert_field_error(r, contains):
    """Validation errors add a per-field list."""
    assert_error(r, 422, contains)
    errors = r.json()["errors"]
    assert errors and all({"field", "message"} <= set(e) for e in errors)


def broken_app(tmp_path, **kwargs):
    return TestClient(create_app(tmp_path))


# ---- API contract / documentation -------------------------------------------------------------
def test_openapi_documents_every_planned_endpoint(client):
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == EXPECTED_PATHS
    assert client.get("/docs").status_code == 200
    assert "ValidationErrorOut" in spec["components"]["schemas"]
    tags = {t["name"] for t in spec["tags"]}
    assert tags == {"meta", "catalog", "recommend"}


def test_health_reports_model_dataset_and_rating_scale(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    h = HealthOut.model_validate(r.json())
    assert h.status == "ok" and h.n_movies == 300 and h.n_users == 120
    assert h.dataset_name == "data" and h.synthetic_data is True       # the fixture folder
    assert (h.rating_scale.min, h.rating_scale.max, h.rating_scale.step) == (
        cfg.RATING_MIN, cfg.RATING_MAX, cfg.RATING_STEP)
    assert h.relevance_threshold == cfg.RELEVANCE_THRESHOLD


def test_genres_and_metrics(client):
    genres = client.get("/api/genres").json()
    assert genres == sorted(genres) and "Comedy" in genres
    metrics = client.get("/api/metrics").json()
    assert {"meta", "ranking", "rating_prediction", "cold_start_simulation"} <= set(metrics)
    assert metrics["meta"]["dataset"]["synthetic"] is True


def test_metrics_404_when_metrics_file_is_missing_or_corrupt(trained, tmp_path):
    (tmp_path / cfg.BUNDLE_FILENAME).write_bytes((trained["artifacts"] / cfg.BUNDLE_FILENAME).read_bytes())
    with TestClient(create_app(tmp_path)) as c:                       # no metrics.json at all
        assert_error(c.get("/api/metrics"), 404, "metrics.json")
        assert c.get("/api/popular").status_code == 200               # rest still works
    (tmp_path / cfg.METRICS_FILENAME).write_text("{ not json")
    with TestClient(create_app(tmp_path)) as c:                       # corrupt metrics.json
        assert_error(c.get("/api/metrics"), 404)
        assert c.get("/api/health").json()["status"] == "ok"


# ---- model not available --------------------------------------------------------------------------
def assert_all_model_endpoints_503(c, contains):
    health = c.get("/api/health")
    assert health.status_code == 200 and health.json()["status"] == "no_model"
    assert contains.lower() in health.json()["detail"].lower()
    for method, url, body in MODEL_ENDPOINTS:
        r = c.request(method, url, json=body)
        assert_error(r, 503, contains)


def test_missing_artifacts_give_clear_503s(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert_all_model_endpoints_503(c, "scripts/train.py")


def test_corrupt_bundle_does_not_crash_startup_or_leak_paths(tmp_path):
    (tmp_path / cfg.BUNDLE_FILENAME).write_bytes(b"this is not a joblib file")
    with TestClient(create_app(tmp_path)) as c:
        assert_all_model_endpoints_503(c, "could not be read")
        assert str(tmp_path) not in c.get("/api/health").text


def test_incomplete_bundle_is_reported(tmp_path):
    joblib.dump({"movies": 1}, tmp_path / cfg.BUNDLE_FILENAME)
    with TestClient(create_app(tmp_path)) as c:
        assert_all_model_endpoints_503(c, "incomplete")


def test_model_trained_with_a_different_rating_scale_is_refused(trained, monkeypatch):
    monkeypatch.setattr(cfg, "RATING_MIN", 1.0)           # config.py changed after training
    with TestClient(create_app(trained["artifacts"])) as c:
        assert_all_model_endpoints_503(c, "rating scale")


def test_requests_before_startup_get_503_not_a_crash(trained):
    c = TestClient(create_app(trained["artifacts"]))      # no `with`: lifespan has not run
    assert c.get("/api/health").json()["status"] == "no_model"
    assert_error(c.get("/api/popular"), 503)


# ---- catalog: listing -----------------------------------------------------------------------------------
def test_list_movies_pagination_is_consistent(client):
    first = client.get("/api/movies", params={"limit": 10, "offset": 0}).json()
    second = client.get("/api/movies", params={"limit": 10, "offset": 10}).json()
    both = client.get("/api/movies", params={"limit": 20}).json()
    MovieListOut.model_validate(first)
    assert first["total"] == second["total"] == both["total"] == 300
    assert (first["limit"], first["offset"], second["offset"]) == (10, 0, 10)
    ids = lambda page: [m["movie_id"] for m in page["items"]]
    assert ids(first) + ids(second) == ids(both)
    assert len(set(ids(both))) == 20                                   # no duplicates


def test_list_movies_offset_past_the_end_is_empty_not_an_error(client):
    r = client.get("/api/movies", params={"offset": 299, "limit": 10}).json()
    assert len(r["items"]) == 1 and r["total"] == 300
    r = client.get("/api/movies", params={"offset": 5000}).json()
    assert r["items"] == [] and r["total"] == 300


def test_list_movies_sort_orders(client, svc):
    def listing(sort, **kw):
        return client.get("/api/movies", params={"sort": sort, "limit": 100, **kw}).json()["items"]

    popular = listing("popular")
    counts = [m["n_ratings"] for m in popular]
    assert counts == sorted(counts, reverse=True)
    for a, b in zip(popular, popular[1:]):                              # ties: lowest id first
        if a["n_ratings"] == b["n_ratings"]:
            assert a["movie_id"] < b["movie_id"]

    top = listing("top_rated")
    scores = [svc.hybrid.pop.score_[svc._id_to_idx[m["movie_id"]]] for m in top]
    assert scores == sorted(scores, reverse=True)

    years = [m["year"] for m in listing("year")]
    known = [y for y in years if y is not None]
    assert known == sorted(known, reverse=True)
    assert years[:len(known)] == known                                  # unknown years last

    titles = [m["title"].lower() for m in listing("title")]
    assert titles == sorted(titles)


def test_list_movies_filters(client):
    comedy = client.get("/api/movies", params={"genre": "Comedy", "limit": 100}).json()
    assert 0 < comedy["total"] < 300
    assert all("Comedy" in m["genres"] for m in comedy["items"])
    popular = client.get("/api/movies", params={"min_ratings": 20, "limit": 100}).json()
    assert 0 < popular["total"] < 300 and all(m["n_ratings"] >= 20 for m in popular["items"])
    assert client.get("/api/movies", params={"min_ratings": 10**6}).json()["items"] == []


@pytest.fixture(scope="module")
def client_with_missing_years(tmp_path_factory):
    """Synthetic dataset where every 4th movie title has no "(YYYY)" -> year is None."""
    root = tmp_path_factory.mktemp("missing-years")
    d = make_fixture(root / "data", n_users=60, n_movies=120, seed=11)
    movies = pd.read_csv(d / "movies.csv")
    rows = movies.index % 4 == 0
    movies.loc[rows, "title"] = movies.loc[rows, "title"].str.replace(r" \(\d{4}\)$", "", regex=True)
    movies.to_csv(d / "movies.csv", index=False)
    run_training(d, root / "art", tune_cf=False, seed=11)
    with TestClient(create_app(root / "art")) as c:
        yield c


def test_year_sort_puts_movies_without_a_year_last(client_with_missing_years):
    c = client_with_missing_years
    items = []
    for offset in (0, 100):
        items += c.get("/api/movies", params={"sort": "year", "limit": 100, "offset": offset}).json()["items"]
    years = [m["year"] for m in items]
    assert len(items) == 120 and years.count(None) == 30
    known = [y for y in years if y is not None]
    assert years == known + [None] * 30                         # unknown years all at the end
    assert known == sorted(known, reverse=True)                 # newest first


@pytest.mark.parametrize("params, needle", [
    ({"sort": "bogus"}, "sort"), ({"limit": 0}, "limit"), ({"limit": 101}, "limit"),
    ({"limit": "abc"}, "limit"), ({"offset": -1}, "offset"), ({"min_ratings": -1}, "min_ratings"),
])
def test_list_movies_rejects_bad_parameters(client, params, needle):
    assert_field_error(client.get("/api/movies", params=params), needle)


def test_list_movies_unknown_genre(client):
    assert_error(client.get("/api/movies", params={"genre": "NotAGenre"}), 422, "Unknown genres")


# ---- catalog: search and lookup --------------------------------------------------------------------------------
def test_search_matches_case_insensitively_and_ranks_prefix_then_popularity(client):
    res = client.get("/api/movies/search", params={"q": "TEST MOVIE 1", "limit": 30}).json()
    [MovieOut.model_validate(m) for m in res]
    assert res and all("test movie 1" in m["title"].lower() or
                       all(t in m["title"].lower() for t in ("test", "movie", "1")) for m in res)
    is_prefix = [m["title"].lower().startswith("test movie 1") for m in res]
    assert is_prefix == sorted(is_prefix, reverse=True)                 # prefix matches first
    prefix = [m["n_ratings"] for m, p in zip(res, is_prefix) if p]
    assert prefix == sorted(prefix, reverse=True)                       # then most popular


def test_search_limit_genre_and_empty_results(client):
    assert len(client.get("/api/movies/search", params={"q": "movie", "limit": 5}).json()) == 5
    comedy = client.get("/api/movies/search", params={"q": "movie", "genre": "Comedy", "limit": 30}).json()
    assert comedy and all("Comedy" in m["genres"] for m in comedy)
    r = client.get("/api/movies/search", params={"q": "zzzz-no-such-title"})
    assert r.status_code == 200 and r.json() == []


@pytest.mark.parametrize("q", ["' OR 1=1 --", ".*", "%", "\\", "<script>alert(1)</script>", "é☃", "a" * 80])
def test_search_treats_special_characters_as_plain_text(client, q):
    r = client.get("/api/movies/search", params={"q": q})
    assert r.status_code == 200 and isinstance(r.json(), list)


@pytest.mark.parametrize("params, needle", [
    ({}, "q"), ({"q": ""}, "q"), ({"q": "a" * 81}, "q"),
    ({"q": "x", "limit": 0}, "limit"), ({"q": "x", "limit": 31}, "limit"),
])
def test_search_rejects_bad_parameters(client, params, needle):
    assert_field_error(client.get("/api/movies/search", params=params), needle)


def test_search_blank_query_and_unknown_genre(client):
    assert_error(client.get("/api/movies/search", params={"q": "   "}), 422, "empty")
    assert_error(client.get("/api/movies/search", params={"q": "a", "genre": "Nope"}), 422, "Unknown genres")


def test_get_movie_by_id(client):
    found = client.get("/api/movies/search", params={"q": "test movie 3"}).json()[0]
    r = client.get(f"/api/movies/{found['movie_id']}")
    assert r.status_code == 200 and r.json() == found
    assert MovieOut.model_validate(r.json()).imdb_url.startswith("https://www.imdb.com/title/tt")


@pytest.mark.parametrize("movie_id", ["123456789", str(10**30)])
def test_unknown_movie_is_404_and_names_the_id(client, movie_id):
    assert_error(client.get(f"/api/movies/{movie_id}"), 404, movie_id)


@pytest.mark.parametrize("movie_id", ["0", "-5", "abc", "1.5"])
def test_malformed_movie_id_is_422(client, movie_id):
    assert_field_error(client.get(f"/api/movies/{movie_id}"), "movie_id")


# ---- similar movies ----------------------------------------------------------------------------------------------------
def test_similar_movies_response(client):
    mid = popular_ids(client, 1)[0]
    r = client.get(f"/api/movies/{mid}/similar", params={"n": 10})
    body = SimilarResponse.model_validate(r.json())
    assert body.mode == "similar" and body.movie.movie_id == mid
    assert len(body.items) == 10 and mid not in [m.movie_id for m in body.items]
    scores = [m.score for m in body.items]
    assert scores == sorted(scores, reverse=True)
    assert body.movie.model_dump() == client.get(f"/api/movies/{mid}").json()
    assert r.json() == client.get(f"/api/movies/{mid}/similar", params={"n": 10}).json()   # deterministic


def test_similar_movies_genre_filter_and_errors(client):
    mid = popular_ids(client, 1)[0]
    items = client.get(f"/api/movies/{mid}/similar", params={"genres": "Comedy"}).json()["items"]
    assert items and all("Comedy" in m["genres"] for m in items)
    assert_error(client.get("/api/movies/123456789/similar"), 404, "123456789")
    assert_error(client.get(f"/api/movies/{mid}/similar", params={"genres": "Nope"}), 422, "Unknown genres")
    for n in (0, 51):
        assert_field_error(client.get(f"/api/movies/{mid}/similar", params={"n": n}), "n")


def test_similar_for_a_movie_nobody_rated_uses_content_only(client, svc):
    unrated = np.flatnonzero(svc._counts == 0)
    assert unrated.size >= 1                      # the fixture contains an unrated movie
    mid = int(svc.movie_ids[unrated[0]])
    body = client.get(f"/api/movies/{mid}/similar", params={"n": 5}).json()
    assert len(body["items"]) == 5 and {m["reason_code"] for m in body["items"]} == {"content"}


# ---- popular -----------------------------------------------------------------------------------------------------------------
def test_popular_response(client):
    r = client.get("/api/popular", params={"n": 15})
    body = RecommendResponse.model_validate(r.json())
    assert body.mode == "popular" and len(body.items) == 15 and body.n_ratings_used == 0
    scores = [m.score for m in body.items]
    assert scores == sorted(scores, reverse=True)
    assert {m.predicted_rating for m in body.items} == {None}
    assert len({m.movie_id for m in body.items}) == 15


def test_popular_filters_and_bounds(client):
    items = client.get("/api/popular", params={"genres": "Comedy, Drama", "n": 30}).json()["items"]
    assert items and all({"Comedy", "Drama"} & set(m["genres"]) for m in items)
    assert_error(client.get("/api/popular", params={"genres": "Nope"}), 422, "Unknown genres")
    for n in (0, 51):
        assert_field_error(client.get("/api/popular", params={"n": n}), "n")


# ---- personalised recommendations ------------------------------------------------------------------------------------------
def test_recommend_response_schema_and_exclusions(client):
    ids = popular_ids(client, 12)
    r = client.post("/api/recommend", json={"ratings": rate(ids[:8]), "n": 10})
    body = RecommendResponse.model_validate(r.json())
    assert body.mode == "hybrid" and body.n_ratings_used == 8 and body.ratings_needed_for_cf == 0
    assert len(body.items) == 10
    returned = [m.movie_id for m in body.items]
    assert len(set(returned)) == 10 and not set(returned) & set(ids[:8])
    lo, hi = cfg.RATING_MIN, cfg.RATING_MAX
    assert all(m.predicted_rating is not None and lo <= m.predicted_rating <= hi for m in body.items)


def test_recommend_is_deterministic_and_ignores_rating_order(client):
    ids = popular_ids(client, 10)
    ratings = [{"movie_id": i, "rating": cfg.RATING_MIN + cfg.RATING_STEP * (k % 5)}
               for k, i in enumerate(ids)]
    a = client.post("/api/recommend", json={"ratings": ratings, "n": 12}).json()
    b = client.post("/api/recommend", json={"ratings": ratings, "n": 12}).json()
    c = client.post("/api/recommend", json={"ratings": ratings[::-1], "n": 12}).json()
    assert a == b
    assert [m["movie_id"] for m in a["items"]] == [m["movie_id"] for m in c["items"]]


@pytest.mark.parametrize("k, mode", [(0, "popular"), (1, "cold_start"), (4, "cold_start"), (5, "hybrid")])
def test_recommend_routing_boundaries_and_n(client, k, mode):
    ids = popular_ids(client, 12)
    body = client.post("/api/recommend", json={"ratings": rate(ids[:k]), "n": 7}).json()
    assert body["mode"] == mode and len(body["items"]) == 7
    assert body["ratings_needed_for_cf"] == max(0, cfg.MIN_RATINGS_FOR_CF - k)


def test_recommend_genre_filter_and_empty_body_defaults(client):
    ids = popular_ids(client, 8)
    items = client.post("/api/recommend", json={"ratings": rate(ids), "genres": ["Drama"], "n": 10}).json()["items"]
    assert items and all("Drama" in m["genres"] for m in items)
    body = client.post("/api/recommend", json={}).json()          # every field has a default
    assert body["mode"] == "popular" and len(body["items"]) == 12


def test_because_field_only_names_movies_the_user_liked(client):
    ids = popular_ids(client, 10)
    title = {i: client.get(f"/api/movies/{i}").json()["title"] for i in ids}
    ratings = [{"movie_id": i, "rating": 5.0 if k < 5 else 1.0} for k, i in enumerate(ids)]
    items = client.post("/api/recommend", json={"ratings": ratings, "n": 30}).json()["items"]
    liked_titles = {title[i] for i in ids[:5]}
    for m in items:
        if m["because"] is not None:
            assert m["reason_code"] == "content" and m["because"] in liked_titles


def test_rating_scale_boundaries_are_accepted(client):
    a, b = popular_ids(client, 2)
    r = client.post("/api/recommend", json={"ratings": [
        {"movie_id": a, "rating": cfg.RATING_MIN}, {"movie_id": b, "rating": cfg.RATING_MAX}]})
    assert r.status_code == 200


def test_rating_every_movie_leaves_nothing_to_recommend_but_is_not_an_error(client, svc):
    all_rated = [{"movie_id": int(m), "rating": 4.0} for m in svc.movie_ids]
    body = client.post("/api/recommend", json={"ratings": all_rated, "n": 10}).json()
    assert body["items"] == [] and body["mode"] == "hybrid"


def test_unknown_movie_in_ratings_is_404(client):
    ids = popular_ids(client, 3)
    r = client.post("/api/recommend", json={"ratings": rate(ids) + rate([123456789])})
    assert_error(r, 404, "123456789")
    assert_error(client.post("/api/recommend", json={"genres": ["Nope"]}), 422, "Unknown genres")


BAD_BODIES = [
    ("rating_above_max", {"ratings": [{"movie_id": 3, "rating": cfg.RATING_MAX + cfg.RATING_STEP}]}, "rating"),
    ("rating_below_min", {"ratings": [{"movie_id": 3, "rating": cfg.RATING_MIN - cfg.RATING_STEP}]}, "rating"),
    ("rating_off_scale", {"ratings": [{"movie_id": 3, "rating": cfg.RATING_MIN + cfg.RATING_STEP / 3}]}, "multiple"),
    ("rating_is_bool", {"ratings": [{"movie_id": 3, "rating": True}]}, "true/false"),
    ("rating_is_text", {"ratings": [{"movie_id": 3, "rating": "abc"}]}, "rating"),
    ("rating_missing", {"ratings": [{"movie_id": 3}]}, "rating"),
    ("movie_id_missing", {"ratings": [{"rating": 4}]}, "movie_id"),
    ("movie_id_zero", {"ratings": [{"movie_id": 0, "rating": 4}]}, "movie_id"),
    ("movie_id_negative", {"ratings": [{"movie_id": -1, "rating": 4}]}, "movie_id"),
    ("movie_id_fraction", {"ratings": [{"movie_id": 1.5, "rating": 4}]}, "movie_id"),
    ("movie_id_is_bool", {"ratings": [{"movie_id": True, "rating": 4}]}, "true/false"),
    ("extra_field_in_rating", {"ratings": [{"movie_id": 3, "rating": 4, "stars": 3}]}, "stars"),
    ("typo_in_top_level_field", {"rating": [{"movie_id": 3, "rating": 4}]}, "extra"),
    ("ratings_not_a_list", {"ratings": "nope"}, "ratings"),
    ("duplicate_movie", {"ratings": [{"movie_id": 3, "rating": 4}, {"movie_id": 3, "rating": 5}]}, "duplicate"),
    ("more_than_500_ratings", {"ratings": [{"movie_id": i, "rating": cfg.RATING_MIN} for i in range(1, 502)]}, "ratings"),
    ("n_zero", {"n": 0}, "n"), ("n_above_50", {"n": 51}, "n"),
    ("n_is_bool", {"n": True}, "true/false"), ("n_is_text", {"n": "many"}, "n"),
    ("genre_with_markup", {"genres": ["<script>"]}, "genre"),
    ("genre_too_long", {"genres": ["x" * 31]}, "genre"),
    ("more_than_10_genres", {"genres": ["Comedy"] * 11}, "genres"),
    ("genres_not_a_list", {"genres": "Comedy"}, "genres"),
]


@pytest.mark.parametrize("body, needle", [(b, n) for _, b, n in BAD_BODIES], ids=[i for i, _, _ in BAD_BODIES])
def test_recommend_rejects_invalid_bodies(client, body, needle):
    assert_field_error(client.post("/api/recommend", json=body), needle)


@pytest.mark.parametrize("raw, headers", [
    (b"not json at all", {"Content-Type": "application/json"}),
    (b"", {"Content-Type": "application/json"}),
    (b'{"ratings": [{"movie_id": 3, "rating": NaN}]}', {"Content-Type": "application/json"}),
    (b'{"ratings": [{"movie_id": 3, "rating": Infinity}]}', {"Content-Type": "application/json"}),
    (b"[]", {"Content-Type": "application/json"}),
    (b'{"ratings": []}', {"Content-Type": "text/plain"}),
], ids=["not_json", "empty_body", "nan_rating", "infinite_rating", "array_instead_of_object", "wrong_content_type"])
def test_recommend_rejects_malformed_raw_requests(client, raw, headers):
    r = client.post("/api/recommend", content=raw, headers=headers)
    assert_error(r, 422)
    assert r.json()["errors"]


# ---- existing MovieLens user (demo) ------------------------------------------------------------------------------------------------
def test_recommend_for_existing_user(client, svc):
    r = client.get("/api/users/1/recommend", params={"n": 9})
    body = RecommendResponse.model_validate(r.json())
    assert body.mode == "hybrid" and len(body.items) == 9
    seen = {int(svc.movie_ids[i]) for i in svc._user_hist[1][0]}
    assert body.n_ratings_used == len(seen) and not seen & {m.movie_id for m in body.items}
    assert r.json() == client.get("/api/users/1/recommend", params={"n": 9}).json()


def test_recommend_for_user_filters_and_errors(client):
    items = client.get("/api/users/1/recommend", params={"genres": "Drama", "n": 10}).json()["items"]
    assert items and all("Drama" in m["genres"] for m in items)
    assert_error(client.get("/api/users/99999/recommend"), 404, "99999")
    assert_error(client.get(f"/api/users/{10**30}/recommend"), 404)
    for bad in ("0", "-1", "abc"):
        assert_field_error(client.get(f"/api/users/{bad}/recommend"), "user_id")
    assert_error(client.get("/api/users/1/recommend", params={"genres": "Nope"}), 422, "Unknown genres")
    assert_field_error(client.get("/api/users/1/recommend", params={"n": 51}), "n")


# ---- generic HTTP behaviour -------------------------------------------------------------------------------------------------------
def test_unknown_routes_and_wrong_methods_return_json_errors(client):
    assert_error(client.get("/api/does-not-exist"), 404)
    assert_error(client.post("/api/health"), 405)
    assert_error(client.get("/api/recommend"), 405)
    assert_error(client.delete("/api/movies/3"), 405)


def test_unexpected_errors_return_a_generic_500_without_leaking_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail /srv/private/path")

    with TestClient(client.app, raise_server_exceptions=False) as c:
        svc = c.app.state.service
        monkeypatch.setattr(svc, "search", boom)
        monkeypatch.setattr(svc, "recommend", boom)
        for r in (c.get("/api/movies/search", params={"q": "x"}),
                  c.post("/api/recommend", json={"ratings": [{"movie_id": 3, "rating": 4}]})):
            assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
            assert "secret" not in r.text and "/srv" not in r.text


def test_cors_preflight_allows_the_frontend_by_default(client):
    r = client.options("/api/recommend", headers={
        "Origin": "http://localhost:5500", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    assert "POST" in r.headers["access-control-allow-methods"]


def test_cors_can_be_restricted_with_an_environment_variable(trained, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://good.example")
    with TestClient(create_app(trained["artifacts"])) as c:
        ok = c.get("/api/health", headers={"Origin": "https://good.example"})
        bad = c.get("/api/health", headers={"Origin": "https://evil.example"})
        assert ok.headers["access-control-allow-origin"] == "https://good.example"
        assert "access-control-allow-origin" not in bad.headers


def test_backend_still_serves_the_static_frontend_files(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


# ---- ML service <-> API integration ---------------------------------------------------------------------------------------------------
def test_api_returns_exactly_what_the_ml_service_computes(client, svc):
    """The API is a thin layer: same inputs -> the same movies, scores and order as the
    underlying HybridRecommender, for every recommendation endpoint."""
    from recsys.hybrid import UserProfile
    ids = popular_ids(client, 9)
    movie_ids = lambda idx: [int(svc.movie_ids[j]) for j in idx]
    api_ids = lambda body: [m["movie_id"] for m in body["items"]]

    # personalised (anonymous user)
    api = client.post("/api/recommend", json={"ratings": rate(ids), "n": 10}).json()
    assert api == RecommendResponse.model_validate(svc.recommend({i: 4.0 for i in ids}, 10, [])).model_dump()
    profile = UserProfile(np.array([svc._id_to_idx[i] for i in ids]), np.full(len(ids), 4.0))
    assert api_ids(api) == movie_ids(svc.hybrid.recommend(profile, n=10).item_idx)

    # similar movies
    api = client.get(f"/api/movies/{ids[0]}/similar", params={"n": 10}).json()
    assert api_ids(api) == movie_ids(svc.hybrid.similar_items(svc._id_to_idx[ids[0]], 10).item_idx)

    # popular
    api = client.get("/api/popular", params={"n": 10}).json()
    assert api_ids(api) == movie_ids(svc.hybrid.popular(10).item_idx)

    # existing user
    api = client.get("/api/users/1/recommend", params={"n": 10}).json()
    idx, ratings = svc._user_hist[1]
    assert api_ids(api) == movie_ids(svc.hybrid.recommend(UserProfile(idx, ratings), n=10).item_idx)
