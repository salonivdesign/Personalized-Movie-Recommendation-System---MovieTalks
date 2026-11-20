"""Tests that the project does not depend on one specific dataset.

All data is SYNTHETIC. These tests prove the code can be re-pointed at another dataset
of the same format (different ids, scale, size, missing optional files, no metadata),
retrained, and that bad input fails with clear messages. They say nothing about
recommendation quality on real MovieLens."""
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import config as cfg
from make_fixture import make_fixture
from recsys import data as D
from recsys.pipeline import run_training
from recsys.service import RecommenderService

ROOT = Path(__file__).resolve().parents[1]


# ---- configuration is centralised ---------------------------------------------------
def test_data_dir_can_be_overridden_by_environment(tmp_path):
    code = "import config; print(config.DEFAULT_DATA_DIR)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "RECSYS_DATA_DIR": str(tmp_path)}, check=True)
    assert out.stdout.strip() == str(tmp_path)


def test_api_schema_uses_the_configured_rating_scale():
    from pydantic import ValidationError

    from api.schemas import RatingIn
    RatingIn(movie_id=1, rating=cfg.RATING_MIN)
    RatingIn(movie_id=1, rating=cfg.RATING_MAX)
    for bad in (cfg.RATING_MAX + cfg.RATING_STEP, cfg.RATING_MIN - cfg.RATING_STEP,
                cfg.RATING_MIN + cfg.RATING_STEP / 3):
        with pytest.raises(ValidationError):
            RatingIn(movie_id=1, rating=bad)


# ---- input validation --------------------------------------------------------------------
def _write(path: Path, text: str):
    path.write_text(text)


def test_load_raw_reports_missing_files_and_columns(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        D.load_raw(tmp_path)
    _write(tmp_path / "movies.csv", "movieId,title,genres\n1,A (2000),Drama\n")
    _write(tmp_path / "ratings.csv", "userId,movieId,stars,timestamp\n1,1,4,1\n")
    with pytest.raises(ValueError, match="rating"):
        D.load_raw(tmp_path)
    _write(tmp_path / "ratings.csv", "")
    with pytest.raises(D.DataError, match="ratings.csv"):
        D.load_raw(tmp_path)


def test_unusable_optional_files_are_ignored_with_a_warning(tmp_path):
    _write(tmp_path / "movies.csv", "movieId,title,genres\n1,A (2000),Drama\n")
    _write(tmp_path / "ratings.csv", "userId,movieId,rating,timestamp\n1,1,4,1\n")
    _write(tmp_path / "tags.csv", "userId,movieId,comment\n1,1,hi\n")      # no 'tag' column
    _write(tmp_path / "links.csv", "movieId,imdbId,tmdbId\n1,5,6\n")       # valid
    with pytest.warns(UserWarning, match="tags.csv ignored"):
        _, _, tags, links = D.load_raw(tmp_path)
    assert tags.empty and list(links["imdbId"]) == [5]


def test_cleaning_coerces_junk_and_rejects_empty_results():
    movies = pd.DataFrame({"movieId": [1, 2], "title": ["A (2000)", "B (2001)"], "genres": ["x", "y"]})
    ratings = pd.DataFrame({"userId": [1, 1, 2, 2], "movieId": [1, 2, 1, 2],
                            "rating": [4.0, "abc", 3.5, 7.0], "timestamp": [1, 2, 3, 4]})
    clean, report = D.clean_ratings(ratings, movies["movieId"])
    assert len(clean) == 2 and report["dropped_null"] == 1 and report["dropped_out_of_range"] == 1
    with pytest.raises(D.DataError, match="No valid ratings"):
        D.clean_ratings(ratings.assign(rating=99.0), movies["movieId"])


# ---- fingerprints and the synthetic guard -----------------------------------------------------
def test_fingerprint_records_files_and_detects_synthetic_marker(trained, tmp_path):
    fp = D.dataset_fingerprint(trained["data_dir"])
    assert fp["synthetic"] is True and fp["name"] == trained["data_dir"].name
    for name in ("ratings.csv", "movies.csv", "tags.csv", "links.csv"):
        entry = fp["files"][name]
        assert entry["bytes"] == (trained["data_dir"] / name).stat().st_size
        assert len(entry["sha256"]) == 64
    assert D.SYNTHETIC_MARKER == "SYNTHETIC_DATA.txt"
    (tmp_path / "ratings.csv").write_text("x")
    assert D.dataset_fingerprint(tmp_path)["synthetic"] is False


def test_artifacts_from_synthetic_data_are_labelled(trained):
    meta = json.loads((trained["artifacts"] / cfg.METRICS_FILENAME).read_text())["meta"]
    assert meta["dataset"]["synthetic"] is True and meta["rating_scale"] == [0.5, 5.0, 0.5]
    assert "SYNTHETIC TEST DATA" in (trained["artifacts"] / "RESULTS.md").read_text()


def _run_update_readme(monkeypatch, artifacts: Path, root: Path, *args):
    sys.path.insert(0, str(ROOT / "scripts"))
    import update_readme
    monkeypatch.setattr(cfg, "ARTIFACT_DIR", artifacts)
    monkeypatch.setattr(cfg, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["update_readme.py", *args])
    update_readme.main()


def test_update_readme_refuses_synthetic_results(trained, tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    original = "<!-- RESULTS:START -->\nplaceholder\n<!-- RESULTS:END -->\n"
    readme.write_text(original)
    with pytest.raises(SystemExit, match="SYNTHETIC"):
        _run_update_readme(monkeypatch, trained["artifacts"], tmp_path)
    assert readme.read_text() == original


def test_update_readme_guard_logic_for_non_synthetic_artifacts(trained, tmp_path, monkeypatch):
    """Tests the GUARD only: metrics.json is edited to look non-synthetic. No real data involved."""
    art = tmp_path / "art"
    art.mkdir()
    (art / "RESULTS.md").write_text("RESULT-TABLES\n")
    report = json.loads((trained["artifacts"] / cfg.METRICS_FILENAME).read_text())
    report["meta"]["dataset"].update(synthetic=False, name="some-other-dataset")
    (art / cfg.METRICS_FILENAME).write_text(json.dumps(report))
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n")

    with pytest.raises(SystemExit, match="allow-other-dataset"):          # wrong dataset name
        _run_update_readme(monkeypatch, art, tmp_path)
    _run_update_readme(monkeypatch, art, tmp_path, "--allow-other-dataset")
    assert "RESULT-TABLES" in readme.read_text() and "old" not in readme.read_text()

    report["meta"]["dataset"]["name"] = cfg.DATASET_NAME                  # expected name
    (art / cfg.METRICS_FILENAME).write_text(json.dumps(report))
    readme.write_text("<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n")
    _run_update_readme(monkeypatch, art, tmp_path)
    assert "RESULT-TABLES" in readme.read_text()


# ---- retraining on different datasets ------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset_b(tmp_path_factory):
    """Different shape: other ids, integer 1-5 ratings, fewer users/movies, no tags/links."""
    d = make_fixture(tmp_path_factory.mktemp("b") / "other-dataset", n_users=70, n_movies=160, seed=3)
    r = pd.read_csv(d / "ratings.csv")
    r["rating"] = r["rating"].clip(1, 5).round()
    r["userId"] += 5000
    r["movieId"] += 10_000
    r.to_csv(d / "ratings.csv", index=False)
    m = pd.read_csv(d / "movies.csv")
    m["movieId"] += 10_000
    m.to_csv(d / "movies.csv", index=False)
    (d / "tags.csv").unlink()
    (d / "links.csv").unlink()
    return d


def test_retraining_on_another_dataset_replaces_artifacts(trained, dataset_b, tmp_path):
    art = tmp_path / "art"
    run_training(trained["data_dir"], art, tune_cf=False, seed=7)
    first = joblib.load(art / cfg.BUNDLE_FILENAME)["meta"]
    run_training(dataset_b, art, tune_cf=False, seed=7)                   # same folder, new data
    second = joblib.load(art / cfg.BUNDLE_FILENAME)["meta"]
    assert first["n_movies"] == 300 and second["n_movies"] == 160
    assert first["dataset"]["files"]["ratings.csv"]["sha256"] != second["dataset"]["files"]["ratings.csv"]["sha256"]
    assert "tags.csv" not in second["dataset"]["files"]
    svc = RecommenderService.load(art)
    assert svc.health()["n_movies"] == 160
    assert svc.popular(5)["items"] and min(svc.movie_ids) > 10_000


def test_rating_scale_change_flows_through_the_pipeline(dataset_b, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RATING_MIN", 1.0)
    monkeypatch.setattr(cfg, "RATING_MAX", 5.0)
    monkeypatch.setattr(cfg, "RATING_STEP", 1.0)
    run_training(dataset_b, tmp_path, tune_cf=False, seed=3)
    bundle = joblib.load(tmp_path / cfg.BUNDLE_FILENAME)
    assert bundle["meta"]["rating_scale"] == [1.0, 5.0, 1.0]
    cf = bundle["hybrid"].cf
    users = np.tile(cf.user_ids_[:5], 30)
    items = np.repeat(np.arange(30), 5)
    pred = cf.predict_pairs(users, items)
    assert pred.min() >= 1.0 and pred.max() <= 5.0
    ratings = pd.DataFrame({"userId": [1, 1], "movieId": [1, 1 + 1], "rating": [0.5, 3.0], "timestamp": [1, 2]})
    clean, report = D.clean_ratings(ratings, [1, 2])
    assert list(clean["rating"]) == [3.0] and report["dropped_out_of_range"] == 1


def test_dataset_without_any_item_metadata_still_trains_and_serves(tmp_path):
    d = make_fixture(tmp_path / "no-meta", n_users=60, n_movies=120, seed=5)
    m = pd.read_csv(d / "movies.csv")
    m["title"] = [f"Plain title {i}" for i in m["movieId"]]            # no (YYYY)
    m["genres"] = cfg.NO_GENRES_LABEL
    m.to_csv(d / "movies.csv", index=False)
    (d / "tags.csv").unlink()
    (d / "links.csv").unlink()
    art = tmp_path / "art"
    run_training(d, art, tune_cf=False, seed=5)
    bundle = joblib.load(art / cfg.BUNDLE_FILENAME)
    assert bundle["hybrid"].content.vectorizer_ is None               # content has no signal
    svc = RecommenderService.load(art)
    assert svc.genres() == []
    ids = [m["movie_id"] for m in svc.popular(8)["items"]]
    rec = svc.recommend({i: 5.0 for i in ids[:6]}, n=5)
    assert rec["mode"] == "hybrid" and len(rec["items"]) == 5
    sim = svc.similar(ids[0], n=5)                                     # CF-only similarity
    assert len(sim["items"]) == 5 and set(i["reason_code"] for i in sim["items"]) == {"cf"}


def test_too_little_data_to_split_skips_evaluation_but_still_trains(tmp_path):
    d = make_fixture(tmp_path / "tiny-users", n_users=40, n_movies=80, seed=9, ratings_per_user=(3, 4))
    report = run_training(d, tmp_path / "art", tune_cf=True, seed=9)
    assert report["split"]["val"] == 0 and report["split"]["test"] == 0
    assert "ranking" not in report and "rating_prediction" not in report
    assert (tmp_path / "art" / cfg.BUNDLE_FILENAME).exists()


# ---- safe artifact writes ----------------------------------------------------------------------------
def test_failed_save_keeps_previous_artifacts_intact(trained, tmp_path, monkeypatch):
    art = tmp_path / "art"
    run_training(trained["data_dir"], art, tune_cf=False, seed=7)
    before = (art / cfg.BUNDLE_FILENAME).read_bytes()

    def broken_dump(obj, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(joblib, "dump", broken_dump)
    with pytest.raises(RuntimeError, match="disk full"):
        run_training(trained["data_dir"], art, tune_cf=False, seed=7)
    assert (art / cfg.BUNDLE_FILENAME).read_bytes() == before
    assert not list(art.glob("*.tmp"))


# ---- command-line recommender (no web server) -----------------------------------------
def test_cli_recommend_prints_top_n_and_flags_synthetic(trained):
    def run(*args):
        return subprocess.run([sys.executable, "scripts/recommend.py", "--artifact-dir",
                               str(trained["artifacts"]), *args], cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8")
    out = run("--user", "1", "--n", "5")
    assert out.returncode == 0, out.stderr
    assert "SYNTHETIC" in out.stdout and sum(l[:2].strip().rstrip(".").isdigit() for l in out.stdout.splitlines()) == 5
    bad = run("--user", "99999999")
    assert bad.returncode == 1 and "not found" in bad.stderr
