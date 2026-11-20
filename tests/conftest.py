import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_fixture import make_fixture  # noqa: E402
from recsys.pipeline import run_training  # noqa: E402


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once on the synthetic fixture; reused by all tests."""
    root = tmp_path_factory.mktemp("recsys")
    data_dir = make_fixture(root / "data")
    art = root / "artifacts"
    report = run_training(data_dir, art, tune_cf=False, seed=7)
    return {"data_dir": data_dir, "artifacts": art, "report": report}


@pytest.fixture(scope="session")
def ml_ctx(trained):
    """Cleaned SYNTHETIC data + models fit on TRAIN only, for unit-testing ML internals."""
    from types import SimpleNamespace

    import config as cfg
    from recsys import data as D
    from recsys.hybrid import HybridRecommender
    from recsys.pipeline import fit_models

    raw_r, raw_m, tags, links = D.load_raw(trained["data_dir"])
    movies = D.clean_movies(raw_m, links)
    ratings, _ = D.clean_ratings(raw_r, movies["movieId"])
    ratings = D.add_item_index(ratings, movies["movieId"].to_numpy())
    train, val, test = D.temporal_user_split(
        ratings, cfg.VAL_FRACTION, cfg.TEST_FRACTION, cfg.MIN_RATINGS_FOR_SPLIT)
    pop, content, cf = fit_models(train, movies, tags, cfg.CF_PARAMS, seed=7)
    hybrid = HybridRecommender(movies, pop, content, cf)
    return SimpleNamespace(movies=movies, tags=tags, ratings=ratings, train=train, val=val,
                           test=test, pop=pop, content=content, cf=cf, hybrid=hybrid,
                           n_items=len(movies))
