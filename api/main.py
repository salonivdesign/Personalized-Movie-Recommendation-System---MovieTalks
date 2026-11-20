"""FastAPI application. Thin HTTP layer over recsys.service.RecommenderService.

* Models are loaded once at startup from saved artifacts (training is a separate
  step: ``python scripts/train.py``).
* The same server also serves the static frontend at "/".

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config as cfg
from api.schemas import (ErrorOut, HealthOut, MovieListOut, MovieOut, RecommendRequest,
                         RecommendResponse, SimilarResponse, SortOption, ValidationErrorOut)
from recsys.service import ArtifactError, RecommenderService, ServiceError

log = logging.getLogger("api")

NO_MODEL_MESSAGE = "Model artifacts not found. Run: python scripts/train.py"

GENRE_QUERY = Query(default=None, max_length=300,
                    description="Comma-separated genres, e.g. Comedy,Drama")


def parse_genres(raw: str | None) -> list[str]:
    return [g.strip() for g in raw.split(",") if g.strip()][:10] if raw else []


def get_service(request: Request) -> RecommenderService:
    """FastAPI dependency: the loaded model service, or a clear 503 if untrained."""
    svc = request.app.state.service
    if svc is None:
        raise HTTPException(503, request.app.state.load_error or NO_MODEL_MESSAGE)
    return svc


Svc = Annotated[RecommenderService, Depends(get_service)]


def create_app(artifact_dir: Path | str | None = None) -> FastAPI:
    artifact_dir = Path(artifact_dir) if artifact_dir else cfg.ARTIFACT_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Whatever goes wrong while loading, the server still starts: /api/health then
        # reports "no_model" and model endpoints answer 503 with a safe, clear message.
        try:
            app.state.service = RecommenderService.load(artifact_dir)
            app.state.load_error = None
            log.info("Model artifacts loaded from %s", artifact_dir)
        except FileNotFoundError as exc:
            app.state.service, app.state.load_error = None, NO_MODEL_MESSAGE
            log.warning("%s", exc)
        except ArtifactError as exc:                 # message is safe to show clients
            app.state.service, app.state.load_error = None, str(exc)
            log.error("Model artifacts unusable: %s", exc, exc_info=exc.__cause__ is not None)
        except Exception:                            # never leak internals to clients
            app.state.service = None
            app.state.load_error = "Model artifacts could not be loaded. See the server log."
            log.exception("Unexpected error while loading model artifacts")
        yield

    app = FastAPI(
        title="Movie Recommender API", version="1.0.0", lifespan=lifespan,
        description="Hybrid movie recommender (popularity + content-based + collaborative "
                    "filtering). Train first with `python scripts/train.py`; the server only "
                    "loads the saved model. Every error response has a `detail` message.",
        openapi_tags=[
            {"name": "meta", "description": "Health check and offline evaluation results."},
            {"name": "catalog", "description": "Browse, list and search movies."},
            {"name": "recommend", "description": "Popular, similar and personalised recommendations."}],
        responses={422: {"model": ValidationErrorOut, "description": "Invalid input"},
                   500: {"model": ErrorOut, "description": "Unexpected server error"},
                   503: {"model": ErrorOut, "description": "Model not available"}})
    app.state.service, app.state.load_error = None, NO_MODEL_MESSAGE   # until lifespan runs
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        """Same shape as every other error: a string `detail`, plus per-field `errors`."""
        errors = [{"field": ".".join(str(p) for p in e["loc"]), "message": str(e["msg"])}
                  for e in exc.errors()]
        summary = "; ".join(f"{e['field']}: {e['message']}" for e in errors[:3])
        return JSONResponse(status_code=422,
                            content={"detail": summary or "Invalid request", "errors": errors})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ---- endpoints ---------------------------------------------------------------
    @app.get("/api/health", response_model=HealthOut, response_model_exclude_none=True,
             tags=["meta"])
    def health(request: Request):
        """Is the server up, and is a trained model loaded?

        Always answers 200 so a hosting platform can tell "server alive" from "server down";
        read `status` ("ok" or "no_model") for model availability. Also reports the dataset
        name, whether it is synthetic test data, and the rating scale the API accepts.
        """
        svc = request.app.state.service
        if svc is None:
            return {"status": "no_model", "detail": request.app.state.load_error or NO_MODEL_MESSAGE}
        return svc.health()

    @app.get("/api/genres", tags=["catalog"])
    def genres(svc: Svc) -> list[str]:
        """All genre names (used by the preference chips)."""
        return svc.genres()

    @app.get("/api/movies", response_model=MovieListOut, tags=["catalog"])
    def list_movies(svc: Svc,
                    limit: Annotated[int, Query(ge=1, le=100)] = 20,
                    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
                    genre: Annotated[str | None, Query(max_length=30)] = None,
                    sort: SortOption = "popular",
                    min_ratings: Annotated[int, Query(ge=0, le=1_000_000)] = 0):
        """Browse the catalog page by page.

        `sort`: popular (most ratings), top_rated (Bayesian weighted rating), year
        (newest first) or title (A-Z). `total` is the number of movies matching the
        filters, so a client can build pagination. An offset past the end returns no items.
        """
        return svc.list_movies(limit, offset, genre, sort, min_ratings)

    @app.get("/api/movies/search", response_model=list[MovieOut], tags=["catalog"])
    def search(svc: Svc,
               q: Annotated[str, Query(min_length=1, max_length=80)],
               limit: Annotated[int, Query(ge=1, le=30)] = 12,
               genre: Annotated[str | None, Query(max_length=30)] = None):
        """Title search: every word in ``q`` must appear; prefix matches and
        popular movies rank first."""
        return svc.search(q, limit, genre)

    @app.get("/api/movies/{movie_id}", response_model=MovieOut, tags=["catalog"])
    def movie(svc: Svc, movie_id: Annotated[int, PathParam(gt=0)]):
        return svc.get_movie(movie_id)

    @app.get("/api/movies/{movie_id}/similar", response_model=SimilarResponse, tags=["recommend"])
    def similar(svc: Svc, movie_id: Annotated[int, PathParam(gt=0)],
                n: Annotated[int, Query(ge=1, le=50)] = 12, genres: str | None = GENRE_QUERY):
        """Item-to-item: content cosine similarity blended with CF latent similarity."""
        return svc.similar(movie_id, n, parse_genres(genres))

    @app.get("/api/popular", response_model=RecommendResponse, tags=["recommend"])
    def popular(svc: Svc, n: Annotated[int, Query(ge=1, le=50)] = 20,
                genres: str | None = GENRE_QUERY):
        """Cold-start list: Bayesian weighted rating, optionally filtered by genre."""
        return svc.popular(n, parse_genres(genres))

    @app.post("/api/recommend", response_model=RecommendResponse, tags=["recommend"])
    def recommend(svc: Svc, body: RecommendRequest):
        """Personalised recommendations for an anonymous user.

        The client sends the user's ratings; the server holds no user state.
        Routing: 0 ratings -> popularity, 1-4 -> content+popularity, >=5 -> hybrid.
        """
        ratings = {r.movie_id: r.rating for r in body.ratings}
        return svc.recommend(ratings, body.n, body.genres)

    @app.get("/api/recommend", include_in_schema=False)
    def recommend_wrong_method():
        """Without this, the frontend's catch-all file mount would answer GET with a
        confusing 404 instead of "wrong method"."""
        raise HTTPException(405, "Use POST with a JSON body, e.g. {\"ratings\": [...]}",
                            headers={"Allow": "POST"})

    @app.get("/api/users/{user_id}/recommend", response_model=RecommendResponse, tags=["recommend"])
    def recommend_for_user(svc: Svc, user_id: Annotated[int, PathParam(gt=0)],
                           n: Annotated[int, Query(ge=1, le=50)] = 12,
                           genres: str | None = GENRE_QUERY):
        """Recommendations for an existing MovieLens user id (demo)."""
        return svc.recommend_for_user(user_id, n, parse_genres(genres))

    @app.get("/api/metrics", tags=["meta"])
    def metrics(svc: Svc):
        """Offline evaluation results written by scripts/train.py."""
        return svc.get_metrics()

    # Frontend last, so /api/* routes win.
    if cfg.FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=cfg.FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
