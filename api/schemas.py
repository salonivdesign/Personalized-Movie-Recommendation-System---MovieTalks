"""Pydantic models: request validation and response shapes (also drive /docs)."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config as cfg

GENRE_RE = re.compile(r"^[A-Za-z0-9\-' ()]{1,30}$")
SortOption = Literal["popular", "top_rated", "year", "title"]


def _reject_bool(v):
    """JSON true/false would otherwise be silently accepted as 1/0."""
    if isinstance(v, bool):
        raise ValueError("must be a number, not true/false")
    return v


def _validate_genres(values: list[str]) -> list[str]:
    for g in values:
        if not GENRE_RE.match(g):
            raise ValueError(f"Invalid genre name: {g!r}")
    return values


class RatingIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # a typo like "stars" is an error, not ignored

    movie_id: int = Field(gt=0, description="MovieLens movieId (find ids with /api/movies/search)")
    # Scale comes from config.py so a dataset with another scale changes in one place.
    rating: float = Field(ge=cfg.RATING_MIN, le=cfg.RATING_MAX,
                          description=f"Rating from {cfg.RATING_MIN} to {cfg.RATING_MAX}")

    @field_validator("movie_id", "rating", mode="before")
    @classmethod
    def not_bool(cls, v):
        return _reject_bool(v)

    @field_validator("rating")
    @classmethod
    def on_scale(cls, v: float) -> float:
        steps = (v - cfg.RATING_MIN) / cfg.RATING_STEP
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"rating must be {cfg.RATING_MIN} plus a multiple of {cfg.RATING_STEP}")
        return v


class RecommendRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{
            "ratings": [{"movie_id": 1, "rating": 4.5}, {"movie_id": 2, "rating": 3.0}],
            "genres": ["Comedy"], "n": 12}]})

    ratings: list[RatingIn] = Field(default_factory=list, max_length=500,
                                    description="The user's ratings. Empty = popular movies.")
    genres: list[str] = Field(default_factory=list, max_length=10,
                              description="Only recommend movies from these genres.")
    n: int = Field(12, ge=1, le=50, description="How many recommendations to return.")

    @field_validator("n", mode="before")
    @classmethod
    def n_not_bool(cls, v):
        return _reject_bool(v)

    @field_validator("genres")
    @classmethod
    def genres_ok(cls, v):
        return _validate_genres(v)

    @model_validator(mode="after")
    def unique_movies(self):
        ids = [r.movie_id for r in self.ratings]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate movie_id in ratings")
        return self


class MovieOut(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    avg_rating: float | None
    n_ratings: int
    imdb_url: str | None


class RecommendationOut(MovieOut):
    reason_code: str
    reason: str
    because: str | None
    predicted_rating: float | None
    score: float


class RecommendResponse(BaseModel):
    mode: str
    mode_description: str
    n_ratings_used: int = 0
    ratings_needed_for_cf: int = 0
    items: list[RecommendationOut]


class SimilarResponse(BaseModel):
    mode: str
    mode_description: str
    movie: MovieOut
    items: list[RecommendationOut]


class MovieListOut(BaseModel):
    total: int
    limit: int
    offset: int
    sort: SortOption
    items: list[MovieOut]


class RatingScaleOut(BaseModel):
    min: float
    max: float
    step: float


class HealthOut(BaseModel):
    status: str                      # "ok" or "no_model"
    detail: str | None = None        # why the model is unavailable
    n_movies: int | None = None
    n_users: int | None = None
    n_ratings: int | None = None
    trained_at_utc: str | None = None
    dataset_name: str | None = None
    synthetic_data: bool | None = None   # True when trained on test data, not MovieLens
    rating_scale: RatingScaleOut | None = None
    relevance_threshold: float | None = None


class ErrorOut(BaseModel):
    """Shape of every error response: a human-readable message."""
    detail: str


class FieldErrorOut(BaseModel):
    field: str
    message: str


class ValidationErrorOut(ErrorOut):
    errors: list[FieldErrorOut]
