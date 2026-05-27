"""Hot-storage retrieval for candidate analogs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.neighbors import BallTree

from .distance_utils import EARTH_RADIUS_KM
from .schemas import DBAnalog, LandRequest

REQUIRED_ANALOG_COLUMNS = (
    "deal_id",
    "is_offer",
    "ts_status_ready",
    "lat",
    "lon",
    "comm_sq",
    "price_m2",
    "region",
)


@dataclass(frozen=True)
class AnalogRetrievalConfig:
    """Candidate search settings for hot-storage analog retrieval."""

    bbox_delta_lat: float = 0.02
    bbox_delta_lon: float = 0.02
    radius_km: float = 3.0
    area_tolerance: float = 0.2
    max_age_days: int = 240
    max_candidates: int | None = None
    require_same_region: bool = True


class AnalogCandidateRetriever:
    """Two-step analog retrieval: coarse bbox, then exact BallTree radius.

    The production service used a hot-storage table and a spatial index. This
    public mockup keeps the same idea on top of a Polars DataFrame:
        1. cheap square prefilter by latitude/longitude;
        2. business filters by area, age, and region;
        3. exact haversine radius filter through sklearn BallTree.
    """

    def __init__(
        self,
        analogs_table: pl.DataFrame,
        config: AnalogRetrievalConfig | None = None,
    ) -> None:
        self.analogs_table = analogs_table
        self.config = config or AnalogRetrievalConfig()
        self._validate_columns()

    def retrieve(self, request: LandRequest) -> list[DBAnalog]:
        """Return analogs sorted by exact haversine distance to the request."""
        candidates = self._coarse_prefilter(request)
        if candidates.is_empty():
            return []

        coords_rad = np.radians(candidates.select(["lat", "lon"]).to_numpy())
        query_rad = np.radians([[request.lat, request.lon]])

        tree = BallTree(coords_rad, metric="haversine")
        indices, distances = tree.query_radius(
            query_rad,
            r=self.config.radius_km / EARTH_RADIUS_KM,
            return_distance=True,
            sort_results=True,
        )

        row_indices = indices[0]
        if row_indices.size == 0:
            return []

        distances_km = distances[0] * EARTH_RADIUS_KM
        if self.config.max_candidates is not None:
            limit = self.config.max_candidates
            row_indices = row_indices[:limit]
            distances_km = distances_km[:limit]

        selected = candidates[row_indices.tolist()].with_columns(
            pl.Series("distance_km", distances_km)
        )
        return [
            DBAnalog(**{column: row[column] for column in REQUIRED_ANALOG_COLUMNS})
            for row in selected.iter_rows(named=True)
        ]

    def _coarse_prefilter(self, request: LandRequest) -> pl.DataFrame:
        cfg = self.config
        min_area = request.comm_sq * (1.0 - cfg.area_tolerance)
        max_area = request.comm_sq * (1.0 + cfg.area_tolerance)
        age_days = (pl.lit(request.request_date) - pl.col("ts_status_ready")).dt.total_days()

        predicates = [
            pl.col("lat").is_between(
                request.lat - cfg.bbox_delta_lat,
                request.lat + cfg.bbox_delta_lat,
            ),
            pl.col("lon").is_between(
                request.lon - cfg.bbox_delta_lon,
                request.lon + cfg.bbox_delta_lon,
            ),
            pl.col("comm_sq").is_between(min_area, max_area),
            age_days.is_between(0, cfg.max_age_days),
            pl.col("price_m2") > 0,
        ]
        if cfg.require_same_region:
            predicates.append(pl.col("region") == request.region)

        return self.analogs_table.filter(*predicates)

    def _validate_columns(self) -> None:
        missing = sorted(set(REQUIRED_ANALOG_COLUMNS) - set(self.analogs_table.columns))
        if missing:
            raise ValueError(f"Analogs table is missing required columns: {missing}")


def retrieve_analogs(
    request: LandRequest,
    analogs_table: pl.DataFrame,
    config: AnalogRetrievalConfig | None = None,
) -> list[DBAnalog]:
    """Convenience function for one-off analog retrieval."""
    return AnalogCandidateRetriever(analogs_table, config).retrieve(request)
