"""Построение признаков пары (target, analog) для фильтрации и ранжирования."""

from __future__ import annotations

import polars as pl

from .distance_utils import haversine
from .schemas import DBAnalog, LandRequest


def build_pairwise_features(
    request: LandRequest, analogs: list[DBAnalog]
) -> pl.DataFrame:
    """Собирает фичи для каждой пары (запрашиваемый объект, аналог).

    Используется на входе AnalogFilterModel и AnalogCorrectionModel.
    Точный набор фичей зависит от того, как обучались бустинги —
    адаптируй колонки под свой обученный артефакт.
    """
    if not analogs:
        return pl.DataFrame()

    df = pl.DataFrame(
        {
            "target_lat": [request.lat] * len(analogs),
            "target_lon": [request.lon] * len(analogs),
            "target_comm_sq": [request.comm_sq] * len(analogs),
            "target_region": [request.region] * len(analogs),
            "analog_lat": [a.lat for a in analogs],
            "analog_lon": [a.lon for a in analogs],
            "analog_comm_sq": [a.comm_sq for a in analogs],
            "analog_price_m2": [a.price_m2 for a in analogs],
            "analog_region": [a.region for a in analogs],
            "analog_is_offer": [int(a.is_offer) for a in analogs],
            "analog_age_days": [
                (request.request_date - a.ts_status_ready).days for a in analogs
            ],
        }
    )

    distances = haversine(
        df["target_lat"].to_numpy(),
        df["target_lon"].to_numpy(),
        df["analog_lat"].to_numpy(),
        df["analog_lon"].to_numpy(),
    )

    return df.with_columns(
        pl.Series("distance_km", distances),
        (pl.col("analog_comm_sq") / pl.col("target_comm_sq")).alias("area_ratio"),
        (pl.col("analog_region") == pl.col("target_region"))
        .cast(pl.Int8)
        .alias("region_match"),
    )
