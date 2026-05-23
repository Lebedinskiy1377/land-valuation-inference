"""Признаки, считающиеся в момент запроса (не precomputed)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from .distance_utils import haversine
from .schemas import DBAnalog, LandRequest


def _safe_numeric_agg(series: pl.Series, agg: Literal["mean", "median"]) -> float:
    """Безопасный агрегат с приведением к float.

    Возвращает NaN, если series пустой или агрегат вернул None
    (теоретически возможно для пустых/all-null серий).
    """
    if series.is_empty():
        return float("nan")
    value = getattr(series, agg)()
    if value is None:
        return float("nan")
    return float(value)  # type: ignore[arg-type]


def compute_online_features(
    request: LandRequest,
    top_analogs: list[DBAnalog],
    cities_reference: pl.DataFrame,
    locality_reference: pl.DataFrame,
    price_m2_pred: float,
) -> pl.DataFrame:
    """Строит DataFrame признаков для MainPriceModel и ConfidenceModel.

    Args:
        request: запрос на оценку.
        top_analogs: топ-N финальных аналогов после фильтрации и ранжирования.
        cities_reference: DataFrame со столбцами
            (name, lat, lon, size_category), где size_category в
            {"huge", "big", "middle", "small"}.
        locality_reference: DataFrame со столбцами (locality_guid, lon).
        price_m2_pred: внешний предикт (фича из соседней модели).

    Returns:
        Polars DataFrame с одной строкой и колонками, перечисленными
        в порядке feature importance основной модели.
    """
    analogs_df = pl.DataFrame(
        {
            "is_offer": [a.is_offer for a in top_analogs],
            "price_m2": [a.price_m2 for a in top_analogs],
            "lat": [a.lat for a in top_analogs],
            "lon": [a.lon for a in top_analogs],
        }
    )

    if analogs_df.height > 0:
        distances_to_analogs = haversine(
            request.lat,
            request.lon,
            analogs_df["lat"].to_numpy(),
            analogs_df["lon"].to_numpy(),
        )
        dist = float(np.min(distances_to_analogs))
    else:
        dist = float("nan")

    city_distances: dict[str, float] = {}
    for size in ("huge", "big", "middle", "small"):
        subset = cities_reference.filter(pl.col("size_category") == size)
        if subset.height == 0:
            city_distances[size] = float("nan")
            continue
        dists = haversine(
            request.lat,
            request.lon,
            subset["lat"].to_numpy(),
            subset["lon"].to_numpy(),
        )
        city_distances[size] = float(np.min(dists))

    locality_match = locality_reference.filter(
        pl.col("locality_guid") == request.locality_guid
    )
    locality_lon = (
        float(locality_match["lon"][0]) if locality_match.height > 0 else float("nan")
    )

    deals = analogs_df.filter(~pl.col("is_offer"))
    offers = analogs_df.filter(pl.col("is_offer"))

    deals_prices = deals["price_m2"]
    offers_prices = offers["price_m2"]

    return pl.DataFrame(
        {
            "lat": [request.lat],
            "comm_sq": [request.comm_sq],
            "region": [request.region],
            "dist": [dist],
            "dist_to_huge_city": [city_distances["huge"]],
            "dist_to_big_city": [city_distances["big"]],
            "dist_to_middle_city": [city_distances["middle"]],
            "dist_to_small_city": [city_distances["small"]],
            "locality_lon": [locality_lon],
            "mean_analogs_deal": [_safe_numeric_agg(deals_prices, "mean")],
            "median_analogs_deal": [_safe_numeric_agg(deals_prices, "median")],
            "mean_analogs_offer": [_safe_numeric_agg(offers_prices, "mean")],
            "median_analogs_offer": [_safe_numeric_agg(offers_prices, "median")],
            "price_m2_pred": [price_m2_pred],
        }
    )
