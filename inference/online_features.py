"""Признаки, считающиеся в момент запроса (не precomputed)."""

from __future__ import annotations

import numpy as np
import polars as pl

from .distance_utils import haversine
from .schemas import DBAnalog, LandRequest


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

    # dist — расстояние до ближайшего аналога
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

    # Расстояния до городов разного размера
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

    # locality_lon: координата центра locality из 2GIS reference
    locality_match = locality_reference.filter(
        pl.col("locality_guid") == request.locality_guid
    )
    locality_lon = (
        float(locality_match["lon"][0]) if locality_match.height > 0 else float("nan")
    )

    # Агрегаты по аналогам отдельно для сделок и офферов
    deals = analogs_df.filter(~pl.col("is_offer"))
    offers = analogs_df.filter(pl.col("is_offer"))

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
            "mean_analogs_deal": [
                float(deals["price_m2"].mean()) if deals.height > 0 else float("nan")
            ],
            "median_analogs_deal": [
                float(deals["price_m2"].median()) if deals.height > 0 else float("nan")
            ],
            "mean_analogs_offer": [
                float(offers["price_m2"].mean()) if offers.height > 0 else float("nan")
            ],
            "median_analogs_offer": [
                (
                    float(offers["price_m2"].median())
                    if offers.height > 0
                    else float("nan")
                )
            ],
            "price_m2_pred": [price_m2_pred],
        }
    )
