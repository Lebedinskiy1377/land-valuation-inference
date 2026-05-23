"""Утилиты для гео-расчётов."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Векторизованное расстояние haversine между точками в км."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2) - np.radians(lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def min_distance_to_cities(
    target_lat: float,
    target_lon: float,
    cities: dict[str, np.ndarray],
) -> float:
    """Минимальное расстояние от точки до набора городов в км.

    cities: словарь с ключами {"lat": np.ndarray, "lon": np.ndarray}.
    """
    distances = haversine(target_lat, target_lon, cities["lat"], cities["lon"])
    return float(np.min(distances)) if distances.size > 0 else float("nan")
