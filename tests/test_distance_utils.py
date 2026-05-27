import numpy as np
import pytest

from inference.distance_utils import haversine, min_distance_to_cities


def test_haversine_is_vectorized_and_zero_for_same_point() -> None:
    distances = haversine(
        55.7558,
        37.6173,
        np.array([55.7558, 59.9386]),
        np.array([37.6173, 30.3141]),
    )

    assert distances[0] == pytest.approx(0.0)
    assert 630.0 < distances[1] < 640.0


def test_min_distance_to_cities_returns_nan_for_empty_reference() -> None:
    distance = min_distance_to_cities(
        55.7558,
        37.6173,
        {"lat": np.array([]), "lon": np.array([])},
    )

    assert np.isnan(distance)
