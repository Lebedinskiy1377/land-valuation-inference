from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from inference import DBAnalog, LandRequest
from inference.online_features import compute_online_features
from inference.pairwise_features import build_pairwise_features


def _request() -> LandRequest:
    return LandRequest(
        lat=55.0,
        lon=37.0,
        locality_guid="locality-1",
        comm_sq=1_000.0,
        region="region-a",
        request_date=datetime(2025, 5, 1),
    )


def _analog(
    deal_id: str,
    *,
    is_offer: bool,
    lat: float,
    lon: float,
    comm_sq: float,
    price_m2: float,
    region: str = "region-a",
) -> DBAnalog:
    return DBAnalog(
        deal_id=deal_id,
        is_offer=is_offer,
        ts_status_ready=datetime(2025, 5, 1) - timedelta(days=10),
        lat=lat,
        lon=lon,
        comm_sq=comm_sq,
        price_m2=price_m2,
        region=region,
    )


def test_build_pairwise_features_adds_geo_and_similarity_columns() -> None:
    analogs = [
        _analog(
            "a1",
            is_offer=False,
            lat=55.001,
            lon=37.001,
            comm_sq=1_100.0,
            price_m2=2_000.0,
        ),
        _analog(
            "a2",
            is_offer=True,
            lat=56.0,
            lon=38.0,
            comm_sq=900.0,
            price_m2=2_500.0,
            region="region-b",
        ),
    ]

    features = build_pairwise_features(_request(), analogs)

    assert features.height == 2
    assert features["area_ratio"].to_list() == pytest.approx([1.1, 0.9])
    assert features["region_match"].to_list() == [1, 0]
    assert features["analog_age_days"].to_list() == [10, 10]
    assert features["distance_km"][0] < features["distance_km"][1]


def test_compute_online_features_handles_deal_offer_aggregates() -> None:
    analogs = [
        _analog(
            "deal",
            is_offer=False,
            lat=55.001,
            lon=37.001,
            comm_sq=1_000.0,
            price_m2=2_000.0,
        ),
        _analog(
            "offer",
            is_offer=True,
            lat=55.002,
            lon=37.002,
            comm_sq=1_050.0,
            price_m2=3_000.0,
        ),
    ]
    cities_reference = pl.DataFrame(
        {
            "name": ["huge-city", "big-city", "middle-city", "small-city"],
            "lat": [55.0, 55.2, 55.4, 55.6],
            "lon": [37.0, 37.2, 37.4, 37.6],
            "size_category": ["huge", "big", "middle", "small"],
        }
    )
    locality_reference = pl.DataFrame(
        {"locality_guid": ["locality-1"], "lon": [37.123]}
    )

    features = compute_online_features(
        request=_request(),
        top_analogs=analogs,
        cities_reference=cities_reference,
        locality_reference=locality_reference,
    )

    row = features.row(0, named=True)
    assert row["mean_analogs_deal"] == pytest.approx(2_000.0)
    assert row["median_analogs_offer"] == pytest.approx(3_000.0)
    assert row["locality_lon"] == pytest.approx(37.123)
    assert row["dist_to_huge_city"] == pytest.approx(0.0)
    assert row["dist"] > 0.0


def test_compute_online_features_handles_empty_analogs() -> None:
    features = compute_online_features(
        request=_request(),
        top_analogs=[],
        cities_reference=pl.DataFrame(
            {
                "name": [],
                "lat": [],
                "lon": [],
                "size_category": [],
            }
        ),
        locality_reference=pl.DataFrame({"locality_guid": [], "lon": []}),
    )

    row = features.row(0, named=True)
    assert np.isnan(row["dist"])
    assert np.isnan(row["mean_analogs_deal"])
    assert np.isnan(row["median_analogs_offer"])
