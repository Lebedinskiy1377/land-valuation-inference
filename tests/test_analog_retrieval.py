from datetime import datetime, timedelta

import polars as pl
import pytest

from inference import AnalogCandidateRetriever, AnalogRetrievalConfig, LandRequest
from inference.analog_retrieval import retrieve_analogs


def _request() -> LandRequest:
    return LandRequest(
        lat=55.0,
        lon=37.0,
        locality_guid="locality-1",
        comm_sq=1_000.0,
        region="region-a",
        request_date=datetime(2025, 5, 1),
    )


def _analogs_table() -> pl.DataFrame:
    request_date = datetime(2025, 5, 1)
    return pl.DataFrame(
        {
            "deal_id": [
                "near",
                "nearer",
                "in-bbox-outside-radius",
                "outside-bbox",
                "wrong-area",
                "old",
                "other-region",
            ],
            "is_offer": [False, True, False, False, False, False, False],
            "ts_status_ready": [
                request_date - timedelta(days=10),
                request_date - timedelta(days=12),
                request_date - timedelta(days=10),
                request_date - timedelta(days=10),
                request_date - timedelta(days=10),
                request_date - timedelta(days=300),
                request_date - timedelta(days=10),
            ],
            "lat": [55.001, 55.0005, 55.01, 55.03, 55.001, 55.001, 55.001],
            "lon": [37.001, 37.0005, 37.01, 37.03, 37.001, 37.001, 37.001],
            "comm_sq": [1_000.0, 1_050.0, 1_000.0, 1_000.0, 1_500.0, 1_000.0, 1_000.0],
            "price_m2": [2_000.0, 2_050.0, 2_100.0, 2_200.0, 2_300.0, 2_400.0, 2_500.0],
            "region": [
                "region-a",
                "region-a",
                "region-a",
                "region-a",
                "region-a",
                "region-a",
                "region-b",
            ],
        }
    )


def test_retriever_prefilters_by_bbox_and_refines_by_balltree_radius() -> None:
    config = AnalogRetrievalConfig(
        bbox_delta_lat=0.02,
        bbox_delta_lon=0.02,
        radius_km=0.3,
        area_tolerance=0.2,
        max_age_days=240,
    )
    retriever = AnalogCandidateRetriever(_analogs_table(), config)

    analogs = retriever.retrieve(_request())

    assert [analog.deal_id for analog in analogs] == ["nearer", "near"]


def test_retriever_can_limit_candidates_after_distance_sorting() -> None:
    analogs = retrieve_analogs(
        _request(),
        _analogs_table(),
        AnalogRetrievalConfig(radius_km=0.3, max_candidates=1),
    )

    assert [analog.deal_id for analog in analogs] == ["nearer"]


def test_retriever_validates_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        AnalogCandidateRetriever(pl.DataFrame({"lat": [55.0]}))
