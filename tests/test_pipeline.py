from datetime import datetime

import numpy as np
import polars as pl
import pytest

from inference import DBAnalog, LandRequest, LandValuationPipeline, ValuationResponse


class ArrayModel:
    def __init__(self, values: list[float]) -> None:
        self.values = np.array(values, dtype=float)

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        return self.values[: features.height]


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        return np.full(features.height, self.value, dtype=float)


class ConstantCorrector:
    def __init__(self, correction: float) -> None:
        self.correction = correction

    def correct(self, lat: float, lon: float) -> float:
        return self.correction


def _request() -> LandRequest:
    return LandRequest(
        lat=55.0,
        lon=37.0,
        locality_guid="locality-1",
        comm_sq=1_000.0,
        region="region-a",
        request_date=datetime(2025, 5, 1),
    )


def _analog(deal_id: str, price_m2: float) -> DBAnalog:
    return DBAnalog(
        deal_id=deal_id,
        is_offer=False,
        ts_status_ready=datetime(2025, 4, 15),
        lat=55.0,
        lon=37.0,
        comm_sq=1_000.0,
        price_m2=price_m2,
        region="region-a",
    )


def _references() -> tuple[pl.DataFrame, pl.DataFrame]:
    cities_reference = pl.DataFrame(
        {
            "name": ["huge", "big", "middle", "small"],
            "lat": [55.0, 55.1, 55.2, 55.3],
            "lon": [37.0, 37.1, 37.2, 37.3],
            "size_category": ["huge", "big", "middle", "small"],
        }
    )
    locality_reference = pl.DataFrame(
        {"locality_guid": ["locality-1"], "lon": [37.0]}
    )
    return cities_reference, locality_reference


def test_pipeline_filters_ranks_and_returns_final_response() -> None:
    cities_reference, locality_reference = _references()
    pipeline = LandValuationPipeline.from_components(
        analog_filter_model=ArrayModel([0.1, 0.9, 0.2]),
        analog_correction_model=ArrayModel([1.20, 1.01]),
        main_price_model=ConstantModel(1_000.0),
        confidence_model=ConstantModel(1.3),
        kriging_corrector=ConstantCorrector(25.0),
        cities_reference=cities_reference,
        locality_reference=locality_reference,
        filter_threshold=0.5,
        top_n_final=1,
    )

    response = pipeline.predict(
        _request(),
        [
            _analog("kept-but-lower-rank", 2_000.0),
            _analog("filtered-out", 3_000.0),
            _analog("kept-top-rank", 2_100.0),
        ],
    )

    assert response.price_m2 == pytest.approx(1_025.0)
    assert response.total_price == pytest.approx(1_025_000.0)
    assert response.confidence == pytest.approx(1.0)
    assert response.decision == "auto_approve"
    assert response.base_price_m2 == pytest.approx(1_000.0)
    assert response.kriging_correction == pytest.approx(25.0)
    assert response.latency is not None
    assert response.latency.total_ms > 0.0
    assert [analog.deal_id for analog in response.used_analogs] == ["kept-top-rank"]


def test_pipeline_returns_empty_response_when_no_analogs_survive_filter() -> None:
    cities_reference, locality_reference = _references()
    pipeline = LandValuationPipeline.from_components(
        analog_filter_model=ArrayModel([0.8]),
        analog_correction_model=ConstantModel(1.0),
        main_price_model=ConstantModel(1_000.0),
        confidence_model=ConstantModel(0.5),
        kriging_corrector=ConstantCorrector(25.0),
        cities_reference=cities_reference,
        locality_reference=locality_reference,
        filter_threshold=0.5,
    )

    response = pipeline.predict(_request(), [_analog("bad", 2_000.0)])

    assert response.price_m2 == pytest.approx(0.0)
    assert response.total_price == pytest.approx(0.0)
    assert response.confidence == pytest.approx(0.0)
    assert response.decision == "no_valuation"
    assert response.used_analogs == []
    assert response.latency is not None


def test_response_clips_confidence_before_range_validation() -> None:
    response = ValuationResponse(
        price_m2=1.0,
        total_price=1.0,
        confidence=-0.5,
        base_price_m2=1.0,
        kriging_correction=0.0,
        used_analogs=[],
    )

    assert response.confidence == pytest.approx(0.0)


def test_pipeline_returns_manual_review_for_low_confidence_prediction() -> None:
    cities_reference, locality_reference = _references()
    pipeline = LandValuationPipeline.from_components(
        analog_filter_model=ArrayModel([0.1]),
        analog_correction_model=ArrayModel([1.0]),
        main_price_model=ConstantModel(1_000.0),
        confidence_model=ConstantModel(0.3),
        kriging_corrector=ConstantCorrector(0.0),
        cities_reference=cities_reference,
        locality_reference=locality_reference,
        filter_threshold=0.5,
        auto_approve_threshold=0.75,
    )

    response = pipeline.predict(_request(), [_analog("ok", 2_000.0)])

    assert response.decision == "manual_review"
