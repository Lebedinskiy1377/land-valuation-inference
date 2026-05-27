"""NDA-safe smoke demo for the land valuation inference pipeline.

The real project used trained LightGBM models and production reference tables.
This example keeps only the public orchestration contract: pairwise filtering,
ranking, online features, spatial correction, and confidence scoring.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from inference import DBAnalog, LandRequest, LandValuationPipeline


class DistanceFilterModel:
    """Toy analog filter: far candidates are more likely to be rejected."""

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        distance_score = features["distance_km"].to_numpy() / 15.0
        area_score = np.abs(features["area_ratio"].to_numpy() - 1.0)
        return np.clip(distance_score + area_score, 0.0, 1.0)


class AnalogRankingModel:
    """Toy ranker: prefers close area and close geography."""

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        area_penalty = 0.08 * np.abs(features["area_ratio"].to_numpy() - 1.0)
        distance_penalty = 0.01 * features["distance_km"].to_numpy()
        return 1.0 + area_penalty + distance_penalty


class MainPriceStub:
    """Toy price model based on median analog prices."""

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        row = features.row(0, named=True)
        candidates = (
            row["median_analogs_deal"],
            row["mean_analogs_deal"],
            row["median_analogs_offer"],
            row["mean_analogs_offer"],
        )
        base = next((value for value in candidates if not np.isnan(value)), 1_000.0)
        location_multiplier = 1.0 + max(0.0, 10.0 - row["dist"]) / 200.0
        return np.array([base * location_multiplier])


class ConfidenceStub:
    """Toy confidence model: closer analogs produce higher confidence."""

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        dist = features["dist"].to_numpy()[0]
        confidence = 0.92 - min(dist, 20.0) / 40.0
        return np.array([confidence])


class ConstantKrigingCorrector:
    """Toy spatial correction in rubles per m²."""

    def correct(self, lat: float, lon: float) -> float:
        return 180.0


def build_demo_pipeline() -> LandValuationPipeline:
    cities_reference = pl.DataFrame(
        {
            "name": ["Moscow", "Tula", "Serpukhov", "Tarusa"],
            "lat": [55.7558, 54.1931, 54.9135, 54.7244],
            "lon": [37.6173, 37.6173, 37.4116, 37.1767],
            "size_category": ["huge", "big", "middle", "small"],
        }
    )
    locality_reference = pl.DataFrame(
        {
            "locality_guid": ["demo-locality"],
            "lon": [37.4],
        }
    )

    return LandValuationPipeline.from_components(
        analog_filter_model=DistanceFilterModel(),
        analog_correction_model=AnalogRankingModel(),
        main_price_model=MainPriceStub(),
        confidence_model=ConfidenceStub(),
        kriging_corrector=ConstantKrigingCorrector(),
        cities_reference=cities_reference,
        locality_reference=locality_reference,
        filter_threshold=0.75,
        top_n_final=5,
    )


def build_demo_request() -> LandRequest:
    return LandRequest(
        lat=55.12,
        lon=37.42,
        locality_guid="demo-locality",
        comm_sq=1_500.0,
        region="Московская область",
        request_date=datetime(2025, 5, 1),
    )


def build_demo_analogs() -> list[DBAnalog]:
    request_date = datetime(2025, 5, 1)
    return [
        DBAnalog(
            deal_id="deal-001",
            is_offer=False,
            ts_status_ready=request_date - timedelta(days=23),
            lat=55.121,
            lon=37.421,
            comm_sq=1_450.0,
            price_m2=2_950.0,
            region="Московская область",
        ),
        DBAnalog(
            deal_id="offer-002",
            is_offer=True,
            ts_status_ready=request_date - timedelta(days=41),
            lat=55.118,
            lon=37.417,
            comm_sq=1_620.0,
            price_m2=3_150.0,
            region="Московская область",
        ),
        DBAnalog(
            deal_id="deal-003",
            is_offer=False,
            ts_status_ready=request_date - timedelta(days=71),
            lat=55.126,
            lon=37.432,
            comm_sq=1_510.0,
            price_m2=3_020.0,
            region="Московская область",
        ),
        DBAnalog(
            deal_id="far-offer-004",
            is_offer=True,
            ts_status_ready=request_date - timedelta(days=12),
            lat=55.42,
            lon=38.05,
            comm_sq=2_200.0,
            price_m2=5_800.0,
            region="Московская область",
        ),
    ]


def main() -> None:
    pipeline = build_demo_pipeline()
    response = pipeline.predict(build_demo_request(), build_demo_analogs())

    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
