"""Feature contracts for inference-time model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class FeatureContract:
    """Expected feature columns for a model input frame."""

    name: str
    columns: tuple[str, ...]

    def validate(self, features: pl.DataFrame) -> pl.DataFrame:
        """Validate required columns and return them in model order."""
        missing = [column for column in self.columns if column not in features.columns]
        if missing:
            raise ValueError(
                f"{self.name} feature frame is missing required columns: {missing}"
            )
        return features.select(self.columns)


PAIRWISE_FEATURES = FeatureContract(
    name="pairwise",
    columns=(
        "target_lat",
        "target_lon",
        "target_comm_sq",
        "target_region",
        "analog_lat",
        "analog_lon",
        "analog_comm_sq",
        "analog_price_m2",
        "analog_region",
        "analog_is_offer",
        "analog_age_days",
        "distance_km",
        "area_ratio",
        "region_match",
    ),
)

MAIN_PRICE_FEATURES = FeatureContract(
    name="main_price",
    columns=(
        "lat",
        "comm_sq",
        "region",
        "dist",
        "dist_to_huge_city",
        "dist_to_big_city",
        "dist_to_middle_city",
        "dist_to_small_city",
        "locality_lon",
        "mean_analogs_deal",
        "median_analogs_deal",
        "mean_analogs_offer",
        "median_analogs_offer",
    ),
)

CONFIDENCE_FEATURES = FeatureContract(
    name="confidence",
    columns=(*MAIN_PRICE_FEATURES.columns, "price_m2_pred"),
)
