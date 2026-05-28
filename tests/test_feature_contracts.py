import polars as pl
import pytest

from inference import MAIN_PRICE_FEATURES, FeatureContract


def test_feature_contract_reorders_columns_and_drops_extras() -> None:
    features = pl.DataFrame(
        {
            "b": [2],
            "extra": [3],
            "a": [1],
        }
    )
    contract = FeatureContract(name="toy", columns=("a", "b"))

    validated = contract.validate(features)

    assert validated.columns == ["a", "b"]
    assert validated.row(0) == (1, 2)


def test_feature_contract_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="main_price"):
        MAIN_PRICE_FEATURES.validate(pl.DataFrame({"lat": [55.0]}))
