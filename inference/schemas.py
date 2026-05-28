"""Pydantic-схемы для inference."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ValuationDecision = Literal["auto_approve", "manual_review", "no_valuation"]


class LandRequest(BaseModel):
    """Входной запрос на оценку земельного участка."""

    model_config = ConfigDict(strict=False)

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    locality_guid: str
    comm_sq: float = Field(..., gt=0.0)
    region: str
    request_date: datetime = Field(default_factory=datetime.utcnow)


class DBAnalog(BaseModel):
    """Аналог из витрины (hot-storage таблица аналогов)."""

    model_config = ConfigDict(strict=False)

    deal_id: str
    is_offer: bool
    ts_status_ready: datetime
    lat: float
    lon: float
    comm_sq: float = Field(..., gt=0.0)
    price_m2: float = Field(..., gt=0.0)
    region: str


class LatencyTrace(BaseModel):
    """Latency breakdown for one inference request in milliseconds."""

    pairwise_ms: float = 0.0
    filter_ms: float = 0.0
    ranking_ms: float = 0.0
    online_features_ms: float = 0.0
    main_model_ms: float = 0.0
    kriging_ms: float = 0.0
    confidence_ms: float = 0.0
    total_ms: float = 0.0


class ValuationResponse(BaseModel):
    """Результат оценки."""

    price_m2: float
    total_price: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision: ValuationDecision = "manual_review"
    base_price_m2: float
    kriging_correction: float
    used_analogs: list[DBAnalog]
    latency: LatencyTrace | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clip_confidence(cls, v: float) -> float:
        value = float(v)
        return max(0.0, min(1.0, value))
