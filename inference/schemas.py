"""Pydantic-схемы для inference."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ValuationResponse(BaseModel):
    """Результат оценки."""

    price_m2: float
    total_price: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    base_price_m2: float
    kriging_correction: float
    used_analogs: list[DBAnalog]

    @field_validator("confidence")
    @classmethod
    def _clip_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))
