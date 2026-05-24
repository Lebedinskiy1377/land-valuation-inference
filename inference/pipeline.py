"""Главный класс inference-пайплайна."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .models import (
    AnalogCorrectionModel,
    AnalogFilterModel,
    ConfidenceModel,
    KrigingCorrector,
    MainPriceModel,
)
from .online_features import compute_online_features
from .pairwise_features import build_pairwise_features
from .schemas import DBAnalog, LandRequest, ValuationResponse


class LandValuationPipeline:
    """Inference-пайплайн оценки стоимости земельного участка.

    Загружает 4 LightGBM-бустинга и кригинг-корректор. На каждый запрос
    выполняет:
        1. AnalogFilterModel - отсекает плохих аналогов.
        2. AnalogCorrectionModel - ранжирует по эквивалентности цены.
        3. compute_online_features - расстояния и агрегаты.
        4. MainPriceModel - базовая цена за м^2.
        5. KrigingCorrector - пространственная коррекция остатков.
        6. ConfidenceModel - уверенность; на вход получает все признаки
           MainPriceModel плюс её собственный предикт (price_m2_pred).

    Аналоги приходят извне как список DBAnalog (предварительно отобранные
    в hot-storage витрине по правилу подбора: радиус 3 км, отклонение
    площади ±20%, возраст до 240 дней - см. слой выборки аналогов).
    """

    def __init__(
        self,
        analog_filter_path: str | Path,
        analog_correction_path: str | Path,
        main_price_path: str | Path,
        confidence_path: str | Path,
        kriging_corrector_path: str | Path,
        cities_reference: pl.DataFrame,
        locality_reference: pl.DataFrame,
        filter_threshold: float = 0.5,
        top_n_final: int = 5,
    ) -> None:
        self.analog_filter_model = AnalogFilterModel(analog_filter_path)
        self.analog_correction_model = AnalogCorrectionModel(analog_correction_path)
        self.main_price_model = MainPriceModel(main_price_path)
        self.confidence_model = ConfidenceModel(confidence_path)
        self.kriging_corrector = KrigingCorrector(kriging_corrector_path)

        self.cities_reference = cities_reference
        self.locality_reference = locality_reference
        self.filter_threshold = filter_threshold
        self.top_n_final = top_n_final

    def predict(
        self,
        request: LandRequest,
        analogs: list[DBAnalog],
    ) -> ValuationResponse:
        if not analogs:
            return self._empty_response()

        # 1. Pairwise features
        pairwise = build_pairwise_features(request, analogs)

        # 2. Фильтрация плохих аналогов
        bad_proba = self.analog_filter_model.predict(pairwise)
        keep_mask = bad_proba < self.filter_threshold
        if not keep_mask.any():
            return self._empty_response()

        filtered_analogs = [
            a for a, keep in zip(analogs, keep_mask) if keep
        ]  # noqa: B905
        pairwise_filtered = pairwise.filter(pl.Series(keep_mask))

        # 3. Ранжирование по близости отношения цен к 1.0
        ratio_pred = self.analog_correction_model.predict(pairwise_filtered)
        ratio_distance = np.abs(ratio_pred - 1.0)

        top_indices = np.argsort(ratio_distance)[: self.top_n_final]
        top_analogs = [filtered_analogs[i] for i in top_indices]

        # 4. Online features (для MainPriceModel)
        features = compute_online_features(
            request=request,
            top_analogs=top_analogs,
            cities_reference=self.cities_reference,
            locality_reference=self.locality_reference,
        )

        # 5. Основной прогноз
        base_price_m2 = float(self.main_price_model.predict(features)[0])

        # 6. Кригинг-коррекция
        kriging_correction = self.kriging_corrector.correct(request.lat, request.lon)
        final_price_m2 = base_price_m2 + kriging_correction

        # 7. Confidence - добавляем price_m2_pred как фичу
        features_with_pred = features.with_columns(
            pl.lit(base_price_m2).alias("price_m2_pred")
        )
        confidence = float(self.confidence_model.predict(features_with_pred)[0])

        return ValuationResponse(
            price_m2=final_price_m2,
            total_price=final_price_m2 * request.comm_sq,
            confidence=confidence,
            base_price_m2=base_price_m2,
            kriging_correction=kriging_correction,
            used_analogs=top_analogs,
        )

    @staticmethod
    def _empty_response() -> ValuationResponse:
        return ValuationResponse(
            price_m2=0.0,
            total_price=0.0,
            confidence=0.0,
            base_price_m2=0.0,
            kriging_correction=0.0,
            used_analogs=[],
        )  # noqa: W292
