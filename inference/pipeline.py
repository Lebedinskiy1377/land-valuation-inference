"""Главный класс inference-пайплайна."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
import polars as pl

from .feature_contracts import CONFIDENCE_FEATURES, MAIN_PRICE_FEATURES, PAIRWISE_FEATURES
from .models import (
    AnalogCorrectionModel,
    AnalogFilterModel,
    ConfidenceModel,
    KrigingCorrector,
    MainPriceModel,
)
from .online_features import compute_online_features
from .pairwise_features import build_pairwise_features
from .schemas import DBAnalog, LandRequest, LatencyTrace, ValuationDecision, ValuationResponse


class FeatureModel(Protocol):
    """Минимальный интерфейс табличной модели для inference."""

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        """Возвращает предсказания для батча признаков."""


class SpatialCorrector(Protocol):
    """Минимальный интерфейс пространственной поправки."""

    def correct(self, lat: float, lon: float) -> float:
        """Возвращает поправку к цене за м²."""


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
        auto_approve_threshold: float = 0.75,
    ) -> None:
        self._configure(
            analog_filter_model=AnalogFilterModel(analog_filter_path),
            analog_correction_model=AnalogCorrectionModel(analog_correction_path),
            main_price_model=MainPriceModel(main_price_path),
            confidence_model=ConfidenceModel(confidence_path),
            kriging_corrector=KrigingCorrector(kriging_corrector_path),
            cities_reference=cities_reference,
            locality_reference=locality_reference,
            filter_threshold=filter_threshold,
            top_n_final=top_n_final,
            auto_approve_threshold=auto_approve_threshold,
        )

    @classmethod
    def from_components(
        cls,
        analog_filter_model: FeatureModel,
        analog_correction_model: FeatureModel,
        main_price_model: FeatureModel,
        confidence_model: FeatureModel,
        kriging_corrector: SpatialCorrector,
        cities_reference: pl.DataFrame,
        locality_reference: pl.DataFrame,
        filter_threshold: float = 0.5,
        top_n_final: int = 5,
        auto_approve_threshold: float = 0.75,
    ) -> LandValuationPipeline:
        """Создаёт пайплайн из уже загруженных компонентов.

        Production-код обычно использует пути к LightGBM/kriging артефактам.
        Этот конструктор нужен для тестов, smoke-demo и интеграций, где модели
        создаются внешним контейнером зависимостей.
        """
        pipeline = cls.__new__(cls)
        pipeline._configure(
            analog_filter_model=analog_filter_model,
            analog_correction_model=analog_correction_model,
            main_price_model=main_price_model,
            confidence_model=confidence_model,
            kriging_corrector=kriging_corrector,
            cities_reference=cities_reference,
            locality_reference=locality_reference,
            filter_threshold=filter_threshold,
            top_n_final=top_n_final,
            auto_approve_threshold=auto_approve_threshold,
        )
        return pipeline

    def _configure(
        self,
        analog_filter_model: FeatureModel,
        analog_correction_model: FeatureModel,
        main_price_model: FeatureModel,
        confidence_model: FeatureModel,
        kriging_corrector: SpatialCorrector,
        cities_reference: pl.DataFrame,
        locality_reference: pl.DataFrame,
        filter_threshold: float,
        top_n_final: int,
        auto_approve_threshold: float,
    ) -> None:
        self.analog_filter_model = analog_filter_model
        self.analog_correction_model = analog_correction_model
        self.main_price_model = main_price_model
        self.confidence_model = confidence_model
        self.kriging_corrector = kriging_corrector

        self.cities_reference = cities_reference
        self.locality_reference = locality_reference
        self.filter_threshold = filter_threshold
        self.top_n_final = top_n_final
        self.auto_approve_threshold = auto_approve_threshold

    def predict(
        self,
        request: LandRequest,
        analogs: list[DBAnalog],
    ) -> ValuationResponse:
        total_start = perf_counter()
        timings: dict[str, float] = {}

        if not analogs:
            return self._empty_response(latency=self._latency_trace(timings, total_start))

        # 1. Pairwise features
        with self._measure(timings, "pairwise_ms"):
            pairwise = PAIRWISE_FEATURES.validate(build_pairwise_features(request, analogs))

        # 2. Фильтрация плохих аналогов
        with self._measure(timings, "filter_ms"):
            bad_proba = self.analog_filter_model.predict(pairwise)
            keep_mask = bad_proba < self.filter_threshold
        if not keep_mask.any():
            return self._empty_response(latency=self._latency_trace(timings, total_start))

        filtered_analogs = [a for a, keep in zip(analogs, keep_mask, strict=False) if keep]
        pairwise_filtered = pairwise.filter(pl.Series(keep_mask))

        # 3. Ранжирование по близости отношения цен к 1.0
        with self._measure(timings, "ranking_ms"):
            ratio_pred = self.analog_correction_model.predict(pairwise_filtered)
            ratio_distance = np.abs(ratio_pred - 1.0)

            top_indices = np.argsort(ratio_distance)[: self.top_n_final]
            top_analogs = [filtered_analogs[i] for i in top_indices]

        # 4. Online features (для MainPriceModel)
        with self._measure(timings, "online_features_ms"):
            features = MAIN_PRICE_FEATURES.validate(
                compute_online_features(
                    request=request,
                    top_analogs=top_analogs,
                    cities_reference=self.cities_reference,
                    locality_reference=self.locality_reference,
                )
            )

        # 5. Основной прогноз
        with self._measure(timings, "main_model_ms"):
            base_price_m2 = float(self.main_price_model.predict(features)[0])

        # 6. Кригинг-коррекция
        with self._measure(timings, "kriging_ms"):
            kriging_correction = self.kriging_corrector.correct(request.lat, request.lon)
            final_price_m2 = base_price_m2 + kriging_correction

        # 7. Confidence - добавляем price_m2_pred как фичу
        with self._measure(timings, "confidence_ms"):
            features_with_pred = CONFIDENCE_FEATURES.validate(
                features.with_columns(pl.lit(base_price_m2).alias("price_m2_pred"))
            )
            confidence = float(self.confidence_model.predict(features_with_pred)[0])

        return ValuationResponse(
            price_m2=final_price_m2,
            total_price=final_price_m2 * request.comm_sq,
            confidence=confidence,
            decision=self._decision(final_price_m2, confidence),
            base_price_m2=base_price_m2,
            kriging_correction=kriging_correction,
            used_analogs=top_analogs,
            latency=self._latency_trace(timings, total_start),
        )

    def _empty_response(self, latency: LatencyTrace | None = None) -> ValuationResponse:
        return ValuationResponse(
            price_m2=0.0,
            total_price=0.0,
            confidence=0.0,
            decision="no_valuation",
            base_price_m2=0.0,
            kriging_correction=0.0,
            used_analogs=[],
            latency=latency,
        )

    def _decision(self, price_m2: float, confidence: float) -> ValuationDecision:
        if price_m2 <= 0.0 or confidence <= 0.0:
            return "no_valuation"
        if confidence >= self.auto_approve_threshold:
            return "auto_approve"
        return "manual_review"

    @staticmethod
    @contextmanager
    def _measure(timings: dict[str, float], key: str):
        start = perf_counter()
        try:
            yield
        finally:
            timings[key] = (perf_counter() - start) * 1000.0

    @staticmethod
    def _latency_trace(timings: dict[str, float], total_start: float) -> LatencyTrace:
        return LatencyTrace(
            pairwise_ms=timings.get("pairwise_ms", 0.0),
            filter_ms=timings.get("filter_ms", 0.0),
            ranking_ms=timings.get("ranking_ms", 0.0),
            online_features_ms=timings.get("online_features_ms", 0.0),
            main_model_ms=timings.get("main_model_ms", 0.0),
            kriging_ms=timings.get("kriging_ms", 0.0),
            confidence_ms=timings.get("confidence_ms", 0.0),
            total_ms=(perf_counter() - total_start) * 1000.0,
        )
