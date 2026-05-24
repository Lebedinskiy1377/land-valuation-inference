"""Обёртки над артефактами моделей: 4 бустинга LightGBM + кригинг-корректор."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.interpolate import RegularGridInterpolator


class BoostingModel(ABC):
    """Базовый класс для LightGBM-моделей в пайплайне.

    Загружает обученный booster из файла (.txt — нативный формат LightGBM,
    или .pkl — сериализованный через pickle).
    """

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        if model_path.suffix == ".txt":
            self._booster = lgb.Booster(model_file=str(model_path))
        elif model_path.suffix in {".pkl", ".pickle"}:
            with model_path.open("rb") as f:
                self._booster = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported model format: {model_path.suffix}. "
                "Expected .txt or .pkl."
            )

    @abstractmethod
    def predict(self, features: pl.DataFrame) -> np.ndarray:
        """Возвращает массив предсказаний для батча признаков."""
        pass


class AnalogFilterModel(BoostingModel):
    """Бинарный классификатор плохих аналогов.

    Таргет на обучении: 1, если |price_target − price_analog| / price_target > 0.35.
    На inference возвращает proba "плохого" аналога в [0, 1].
    В пайплайне отсекаем кандидатов с proba >= filter_threshold.
    """

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        return np.asarray(self._booster.predict(features.to_numpy()))


class AnalogCorrectionModel(BoostingModel):
    """Регрессор отношения цен (price_target / price_analog).

    На inference используется для ранжирования: сортируем аналоги по
    |predicted_ratio − 1.0|, берём top-N ближайших к 1.0 как наиболее
    эквивалентные по цене.
    """

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        return np.asarray(self._booster.predict(features.to_numpy()))


class MainPriceModel(BoostingModel):
    """Основной регрессор цены за квадратный метр (с monotonic constraints)."""

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        return np.asarray(self._booster.predict(features.to_numpy()))


class ConfidenceModel(BoostingModel):
    """Регрессор уверенности модели в прогнозе. Возвращает скор в [0, 1]."""

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        return np.asarray(self._booster.predict(features.to_numpy()))


class KrigingCorrector:
    """Precomputed kriging grid с билинейной интерполяцией.

    Артефакт - pickle со словарём:
        {
            "lats": np.ndarray (1D, отсортированный),
            "lons": np.ndarray (1D, отсортированный),
            "corrections": np.ndarray (2D, shape (len(lats), len(lons))),
        }

    Inference: O(1) на точку через RegularGridInterpolator.
    Объяснение: наивный кригинг на N точек обучения имеет O(N^3) на
    обращении матрицы ковариаций - не помещается в latency бюджет.
    Предрасчитанная сетка позволяет свести inference к интерполяции
    между ближайшими узлами без потери качества (поверхность остатков
    гладкая по построению).
    """

    def __init__(self, artifact_path: str | Path) -> None:
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Kriging artifact not found: {artifact_path}")

        with artifact_path.open("rb") as f:
            data = pickle.load(f)

        self._interpolator = RegularGridInterpolator(
            points=(data["lats"], data["lons"]),
            values=data["corrections"],
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def correct(self, lat: float, lon: float) -> float:
        return float(self._interpolator([[lat, lon]])[0])
