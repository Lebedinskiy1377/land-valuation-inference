# NDA-safe scope

Этот репозиторий - публичный код-макет production inference библиотеки для
проекта оценки стоимости земельных участков. Он показывает архитектуру,
контракты компонент и оркестрацию пяти моделей, но не раскрывает закрытые
артефакты Домклика / Сбера.

## Что включено

- Pydantic-схемы входного запроса, аналога и ответа модели.
- Feature contracts для валидации колонок на входе моделей.
- Обёртки над LightGBM-бустингами и precomputed kriging grid.
- Публичный макет retrieval-слоя для аналогов: bbox-prefilter и BallTree
  haversine radius search.
- Построение pairwise-признаков для фильтрации и ранжирования аналогов.
- Online feature engineering на Polars / NumPy.
- `LandValuationPipeline` как production-like оркестратор инференса.
- Smoke-demo с toy-моделями, которое можно запустить без приватных артефактов.
- Unit-тесты ключевых участков: гео-расчёты, признаки, фильтрация,
  ранжирование, confidence clipping.
- Latency trace, benchmark и минимальный optional FastAPI wrapper.
- Dockerfile, Docker Compose, Makefile и CI-конфигурация для воспроизводимой
  проверки макета.
- Локальная Postgres-схема с синтетической seed-таблицей аналогов для проверки
  hot-storage контракта.

## Что намеренно не включено

- Обученные LightGBM-модели и pickle-артефакты kriging grid.
- Реальные данные, обучающие витрины, выгрузки из БД и feature store.
- SQL / ETL / Airflow DAG production-контура.
- Реальные таблицы hot-storage и production-индексы БД.
- Точные списки приватных признаков, веса моделей, SHAP-отчёты и MLflow runs.
- Интеграционные настройки сервисов Сбера, Домклика, 2GIS и внутренних API.
- Часть фичей
