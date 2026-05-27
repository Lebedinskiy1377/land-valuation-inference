# Land Valuation Inference Library

Production-inference библиотека для ML-системы оценки стоимости земельных
участков. Разработана в Домклике (экосистема Сбера) в 2024–2025.

> **NDA notice.** Это архитектурный макет публичной части
> inference-пайплайна. Реальные обученные модели, данные, признаки и
> тренировочный код принадлежат Сберу / Домклику и под NDA. В этом
> репозитории - только архитектура, интерфейсы компонент и логика
> оркестрации пяти моделей.

Репозиторий подготовлен как самодостаточный NDA-safe срез проекта для
портфолио / Junior ML Contest: оригинальных артефактов здесь нет, но код можно
запустить на мок-моделях и проверить тестами.

## Содержание

- [Контекст](#контекст)
- [Архитектура](#архитектура)
- [Метрики в production](#метрики-в-production)
- [Структура репозитория](#структура-репозитория)
- [Установка](#установка)
- [Быстрый запуск макета](#быстрый-запуск-макета)
- [Docker](#docker)
- [Проверки качества](#проверки-качества)
- [Использование](#использование)
- [Формат артефактов](#формат-артефактов)
- [Технологии](#технологии)
- [Автор](#автор)

## Контекст

ML-система оценки стоимости земельных участков для экосистемы Сбера.
Применяется в двух системах:

1. **Публичный сервис «Моя недвижимость»** - real-time UI для оценки
   объекта по адресу или кадастровому номеру.
2. **Кредитный конвейер ипотечного процесса Сбербанка** - на основе
   оценки залога и уверенности модели заявка либо автоматически
   одобряется, либо отправляется на ручную проверку оценщику.

Земельные участки сложнее в оценке, чем квартиры или дома: слабая
стандартизация, цена сильно зависит от локации до уровня метров,
много нерыночных сделок, шумный кадастр.

До разработки модели оценка велась простой эвристикой (средняя цена
аналогов в радиусе) с **MAPE 47.6%** и покрытием **40%** объектов.
Большая часть заявок уходила на ручную оценку.

## Архитектура

Пайплайн состоит из четырёх LightGBM-бустингов и геостатистического
корректора:

```text
                    LandRequest + List[DBAnalog]
                              │
                              ▼
                ┌────────────────────────────────┐
                │  build_pairwise_features       │
                │  (target × analog features)    │
                └─────────────┬──────────────────┘
                              │
                              ▼
                ┌────────────────────────────────┐
                │  1. AnalogFilterModel          │
                │  бинарная классификация        │
                │  (отсев плохих аналогов)       │
                │  метрики: F1, ROC AUC          │
                └─────────────┬──────────────────┘
                              │
                              ▼
                ┌────────────────────────────────┐
                │  2. AnalogCorrectionModel      │
                │  регрессия на price ratio      │
                │  (ранжирование, top-5)         │
                │  метрики: Spearman, MAE        │
                └─────────────┬──────────────────┘
                              │
                              ▼
                ┌────────────────────────────────┐
                │  compute_online_features       │
                │  (расстояния, агрегаты)        │
                └─────────────┬──────────────────┘
                              │
                              ▼
                ┌────────────────────────────────┐
                │  3. MainPriceModel             │
                │  regression w/ monotonic       │
                │  constraints                   │
                │  → base_price_m2               │
                │  метрики: MAPE, sMAPE          │
                └─────────────┬──────────────────┘
                              │
                              ▼
                ┌────────────────────────────────┐
                │  4. KrigingCorrector           │
                │  precomputed grid              │
                │  → spatial correction          │
                └─────────────┬──────────────────┘
                              │
                              ▼
                ┌────────────────────────────────┐
                │  5. ConfidenceModel            │
                │  → confidence score            │
                │  метрики: F1, ROC AUC          │
                └─────────────┬──────────────────┘
                              │
                              ▼
                       ValuationResponse
```

Подробная диаграмма: [docs/architecture.png](docs/architecture.png).

### Ключевые компоненты

**AnalogFilterModel** - бинарная классификация. Таргет на обучении:
1 если |price_target − price_analog| / price_target > 0.35.
На inference отсекает кандидатов с высокой proba "плохого аналога".

**AnalogCorrectionModel** - регрессия отношения цен
(price_target / price_analog). Используется только для ранжирования:
аналоги сортируются по близости предсказания к 1.0, оставляется топ-5.
Не модифицирует цены аналогов перед агрегацией.

**MainPriceModel** - основной регрессор цены за м² с monotonic
constraints на ключевых признаках. Входы: признаки объекта,
гео-контекст, агрегаты по топ-5 финальным аналогам.

**KrigingCorrector** - precomputed kriging grid с билинейной
интерполяцией. Решает проблему O(N³) асимптотики наивного кригинга
на 150 000 точек обучения. Подробнее ниже.

**ConfidenceModel** - оценка уверенности прогноза. Используется
кредитным конвейером для решения «автоодобрение или ручная проверка».

### Precomputed kriging grid

Наивная реализация инференса гауссовского кригинга на 150 000 точек
обучения имеет O(N³) асимптотику на обращении матрицы ковариаций и
не помещается в latency-бюджет real-time запроса.

Решение: предрасчитанная регулярная сетка значений кригинговой
коррекции с адаптивной плотностью узлов (плотнее в регионах с
высокой плотностью обучающих данных, реже в разреженных). Inference
сводится к O(1) через билинейную интерполяцию между ближайшими
узлами сетки.

Поскольку кригинговая поверхность остатков гладкая по построению,
ошибка интерполяции пренебрежимо мала по сравнению с ошибкой
основной модели.

### Online feature engineering

Часть признаков нельзя посчитать оффлайн, поскольку они зависят от
запроса:

- Расстояния от точки до ближайших аналогов и до городов разного
  размера (huge / big / middle / small).
- Агрегаты от уже отобранного топ-5 (отдельно для сделок и офферов).
- locality_lon - координата центра locality из 2GIS reference по
  locality_guid.

Реализовано на Polars и NumPy с векторизованными haversine для
соответствия latency-требованиям.

## Метрики в production

На 150 000 объектов в production-системах Сбера:

| Категория      | Метрика                       | Baseline | Финал                |
|----------------|-------------------------------|----------|----------------------|
| Точность       | MAPE                          | 47.6%    | **18%**              |
|                | Покрытие                      | 40%      | **72%**              |
| Бизнес-эффект  | Доля автоодобрений            | 24%      | **49%**              |
|                | Экономия на ручных оценках    | -        | ~десятки млн ₽/год*  |

\* Прокси-оценка через объём заявок, переведённых с ручной оценки на
автоматическую, и среднюю стоимость ручной оценки. Прямые бизнес-метрики
(GMV, дефолтность) находились на стороне Сбера и в DS-команду Домклика
не возвращались.

MAPE снижен в 2.6 раза, покрытие выросло почти в 2 раза. Доля
автоматических одобрений в кредитном конвейере выросла с 24% до 49%.

## Структура репозитория

```text
land-valuation-inference/
├── inference/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic: LandRequest, DBAnalog, ValuationResponse
│   ├── models.py               # Обёртки над LightGBM-бустингами и KrigingCorrector
│   ├── distance_utils.py       # Векторизованный haversine
│   ├── analog_retrieval.py     # BBox prefilter + BallTree отбор аналогов
│   ├── pairwise_features.py    # Признаки пары (target, analog)
│   ├── online_features.py      # Online-фичи: расстояния, агрегаты
│   └── pipeline.py             # LandValuationPipeline - оркестрация
├── examples/
│   └── smoke_demo.py           # Запуск пайплайна на toy-моделях без NDA-артефактов
├── tests/                      # Unit-тесты публичного макета
├── .github/workflows/
│   └── ci.yml                  # CI: lint, mypy, pytest, smoke demo
├── docs/
│   ├── architecture.png        # Архитектурная диаграмма
│   └── nda_scope.md            # Что включено / исключено из публичной версии
├── db/
│   ├── init/                   # Schema + seed для локального Postgres
│   └── queries/                # Пример bbox SQL перед BallTree refinement
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
└── LICENSE
```

## Установка

Требуется Python 3.11+.

```bash
git clone https://github.com/<username>/land-valuation-inference.git
cd land-valuation-inference

python -m venv venv
source venv/bin/activate

pip install -e ".[dev]"
```

## Быстрый запуск макета

Smoke-demo использует toy-модели с тем же интерфейсом, что и production
LightGBM-компоненты. Это нужно только для проверки оркестрации пайплайна без
закрытых артефактов.

```bash
python -m examples.smoke_demo
pytest
```

Для полного локального набора проверок:

```bash
make check
```

## Docker

Docker-контур нужен для воспроизводимой проверки публичного макета без
production-артефактов.

```bash
docker build -t land-valuation-inference .
docker run --rm land-valuation-inference
```

Через Docker Compose:

```bash
docker compose run --rm demo
docker compose run --rm test
docker compose run --rm lint
```

Локальный Postgres с синтетической hot-storage таблицей аналогов:

```bash
docker compose up -d postgres
docker compose run --rm db-check
```

## Проверки качества

```bash
python -m ruff check .
python -m mypy inference
python -m pytest
python -m examples.smoke_demo
```

## Использование

```python
import polars as pl

from inference import AnalogRetrievalConfig, LandRequest, LandValuationPipeline, retrieve_analogs

pipeline = LandValuationPipeline(
    analog_filter_path="artifacts/analog_filter.txt",
    analog_correction_path="artifacts/analog_correction.txt",
    main_price_path="artifacts/main_price.txt",
    confidence_path="artifacts/confidence.txt",
    kriging_corrector_path="artifacts/kriging_grid.pkl",
    cities_reference=pl.read_parquet("data/cities.parquet"),
    locality_reference=pl.read_parquet("data/localities.parquet"),
)

request = LandRequest(
    lat=55.7558,
    lon=37.6173,
    locality_guid="moscow-locality-guid",
    comm_sq=1500.0,
    region="Москва",
)

analogs = retrieve_analogs(
    request=request,
    analogs_table=pl.read_parquet("data/hot_storage_analogs.parquet"),
    config=AnalogRetrievalConfig(
        bbox_delta_lat=0.02,
        bbox_delta_lon=0.02,
        radius_km=3.0,
        area_tolerance=0.2,
        max_age_days=240,
    ),
)

result = pipeline.predict(request, analogs)

print(f"Price per m²: {result.price_m2:,.0f}")
print(f"Total: {result.total_price:,.0f}")
print(f"Confidence: {result.confidence:.2%}")
```

Метод `pipeline.predict()` без артефактов работать не будет -
для запуска нужны пути к файлам бустингов и precomputed kriging grid.
Для публичного запуска без артефактов см. `examples/smoke_demo.py`.

## Формат артефактов

**LightGBM-бустинги:** `.txt` (нативный формат через
`Booster.save_model`) или `.pkl` (сериализованный booster).

**KrigingCorrector:** pickle-файл со словарём:

```python
{
    "lats": np.ndarray,           # 1D, отсортированный по возрастанию
    "lons": np.ndarray,           # 1D, отсортированный по возрастанию
    "corrections": np.ndarray,    # 2D, shape (len(lats), len(lons))
}
```

**cities_reference:** Polars DataFrame с колонками
`name`, `lat`, `lon`, `size_category` (huge / big / middle / small).

**locality_reference:** Polars DataFrame с колонками
`locality_guid`, `lon`.

## Технологии

**В этом репозитории:**
Python · LightGBM · Polars · NumPy · SciPy · Pydantic v2 · pytest

**В production-системе дополнительно:**
Optuna (подбор гиперпараметров), GeoPandas / Shapely (гео-операции при
подготовке обучающих данных), scikit-learn (валидация), FastAPI и Airflow
(обёртка inference в сервис), Docker и Kubernetes (развёртывание),
PostgreSQL (PostGIS) / Greenplum (ETL обучающих витрин)

## Автор

Дмитрий Лебединский, Data Scientist, Альфа-Банк.
Проект разработан в Домклике (экосистема Сбера) в 2024–2025.
