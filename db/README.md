# Test Postgres

Минимальная локальная БД для demo hot-storage слоя аналогов.

```bash
docker compose up -d postgres
docker compose run --rm db-check
docker compose exec postgres psql -U land_user -d land_valuation
```

Внутри:

- `hot_storage_analogs` - маленькая seed-таблица аналогов.
- `init/001_schema.sql` - схема и индексы для bbox-prefilter.
- `init/002_seed.sql` - синтетические NDA-safe строки.
- `queries/select_candidates_bbox.sql` - пример SQL-кандидатов перед точным
  BallTree-refinement в Python.

Сбросить локальное состояние:

```bash
docker compose down -v
```
