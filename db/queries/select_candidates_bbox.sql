-- Coarse hot-storage query before exact BallTree refinement in Python.
-- Parameters here match examples/smoke_demo.py.

SELECT
    deal_id,
    is_offer,
    ts_status_ready,
    lat,
    lon,
    comm_sq,
    price_m2,
    region
FROM hot_storage_analogs
WHERE lat BETWEEN 55.12 - 0.02 AND 55.12 + 0.02
  AND lon BETWEEN 37.42 - 0.02 AND 37.42 + 0.02
  AND comm_sq BETWEEN 1500.0 * 0.8 AND 1500.0 * 1.2
  AND ts_status_ready >= TIMESTAMPTZ '2025-05-01T00:00:00Z' - INTERVAL '240 days'
  AND ts_status_ready <= TIMESTAMPTZ '2025-05-01T00:00:00Z'
  AND price_m2 > 0
  AND region = 'Московская область'
ORDER BY ts_status_ready DESC;
