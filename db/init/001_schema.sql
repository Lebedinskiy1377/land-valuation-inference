CREATE TABLE IF NOT EXISTS hot_storage_analogs (
    deal_id TEXT PRIMARY KEY,
    is_offer BOOLEAN NOT NULL,
    ts_status_ready TIMESTAMPTZ NOT NULL,
    lat DOUBLE PRECISION NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lon DOUBLE PRECISION NOT NULL CHECK (lon BETWEEN -180 AND 180),
    comm_sq DOUBLE PRECISION NOT NULL CHECK (comm_sq > 0),
    price_m2 DOUBLE PRECISION NOT NULL CHECK (price_m2 > 0),
    region TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hot_storage_analogs_lat_lon
    ON hot_storage_analogs (lat, lon);

CREATE INDEX IF NOT EXISTS idx_hot_storage_analogs_region
    ON hot_storage_analogs (region);

CREATE INDEX IF NOT EXISTS idx_hot_storage_analogs_ready
    ON hot_storage_analogs (ts_status_ready);
