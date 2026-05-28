"""Minimal FastAPI wrapper for the smoke inference pipeline.

Install with:
    pip install -e ".[api]"
"""

from __future__ import annotations

from fastapi import FastAPI

from examples.smoke_demo import (
    build_demo_analogs,
    build_demo_pipeline,
    build_demo_request,
)

app = FastAPI(title="Land Valuation Inference Demo", version="0.1.0")
pipeline = build_demo_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/demo")
def predict_demo() -> dict[str, object]:
    response = pipeline.predict(build_demo_request(), build_demo_analogs())
    return response.model_dump(mode="json")
