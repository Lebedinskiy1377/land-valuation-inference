"""Latency benchmark for the NDA-safe smoke pipeline."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from examples.smoke_demo import (
    build_demo_analogs,
    build_demo_pipeline,
    build_demo_request,
)


def run_benchmark(iterations: int) -> dict[str, float]:
    pipeline = build_demo_pipeline()
    request = build_demo_request()
    analogs = build_demo_analogs()

    durations_ms: list[float] = []
    for _ in range(iterations):
        start = perf_counter()
        pipeline.predict(request, analogs)
        durations_ms.append((perf_counter() - start) * 1000.0)

    values = np.array(durations_ms)
    return {
        "iterations": float(iterations),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    result = run_benchmark(args.iterations)
    for key, value in result.items():
        if key == "iterations":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
