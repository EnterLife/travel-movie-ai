"""Run the deterministic large-project metadata benchmark."""

from __future__ import annotations

import argparse
import time

from travelmovieai.application.scale_benchmark import run_synthetic_metadata_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, default=512)
    parser.add_argument("--source-gib", type=float, default=128.0)
    arguments = parser.parse_args()
    started = time.perf_counter()
    result = run_synthetic_metadata_benchmark(
        asset_count=arguments.assets,
        total_source_bytes=round(arguments.source_gib * 1024**3),
    )
    elapsed = time.perf_counter() - started
    print(result.model_dump_json(indent=2))
    print(f"metadata_benchmark_elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
