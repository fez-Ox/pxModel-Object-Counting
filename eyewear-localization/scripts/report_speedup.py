#!/usr/bin/env python3
"""Report latency and speedup metrics for two pipeline result directories.

The script compares result JSON timing records without treating the optimized
run as a correctness oracle.  Positive speedup means the optimized run has a
lower timing value than the reference run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

TIMING_FIELDS = (
    "l0_perception_seconds",
    "c1_onproduct_seconds",
    "c2_c4_fusion_seconds",
    "brand_association_total_seconds",
    "total_pipeline_seconds",
)


def _json_dir(path: str | Path) -> Path:
    root = Path(path).expanduser()
    nested = root / "outputs"
    return nested if nested.is_dir() else root


def load_timings(path: str | Path) -> dict[str, dict[str, float]]:
    """Load timing maps keyed by result filename stem."""
    directory = _json_dir(path)
    timings: dict[str, dict[str, float]] = {}
    for result_path in sorted(directory.glob("*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        values = payload.get("timings", {})
        if not isinstance(values, dict):
            continue
        timings[result_path.stem] = {
            field: float(values[field])
            for field in TIMING_FIELDS
            if field in values
        }
    return timings


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    """Return aggregate latency statistics in seconds."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def compare_runs(reference: str | Path, optimized: str | Path) -> dict[str, Any]:
    """Compare common result JSONs and calculate speedup percentages."""
    reference_values = load_timings(reference)
    optimized_values = load_timings(optimized)
    common = sorted(set(reference_values) & set(optimized_values))
    if not common:
        raise ValueError("reference and optimized directories have no common JSON results")

    metrics: dict[str, Any] = {
        "reference": str(_json_dir(reference)),
        "optimized": str(_json_dir(optimized)),
        "images": common,
        "missing_from_reference": sorted(set(optimized_values) - set(reference_values)),
        "missing_from_optimized": sorted(set(reference_values) - set(optimized_values)),
        "timings": {},
        "per_image_total": {},
    }
    for field in TIMING_FIELDS:
        reference_list = [reference_values[name][field] for name in common if field in reference_values[name] and field in optimized_values[name]]
        optimized_list = [optimized_values[name][field] for name in common if field in reference_values[name] and field in optimized_values[name]]
        if not reference_list:
            continue
        reference_summary = summarize(reference_list)
        optimized_summary = summarize(optimized_list)
        reference_mean = reference_summary["mean"]
        optimized_mean = optimized_summary["mean"]
        metrics["timings"][field] = {
            "reference": reference_summary,
            "optimized": optimized_summary,
            "mean_delta_seconds": optimized_mean - reference_mean,
            "mean_speedup_percent": (
                (reference_mean - optimized_mean) / reference_mean * 100.0
                if reference_mean
                else 0.0
            ),
            "mean_speedup_factor": optimized_mean and reference_mean / optimized_mean,
        }
    for name in common:
        if "total_pipeline_seconds" in reference_values[name] and "total_pipeline_seconds" in optimized_values[name]:
            reference_total = reference_values[name]["total_pipeline_seconds"]
            optimized_total = optimized_values[name]["total_pipeline_seconds"]
            metrics["per_image_total"][name] = {
                "reference_seconds": reference_total,
                "optimized_seconds": optimized_total,
                "speedup_percent": (
                    (reference_total - optimized_total) / reference_total * 100.0
                    if reference_total
                    else 0.0
                ),
            }
    return metrics


def _print_report(metrics: dict[str, Any]) -> None:
    print(f"Reference: {metrics['reference']}")
    print(f"Optimized: {metrics['optimized']}")
    print(f"Common images: {len(metrics['images'])}")
    if metrics["missing_from_reference"]:
        print("Missing from reference:", ", ".join(metrics["missing_from_reference"]))
    if metrics["missing_from_optimized"]:
        print("Missing from optimized:", ", ".join(metrics["missing_from_optimized"]))
    print("\nLatency and speedup metrics")
    print("metric | reference mean | optimized mean | delta seconds | speedup | factor")
    for field, values in metrics["timings"].items():
        print(
            f"{field} | {values['reference']['mean']:.3f}s | "
            f"{values['optimized']['mean']:.3f}s | "
            f"{values['mean_delta_seconds']:+.3f}s | "
            f"{values['mean_speedup_percent']:+.2f}% | "
            f"{values['mean_speedup_factor']:.3f}x"
        )
    print("\nTotal latency by image")
    for name, values in metrics["per_image_total"].items():
        print(
            f"{name} | reference={values['reference_seconds']:.3f}s | "
            f"optimized={values['optimized_seconds']:.3f}s | "
            f"speedup={values['speedup_percent']:+.2f}%"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference result directory or its outputs subdirectory")
    parser.add_argument("optimized", type=Path, help="Optimized result directory or its outputs subdirectory")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the report as JSON")
    args = parser.parse_args(argv)
    try:
        metrics = compare_runs(args.reference, args.optimized)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.as_json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        _print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
