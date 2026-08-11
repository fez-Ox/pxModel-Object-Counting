#!/usr/bin/env python3
"""Validate pipeline output JSONs against dataset_ground_truth.json.

Per image, counts final_brand assignments from <results>/json/<stem>.json and
compares the brand breakdown against the ground-truth brand_breakdown. A brand
present in the output but absent from ground truth (or vice versa), or a count
mismatch, marks the image as FAIL.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_brand_breakdown(result_path: Path) -> Counter:
    data = json.loads(result_path.read_text())
    outputs = data.get("outputs", data.get("instances", []))
    counts: Counter = Counter()
    for item in outputs:
        brand = item.get("final_brand") or item.get("brand")
        if brand and brand != "unknown":
            counts[brand] += 1
    return counts


def validate(results_dir: Path, ground_truth: Path) -> int:
    gt = json.loads(ground_truth.read_text())
    gt_images: dict = gt["images"]
    image_keys = {stem: key for key in gt_images for stem in [Path(key).stem]}

    result_files = sorted(results_dir.glob("*.json"))
    result_files = [p for p in result_files if p.stem != "dataset-metadata"]
    failures = 0
    passes = 0
    print(f"{'Image':<12} | {'Result':<6} | {'Expected':<40} | {'Actual':<40}")
    print("-" * 110)

    for result_path in result_files:
        stem = result_path.stem
        gt_key = image_keys.get(stem)
        if gt_key is None:
            print(f"{stem:<12} | {'SKIP':<6} | (not in ground truth)")
            continue
        expected = Counter(gt_images[gt_key].get("brand_breakdown", {}))
        actual = load_brand_breakdown(result_path)
        ok = expected == actual
        expected_str = dict(sorted(expected.items())) if expected else "{}"
        actual_str = dict(sorted(actual.items())) if actual else "{}"
        status = "PASS" if ok else "FAIL"
        print(f"{stem:<12} | {status:<6} | {expected_str} | {actual_str}")
        if ok:
            passes += 1
        else:
            failures += 1

    print("-" * 110)
    print(f"Total: {passes} passed, {failures} failed, {len(result_files)} checked")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate brand-count attribution against ground truth.")
    parser.add_argument("--results-dir", default="output/json",
                        help="Directory containing per-image result JSONs (default: output/json)")
    parser.add_argument("--ground-truth", default="eyewear-localization/dataset_ground_truth.json",
                        help="Path to ground-truth JSON (default: eyewear-localization/dataset_ground_truth.json)")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    ground_truth = Path(args.ground_truth)
    if not results_dir.exists():
        print(f"error: results dir not found: {results_dir}", file=sys.stderr)
        return 2
    if not ground_truth.exists():
        print(f"error: ground truth not found: {ground_truth}", file=sys.stderr)
        return 2
    return validate(results_dir, ground_truth)


if __name__ == "__main__":
    sys.exit(main())
