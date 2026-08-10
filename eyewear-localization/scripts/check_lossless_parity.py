#!/usr/bin/env python3
"""Compare reference and optimized pipeline JSON without trusting final labels only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


_IGNORED_TOP_LEVEL = {"timings"}


def _comparison_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Remove execution metadata while retaining every result-bearing field."""
    payload = copy.deepcopy(value)
    for key in _IGNORED_TOP_LEVEL:
        payload.pop(key, None)
    effective = payload.get("effective_config")
    if isinstance(effective, dict):
        # The optimized run necessarily records different batch/compiler knobs;
        # those are execution metadata, not attribution output.
        effective.pop("performance", None)
    return payload


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: types differ ({type(left).__name__} != {type(right).__name__})"
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            child = f"{path}.{key}"
            if key not in left:
                return f"{child}: missing from reference"
            if key not in right:
                return f"{child}: missing from optimized result"
            difference = _first_difference(left[key], right[key], child)
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: list lengths differ ({len(left)} != {len(right)})"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def compare(reference_path: str | Path, optimized_path: str | Path) -> tuple[bool, str | None]:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    optimized = json.loads(Path(optimized_path).read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or not isinstance(optimized, dict):
        raise ValueError("both result files must contain a JSON object")
    reference = _comparison_payload(reference)
    optimized = _comparison_payload(optimized)
    difference = _first_difference(reference, optimized, "$")
    return difference is None, difference


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require full result parity between reference and optimized JSON outputs."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("optimized", type=Path)
    args = parser.parse_args()
    equal, difference = compare(args.reference, args.optimized)
    if equal:
        print("PARITY_OK: all result-bearing fields match")
        return 0
    print(f"PARITY_FAILED: {difference}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
