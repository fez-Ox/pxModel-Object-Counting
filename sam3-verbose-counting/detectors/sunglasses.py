"""Sunglasses detection task.

Counts pairs of sunglasses displayed on a retail rack, excluding pairs *worn*
on a person. This is the project's first concrete detection and is fully
decoupled from any future detection: it only wraps the shared SAM3 counter
(:class:`infer.Sam3VerboseCounter`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detectors.base import DetectionOptions, DetectionTask


class SunglassesTask(DetectionTask):
    name = "sunglasses"
    description = (
        "Count sunglasses displayed on a retail rack, excluding pairs worn on people."
    )
    default_prompt = "all pairs of black sunglasses displayed on the retail rack"
    default_filter_prompt = "faces of people"

    def run(self, counter: Any, image_path: Path, options: DetectionOptions) -> dict:
        return counter.infer(
            image_path,
            self.resolve_prompt(options),
            filter_prompt=options.filter_prompt or self.default_filter_prompt,
            filter_center=options.filter_center,
            filter_iou=options.filter_iou,
            box_cleanup=options.box_cleanup,
            box_duplicate_iou=options.box_duplicate_iou,
            box_min_children=options.box_min_children,
            box_min_area_ratio=options.box_min_area_ratio,
        )
