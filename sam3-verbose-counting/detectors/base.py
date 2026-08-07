"""Pluggable detection-task interface.

Every concrete detection (e.g. sunglasses) lives in its own module under
``detectors/`` and is **completely decoupled** from every other detection. A
task only depends on the detection-agnostic core in ``infer.py`` (the
:class:`Sam3VerboseCounter`). Adding a new detection in the future means adding
a new ``detectors/<name>.py`` module and registering it — no changes to the
sunglasses task (or any other task) are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectionOptions:
    """Detection-agnostic SAM3 inference options shared by all tasks.

    Each field maps 1:1 onto :meth:`infer.Sam3VerboseCounter.infer` parameters
    so tasks can forward them unchanged while overriding only what they own.
    """

    prompt: str | None = None
    filter_prompt: str | None = None
    filter_center: bool = True
    filter_iou: float = 0.0
    box_cleanup: bool = True
    box_duplicate_iou: float = 0.9
    box_min_children: int = 2
    box_min_area_ratio: float = 1.25


class DetectionTask:
    """Base class for a concrete object-detection task.

    Subclasses declare a stable ``name`` plus sensible defaults and implement
    :meth:`run`, which translates the task's concept into a call on the shared
    SAM3 counter. Tasks must not depend on other tasks.
    """

    name: str = ""
    description: str = ""
    default_prompt: str = ""
    default_filter_prompt: str | None = None

    def run(self, counter: Any, image_path: Path, options: DetectionOptions) -> dict:
        """Run this task on one image and return the standard result dict."""
        raise NotImplementedError

    def resolve_prompt(self, options: DetectionOptions) -> str:
        """Return the effective prompt (explicit override or task default)."""
        return (options.prompt or self.default_prompt).strip()
