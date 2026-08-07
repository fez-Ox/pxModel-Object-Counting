"""Detection-task registry.

Registering a task here makes it selectable from the CLI (``--detector``) and
the library API. Adding a future detection = drop a new module in
``detectors/``, subclass :class:`DetectionTask`, and decorate it with
:func:`register` — no code in existing tasks (or the sunglasses task) needs to
change.
"""

from __future__ import annotations

from detectors.base import DetectionOptions, DetectionTask
from detectors.sunglasses import SunglassesTask

_TASKS: dict[str, type[DetectionTask]] = {}


def register(cls: type[DetectionTask]) -> type[DetectionTask]:
    """Register a :class:`DetectionTask` subclass by its ``name``."""
    if not cls.name:
        raise ValueError(f"detection task {cls.__name__} must declare a non-empty name")
    if cls.name in _TASKS:
        raise ValueError(f"duplicate detection task name: {cls.name!r}")
    _TASKS[cls.name] = cls
    return cls


register(SunglassesTask)


def get_detector(name: str) -> DetectionTask:
    """Return a fresh instance of the named detection task."""
    try:
        return _TASKS[name]()
    except KeyError:
        raise KeyError(
            f"unknown detection task {name!r}; available: {', '.join(sorted(_TASKS))}"
        ) from None


def list_detectors() -> list[str]:
    """Return the names of all registered detection tasks."""
    return sorted(_TASKS)


__all__ = [
    "DetectionOptions",
    "DetectionTask",
    "get_detector",
    "list_detectors",
    "register",
]
