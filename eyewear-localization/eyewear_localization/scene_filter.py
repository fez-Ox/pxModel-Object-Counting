"""L0 scene filtering for non-display eyewear detections.

The filter is deliberately separate from brand attribution.  It removes
instances supported by class-agnostic scene detections (people, posters, and
shelves) and preserves an audit record for every removal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from eyewear_localization.perception import LocalizationDetection, SAM3Localizer
from eyewear_localization.schemas import Instance, InstanceExclusion, PosterRegion, xywh_to_xyxy, xyxy_to_xywh


class SceneFilter(Protocol):
    name: str
    reliability: float

    def filter(
        self, image_path: Path, instances: list[Instance]
    ) -> tuple[list[Instance], list[InstanceExclusion]]: ...


def _center(box: list[float]) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _contains(box: list[float], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _matches(instance: Instance, detection: LocalizationDetection, iou: float) -> bool:
    target = xywh_to_xyxy(instance.bbox)
    reference = detection.box
    return _contains(reference, _center(instance.bbox)) or (
        iou > 0 and _iou(target, reference) >= iou
    )


class NullSceneFilter:
    name = "null"
    reliability = 0.1

    def __init__(self, reason: str = "scene filter unavailable") -> None:
        self.reason = reason

    def filter(
        self, image_path: Path, instances: list[Instance]
    ) -> tuple[list[Instance], list[InstanceExclusion]]:
        return list(instances), []


class SAM3SceneFilter:
    """Use the already-loaded class-agnostic SAM3 predictor for scene filters."""

    name = "sam3-scene-filter"
    reliability = 0.8

    def __init__(
        self,
        localizer: SAM3Localizer,
        *,
        require_shelf: bool = True,
        person_threshold: float = 0.25,
        poster_threshold: float = 0.25,
        shelf_threshold: float = 0.20,
    ) -> None:
        self.localizer = localizer
        self.require_shelf = require_shelf
        self.person_threshold = person_threshold
        self.poster_threshold = poster_threshold
        self.shelf_threshold = shelf_threshold
        self.last_poster_regions: list[PosterRegion] = []

    def _detect(self, image_path: Path) -> dict[str, list[LocalizationDetection]]:
        prompts = {
            "person": ("people", "person", "faces of people"),
            "poster": ("advertisements", "posters", "billboards"),
            "shelf": ("retail shelves", "display shelves", "shelf"),
        }
        thresholds = {
            "person": self.person_threshold,
            "poster": self.poster_threshold,
            "shelf": self.shelf_threshold,
        }
        detections: dict[str, list[LocalizationDetection]] = {
            "person": [], "poster": [], "shelf": []
        }
        for kind, candidates in prompts.items():
            per_prompt = self.localizer.detect_prompts(
                image_path,
                candidates,
                thresholds=[thresholds[kind]] * len(candidates),
            )
            for prompt_detections in per_prompt:
                detections[kind].extend(prompt_detections)
            # Cross-prompt duplicates are harmless for filtering but reducing
            # them keeps the audit support compact.
            unique: list[LocalizationDetection] = []
            for candidate in sorted(detections[kind], key=lambda item: item.score, reverse=True):
                if any(SAM3Localizer._iou(candidate.box, previous.box) >= 0.8 for previous in unique):
                    continue
                unique.append(candidate)
            detections[kind] = unique
        return detections

    def filter(
        self, image_path: Path, instances: list[Instance]
    ) -> tuple[list[Instance], list[InstanceExclusion]]:
        detections = self._detect(image_path)
        self.last_poster_regions = [
            PosterRegion(
                xyxy_to_xywh(item.box),
                item.score,
                source=f"sam3:{item.prompt}",
            )
            for item in detections["poster"]
        ]
        shelf_boxes = detections["shelf"]
        kept: list[Instance] = []
        excluded: list[InstanceExclusion] = []
        for instance in instances:
            reasons: list[str] = []
            support: dict[str, list[dict[str, object]]] = {}
            for kind, reason, overlap_iou in (
                ("person", "worn_or_on_person", 0.02),
                ("poster", "inside_advertisement", 0.05),
            ):
                matches = [
                    detection for detection in detections[kind]
                    if _matches(instance, detection, overlap_iou)
                ]
                if matches:
                    reasons.append(reason)
                    support[kind] = [
                        {"prompt": item.prompt, "box": item.box, "score": item.score}
                        for item in matches
                    ]

            if self.require_shelf and shelf_boxes:
                shelf_matches = [
                    detection for detection in shelf_boxes if _matches(instance, detection, 0.01)
                ]
                if not shelf_matches:
                    reasons.append("not_on_detected_shelf")
                else:
                    support["shelf"] = [
                        {"prompt": item.prompt, "box": item.box, "score": item.score}
                        for item in shelf_matches
                    ]

            if reasons:
                excluded.append(
                    InstanceExclusion(
                        instance_id=instance.id,
                        bbox=instance.bbox,
                        reasons=reasons,
                        support=support,
                    )
                )
            else:
                kept.append(instance)
        return kept, excluded


def build_scene_filter(
    localizer: object,
    *,
    require_shelf: bool = True,
    person_threshold: float = 0.25,
    poster_threshold: float = 0.25,
    shelf_threshold: float = 0.20,
) -> SceneFilter:
    if isinstance(localizer, SAM3Localizer):
        return SAM3SceneFilter(
            localizer,
            require_shelf=require_shelf,
            person_threshold=person_threshold,
            poster_threshold=poster_threshold,
            shelf_threshold=shelf_threshold,
        )
    return NullSceneFilter("class-agnostic SAM3 scene filter unavailable")
