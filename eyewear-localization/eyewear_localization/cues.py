"""Cue implementations C1, C2, and C4.

Cues only emit evidence.  They never decide an instance's final brand; that
responsibility belongs to :mod:`eyewear_localization.fusion`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any, Callable, Protocol

from eyewear_localization.gazetteer import Gazetteer
from eyewear_localization.schemas import Evidence, Instance, PosterRegion, Scope, Sign, TextDetection


def _xyxy(box: list[float]) -> list[float]:
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _xywh(box: list[float]) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def _area(box: list[float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _contains(box: list[float], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]


def _iou(left: list[float], right: list[float]) -> float:
    left = _xyxy(left)
    right = _xyxy(right)
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = _area(_xywh(left)) + _area(_xywh(right)) - intersection
    return intersection / union if union else 0.0


def _union(instances: list[Instance]) -> list[float] | None:
    if not instances:
        return None
    boxes = [_xyxy(instance.bbox) for instance in instances]
    return _xywh([
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ])


class OnProductBrandingCue:
    """C1: crop each instance and run fine-text OCR plus gazetteer matching."""

    name = "C1"

    def __init__(
        self,
        ocr: Any,
        gazetteer: Gazetteer,
        *,
        margin: float = 0.15,
        scale: float = 3.0,
    ) -> None:
        self.ocr = ocr
        self.gazetteer = gazetteer
        self.margin = max(0.0, float(margin))
        self.scale = max(1.0, float(scale))

    @staticmethod
    def _load(image_path: str | Path):
        from PIL import Image

        with Image.open(image_path) as image:
            return image.convert("RGB")

    def emit(self, image_path: str | Path, instances: list[Instance]) -> list[Evidence]:
        image = self._load(image_path)
        width, height = image.size
        evidence: list[Evidence] = []
        for instance in instances:
            x, y, box_width, box_height = instance.bbox
            margin_x = box_width * self.margin
            margin_y = box_height * self.margin
            left = max(0.0, x - margin_x)
            top = max(0.0, y - margin_y)
            right = min(float(width), x + box_width + margin_x)
            bottom = min(float(height), y + box_height + margin_y)
            crop = image.crop((round(left), round(top), round(right), round(bottom)))
            crop_width, crop_height = crop.size
            if self.scale != 1.0:
                crop = crop.resize(
                    (max(1, round(crop_width * self.scale)), max(1, round(crop_height * self.scale)))
                )
            try:
                detections: list[TextDetection] = self.ocr.detect(crop)
            except Exception:
                detections = []
            for detection in detections:
                match = self.gazetteer.match(detection.text)
                if match is None:
                    continue
                local_x, local_y, local_w, local_h = detection.bbox
                global_box = [
                    left + local_x / self.scale,
                    top + local_y / self.scale,
                    local_w / self.scale,
                    local_h / self.scale,
                ]
                evidence.append(
                    Evidence(
                        instance_id=instance.id,
                        brand=match.brand,
                        confidence=max(0.0, min(1.0, detection.confidence * match.score)),
                        cue=self.name,
                        support={
                            "raw_text": detection.text,
                            "text_bbox": global_box,
                            "crop_bbox": list(instance.bbox),
                            "match_method": match.method,
                            "ocr_confidence": detection.confidence,
                        },
                    )
                )
        return evidence


class SignageScopeCue:
    """C2: infer sign scope from geometry without assuming a layout family."""

    name = "C2"

    def __init__(self, *, min_scope_confidence: float = 0.2) -> None:
        self.min_scope_confidence = float(min_scope_confidence)

    @staticmethod
    def _poster_contains_sign(sign: Sign, posters: list[PosterRegion]) -> bool:
        center = _center(sign.bbox)
        return any(_contains(poster.bbox, center) or _iou(sign.bbox, poster.bbox) >= 0.1 for poster in posters)

    @staticmethod
    def _candidate_items(
        sign: Sign,
        instances: list[Instance],
        *,
        kind: str,
        median_width: float,
        median_height: float,
        all_signs: list[Sign],
    ) -> list[Instance]:
        sx, sy, sw, sh = sign.bbox
        scx, scy = _center(sign.bbox)
        if kind == "bay_header":
            below = [
                instance
                for instance in instances
                if (instance.centroid or _center(instance.bbox))[1]
                >= sy + sh - median_height * 0.25
            ]
            if not below:
                return []

            # If several signs share a header band, use their midpoints as
            # inferred bay boundaries. This handles narrow OCR boxes above a
            # wide column without assigning the neighboring bay.
            sibling_centers = sorted(
                _center(other.bbox)[0]
                for other in all_signs
                if other.sign_id != sign.sign_id
                and abs(_center(other.bbox)[1] - scy) <= max(2.0 * median_height, 20.0)
            )
            if sibling_centers:
                left_boundary = max(
                    (center + scx) / 2.0 for center in sibling_centers if center < scx
                ) if any(center < scx for center in sibling_centers) else float("-inf")
                right_boundary = min(
                    (center + scx) / 2.0 for center in sibling_centers if center > scx
                ) if any(center > scx for center in sibling_centers) else float("inf")
                bounded = [
                    instance
                    for instance in below
                    if left_boundary <= (instance.centroid or _center(instance.bbox))[0] <= right_boundary
                ]
                if bounded:
                    return bounded

            # With one header (or an empty inferred bay), select the nearest
            # horizontal display group. The gap threshold is learned from
            # detected object width rather than assuming a fixed number of
            # columns or rows.
            ordered = sorted(below, key=lambda item: (item.centroid or _center(item.bbox))[0])
            groups: list[list[Instance]] = [[]]
            max_gap = max(5.0 * median_width, 2.0 * sw, 1.0)
            for item in ordered:
                center_x = (item.centroid or _center(item.bbox))[0]
                if groups[-1]:
                    previous_x = (groups[-1][-1].centroid or _center(groups[-1][-1].bbox))[0]
                    if center_x - previous_x > max_gap:
                        groups.append([])
                groups[-1].append(item)
            return min(
                groups,
                key=lambda group: min(
                    abs((item.centroid or _center(item.bbox))[0] - scx) for item in group
                ),
            )

        output: list[Instance] = []
        for instance in instances:
            ix, iy, iw, ih = instance.bbox
            icx, icy = instance.centroid or _center(instance.bbox)
            horizontal_distance = abs(icx - scx)
            x_overlap = max(0.0, min(sx + sw, ix + iw) - max(sx, ix))
            if kind == "row_label":
                same_row = abs(icy - scy) <= max(1.5 * median_height, sh * 2.0)
                row_items = [
                    other for other in instances
                    if abs((other.centroid or _center(other.bbox))[1] - scy)
                    <= max(1.5 * median_height, sh * 2.0)
                ]
                row_left = min((other.bbox[0] for other in row_items), default=ix)
                row_right = max((other.bbox[0] + other.bbox[2] for other in row_items), default=ix + iw)
                sign_is_row_end = sx >= row_right or sx + sw <= row_left
                near_end = sign_is_row_end or horizontal_distance <= max(4.0 * median_width, sw * 3.0)
                if same_row and near_end:
                    output.append(instance)
            elif kind == "shelf_edge_tag":
                close_y = abs(icy - scy) <= max(2.5 * median_height, sh * 3.0)
                attached = x_overlap > 0 or horizontal_distance <= max(1.5 * median_width, sw)
                if close_y and attached:
                    output.append(instance)
            elif kind == "hanging_sign":
                if icy > sy + sh and (x_overlap > 0 or horizontal_distance <= 4.0 * median_width):
                    output.append(instance)
        return output

    @staticmethod
    def _hypothesis_score(kind: str, sign: Sign, items: list[Instance], median_height: float) -> float:
        if not items:
            return 0.0
        region = _union(items)
        if region is None:
            return 0.0
        sign_center = _center(sign.bbox)
        region_center = _center(region)
        alignment = 1.0 if region[0] <= sign_center[0] <= region[0] + region[2] else 0.55
        if kind in {"bay_header", "hanging_sign"}:
            gap = max(0.0, region[1] - (sign.bbox[1] + sign.bbox[3]))
        else:
            gap = abs(region_center[1] - sign_center[1])
        attachment = max(0.0, 1.0 - gap / max(4.0 * median_height, 1.0))
        density = min(1.0, len(items) / 3.0)
        priors = {
            "bay_header": 0.10,
            "row_label": 0.08,
            "shelf_edge_tag": 0.06,
            "hanging_sign": 0.02,
        }
        return min(0.99, priors[kind] + 0.38 * alignment + 0.30 * attachment + 0.22 * density)

    def associate(
        self,
        instances: list[Instance],
        signs: list[Sign],
        posters: list[PosterRegion],
    ) -> tuple[list[Sign], list[Evidence]]:
        if instances:
            median_width = max(1.0, median(instance.bbox[2] for instance in instances))
            median_height = max(1.0, median(instance.bbox[3] for instance in instances))
        else:
            median_width = median_height = 1.0

        updated: list[Sign] = []
        evidence: list[Evidence] = []
        kinds = ("bay_header", "row_label", "shelf_edge_tag", "hanging_sign")
        for sign in signs:
            if self._poster_contains_sign(sign, posters):
                scoped = replace(sign, scope=Scope("none", None, 0.99))
                updated.append(scoped)
                continue

            best_kind = "none"
            best_items: list[Instance] = []
            best_score = 0.0
            for kind in kinds:
                items = self._candidate_items(
                    sign,
                    instances,
                    kind=kind,
                    median_width=median_width,
                    median_height=median_height,
                    all_signs=signs,
                )
                score = self._hypothesis_score(kind, sign, items, median_height)
                if score > best_score:
                    best_kind, best_items, best_score = kind, items, score

            if best_score < self.min_scope_confidence or not best_items:
                scoped = replace(sign, scope=Scope("none", None, max(0.0, best_score)))
                updated.append(scoped)
                continue

            region = _union(best_items)
            scope = Scope(best_kind, region, best_score)
            scoped = replace(sign, scope=scope)
            updated.append(scoped)
            for instance in best_items:
                if not _contains(region or [], tuple(instance.centroid or _center(instance.bbox))):
                    continue
                if not sign.brand:
                    continue
                evidence.append(
                    Evidence(
                        instance_id=instance.id,
                        brand=sign.brand,
                        confidence=max(0.0, min(1.0, sign.confidence * scope.confidence)),
                        cue=self.name,
                        support={
                            "sign_id": sign.sign_id,
                            "sign_text": sign.text,
                            "sign_bbox": list(sign.bbox),
                            "scope": scope.to_dict(),
                        },
                    )
                )
        return updated, evidence

    def emit(
        self,
        instances: list[Instance],
        signs: list[Sign],
        posters: list[PosterRegion],
    ) -> list[Evidence]:
        """Convenience API when callers do not need updated sign scopes."""
        _, evidence = self.associate(instances, signs, posters)
        return evidence


class StylePriorCue:
    """C4 adapter; a missing gallery is a safe no-op rather than a fake label."""

    name = "C4"

    def __init__(
        self,
        matcher: Callable[[str | Path, Instance], tuple[str, float] | None] | None = None,
    ) -> None:
        self.matcher = matcher

    def emit(self, image_path: str | Path, instances: list[Instance]) -> list[Evidence]:
        if self.matcher is None:
            return []
        evidence: list[Evidence] = []
        for instance in instances:
            result = self.matcher(image_path, instance)
            if result is None:
                continue
            brand, confidence = result
            evidence.append(
                Evidence(
                    instance_id=instance.id,
                    brand=brand,
                    confidence=confidence,
                    cue=self.name,
                    support={"source": "style_gallery"},
                )
            )
        return evidence


class NullAuditor:
    """L3 fallback used when no VLM auditor is configured."""

    name = "null"
    reliability = 0.0

    def emit(self, image_path: str | Path, instances: list[Instance], candidates: list[Any]) -> list[Evidence]:
        return []
