"""Cue implementations C1, C2, and C4.

Cues only emit evidence.  They never decide an instance's final brand; that
responsibility belongs to :mod:`eyewear_localization.fusion`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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


# ---------------------------------------------------------------------------
# C1 — On-product branding (crop → upscale → sharpen → multi-scale OCR)
# ---------------------------------------------------------------------------

class OnProductBrandingCue:
    """C1: crop each instance and run fine-text OCR plus gazetteer matching.

    Improvements over the baseline:
    * Larger crop margin (0.25) to capture temple arms beyond the bbox.
    * Post-upscale sharpening to reduce blur artifacts that hurt OCR.
    * Multi-scale OCR: runs at each configured scale and keeps the best
      gazetteer-matching detection per instance.
    * Paragraph-mode second pass to merge fragmented partial detections
      on temples (e.g. "CAR" + "TIER" → "CARTIER").
    """

    name = "C1"

    def __init__(
        self,
        ocr: Any,
        gazetteer: Gazetteer,
        *,
        margin: float = 0.25,
        scales: tuple[float, ...] | list[float] = (2.0, 4.0),
        sharpen: bool = True,
    ) -> None:
        self.ocr = ocr
        self.gazetteer = gazetteer
        self.margin = max(0.0, float(margin))
        self.scales = tuple(max(1.0, float(s)) for s in scales) if scales else (2.0,)
        self.sharpen = sharpen

    @staticmethod
    def _load(image_path: str | Path):
        from PIL import Image

        with Image.open(image_path) as image:
            return image.convert("RGB")

    @staticmethod
    def _sharpen(image: Any) -> Any:
        """Apply UnsharpMask to recover detail after upscale interpolation."""
        from PIL import ImageFilter

        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    def _ocr_at_scale(self, crop: Any, scale: float) -> list[TextDetection]:
        """Upscale, optionally sharpen, and run OCR at one scale."""
        crop_width, crop_height = crop.size
        if scale != 1.0:
            working = crop.resize(
                (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
            )
        else:
            working = crop
        if self.sharpen and scale > 1.0:
            working = self._sharpen(working)
        try:
            detections: list[TextDetection] = self.ocr.detect(working)
        except Exception:
            detections = []
        # Rescale detection bboxes back to crop coordinates.
        rescaled: list[TextDetection] = []
        for detection in detections:
            local_x, local_y, local_w, local_h = detection.bbox
            rescaled.append(TextDetection(
                text=detection.text,
                bbox=[local_x / scale, local_y / scale, local_w / scale, local_h / scale],
                confidence=detection.confidence,
                source=detection.source,
            ))
        return rescaled

    def _ocr_paragraph_pass(self, crop: Any, scale: float) -> list[TextDetection]:
        """Run OCR with paragraph=True to merge fragmented text segments."""
        import numpy as np

        crop_width, crop_height = crop.size
        if scale != 1.0:
            working = crop.resize(
                (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
            )
        else:
            working = crop
        if self.sharpen and scale > 1.0:
            working = self._sharpen(working)
        # Some OCR backends may not support paragraph mode; guard gracefully.
        reader = getattr(self.ocr, "reader", None)
        if reader is None:
            return []
        try:
            raw = reader.readtext(np.asarray(working), detail=1, paragraph=True, mag_ratio=1.0)
        except Exception:
            return []
        output: list[TextDetection] = []
        for item in raw:
            if len(item) < 3:
                continue
            polygon, text, confidence = item[0], str(item[1]), float(item[2])
            xs = [float(pt[0]) / scale for pt in polygon]
            ys = [float(pt[1]) / scale for pt in polygon]
            try:
                output.append(TextDetection(
                    text=text,
                    bbox=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    confidence=max(0.0, min(1.0, confidence)),
                    source="easyocr-paragraph",
                ))
            except ValueError:
                continue
        return output

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

            # Collect detections across all configured scales and paragraph pass.
            all_detections: list[tuple[TextDetection, float]] = []
            for scale in self.scales:
                for detection in self._ocr_at_scale(crop, scale):
                    all_detections.append((detection, scale))
            # Paragraph pass at the largest scale to catch merged fragments.
            if self.scales:
                best_scale = max(self.scales)
                for detection in self._ocr_paragraph_pass(crop, best_scale):
                    all_detections.append((detection, best_scale))

            # For each brand, keep only the highest-confidence match across
            # all scales and modes. This avoids duplicate evidence while
            # allowing different scales to succeed on different text.
            best_per_brand: dict[str, tuple[Evidence, float]] = {}
            for detection, scale in all_detections:
                match = self.gazetteer.match(detection.text)
                if match is None:
                    continue
                local_x, local_y, local_w, local_h = detection.bbox
                global_box = [
                    left + local_x,
                    top + local_y,
                    local_w,
                    local_h,
                ]
                combined_confidence = max(0.0, min(1.0, detection.confidence * match.score))
                ev = Evidence(
                    instance_id=instance.id,
                    brand=match.brand,
                    confidence=combined_confidence,
                    cue=self.name,
                    support={
                        "raw_text": detection.text,
                        "text_bbox": global_box,
                        "crop_bbox": list(instance.bbox),
                        "match_method": match.method,
                        "ocr_confidence": detection.confidence,
                        "scale_used": scale,
                    },
                )
                existing = best_per_brand.get(match.brand)
                if existing is None or combined_confidence > existing[1]:
                    best_per_brand[match.brand] = (ev, combined_confidence)
            evidence.extend(ev for ev, _ in best_per_brand.values())
        return evidence


# ---------------------------------------------------------------------------
# C2 — Signage scope (column-boundary-first approach)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Column:
    """An inferred display column with brand and x-boundaries."""
    brand: str | None
    left: float
    right: float
    sign: Sign
    confidence: float


def _infer_columns(
    header_signs: list[Sign],
    image_width: float | None = None,
    median_width: float = 20.0,
) -> list[_Column]:
    """Infer display columns from header-band signs.

    Column boundaries are placed at the midpoint of adjacent sign *edges*
    (not centers), which handles signs of different widths correctly.
    Adjacent signs for the same brand are merged into one wider column.
    """
    if not header_signs:
        return []
    sorted_signs = sorted(header_signs, key=lambda s: s.bbox[0])

    # --- Build raw columns from individual signs ---
    raw: list[_Column] = []
    for i, sign in enumerate(sorted_signs):
        # Left boundary: midpoint between this sign's left edge and the
        # previous sign's right edge.
        if i == 0:
            left = 0.0
        else:
            prev = sorted_signs[i - 1]
            prev_right_edge = prev.bbox[0] + prev.bbox[2]
            this_left_edge = sign.bbox[0]
            left = (prev_right_edge + this_left_edge) / 2.0
        # Right boundary: midpoint between this sign's right edge and the
        # next sign's left edge.
        if i == len(sorted_signs) - 1:
            if image_width is not None:
                right = image_width
            elif len(sorted_signs) == 1:
                right = sign.bbox[0] + sign.bbox[2] + max(8.0 * median_width, 200.0)
            else:
                right = sign.bbox[0] + sign.bbox[2] + 500
        else:
            next_sign = sorted_signs[i + 1]
            this_right_edge = sign.bbox[0] + sign.bbox[2]
            next_left_edge = next_sign.bbox[0]
            right = (this_right_edge + next_left_edge) / 2.0
        raw.append(_Column(brand=sign.brand, left=left, right=right, sign=sign, confidence=sign.confidence))

    # --- Merge adjacent columns with the same brand ---
    merged: list[_Column] = [raw[0]]
    for col in raw[1:]:
        prev = merged[-1]
        if col.brand and prev.brand and col.brand == prev.brand:
            # Merge: extend the previous column to cover this one.
            merged[-1] = _Column(
                brand=prev.brand,
                left=prev.left,
                right=col.right,
                sign=prev.sign,  # keep the first sign as representative
                confidence=max(prev.confidence, col.confidence),
            )
        else:
            merged.append(col)
    return merged


class SignageScopeCue:
    """C2: infer sign scope from geometry using a column-boundary-first approach.

    The primary strategy is header-column inference: signs that share a
    horizontal header band are used to derive vertical column boundaries.
    Each eyewear instance is assigned to the column whose x-range contains
    its centroid.  This approach avoids the convoluted multi-hypothesis scoring
    of the original implementation.

    Non-header signs (row labels, shelf-edge tags) are handled by simpler
    proximity logic as a fallback.
    """

    name = "C2"

    def __init__(self, *, min_scope_confidence: float = 0.2) -> None:
        self.min_scope_confidence = float(min_scope_confidence)

    @staticmethod
    def _poster_contains_sign(sign: Sign, posters: list[PosterRegion]) -> bool:
        center = _center(sign.bbox)
        return any(_contains(poster.bbox, center) or _iou(sign.bbox, poster.bbox) >= 0.1 for poster in posters)

    @staticmethod
    def _find_header_band(
        signs: list[Sign],
        instances: list[Instance],
        median_height: float,
    ) -> tuple[list[Sign], list[Sign]]:
        """Partition signs into header-band signs and remaining signs.

        Header signs are those near the topmost sign row (within
        2× median instance height of the minimum sign y-center) AND
        whose bottom edge is above the instance row.
        """
        if not signs:
            return [], []
        min_y_center = min(_center(s.bbox)[1] for s in signs)
        band_threshold = max(2.0 * median_height, 40.0)
        min_inst_y = min((inst.bbox[1] for inst in instances), default=float("inf")) if instances else float("inf")

        headers: list[Sign] = []
        others: list[Sign] = []
        for sign in signs:
            sign_y_center = _center(sign.bbox)[1]
            sign_bottom = sign.bbox[1] + sign.bbox[3]
            is_above = sign_bottom <= min_inst_y + median_height * 0.5
            if abs(sign_y_center - min_y_center) <= band_threshold and is_above:
                headers.append(sign)
            else:
                others.append(sign)
        return headers, others

    @staticmethod
    def _assign_to_column(
        instance: Instance,
        columns: list[_Column],
    ) -> _Column | None:
        """Return the column whose x-range contains the instance centroid."""
        cx = (instance.centroid or _center(instance.bbox))[0]
        for col in columns:
            if col.left <= cx <= col.right:
                return col
        # Fallback: find the nearest column if within reasonable distance (<= 8x median width)
        if columns:
            nearest = min(columns, key=lambda c: abs((c.left + c.right) / 2.0 - cx))
            dist = max(0.0, nearest.left - cx) if cx < nearest.left else max(0.0, cx - nearest.right)
            if dist <= 80.0:
                return nearest
        return None

    @staticmethod
    def _column_containment_confidence(
        instance: Instance,
        column: _Column,
    ) -> float:
        """Score how well an instance fits within a column.

        Higher when the instance centroid is well within the column boundaries
        (not near the edges) and the column boundaries are wide relative to
        the instance.
        """
        cx = (instance.centroid or _center(instance.bbox))[0]
        col_width = max(1.0, column.right - column.left)
        # Fraction of the way from the nearest edge (0 = at edge, 0.5 = centered).
        dist_from_left = cx - column.left
        dist_from_right = column.right - cx
        margin_fraction = min(dist_from_left, dist_from_right) / (col_width / 2.0)
        margin_fraction = max(0.0, min(1.0, margin_fraction))
        # Base confidence from the sign OCR confidence, modulated by position.
        return max(0.0, min(0.95, column.confidence * (0.5 + 0.5 * margin_fraction)))

    @staticmethod
    def _proximity_scope(
        sign: Sign,
        instances: list[Instance],
        median_width: float,
        median_height: float,
    ) -> tuple[str, list[Instance], float]:
        """Simple proximity-based scope for non-header signs (row labels, shelf-edge tags)."""
        sx, sy, sw, sh = sign.bbox
        scx, scy = _center(sign.bbox)

        # Row label: sign is on the same horizontal band as nearby instances.
        row_items = [
            inst for inst in instances
            if abs((inst.centroid or _center(inst.bbox))[1] - scy) <= max(1.5 * median_height, sh * 2.0)
        ]
        if row_items:
            # Only include items reasonably close horizontally.
            close_row_items = [
                inst for inst in row_items
                if abs((inst.centroid or _center(inst.bbox))[0] - scx) <= max(6.0 * median_width, sw * 4.0)
            ]
            if close_row_items:
                return "row_label", close_row_items, min(0.75, sign.confidence * 0.8)

        # Shelf-edge tag: sign is vertically close and horizontally attached.
        shelf_items = [
            inst for inst in instances
            if (
                abs((inst.centroid or _center(inst.bbox))[1] - scy) <= max(2.5 * median_height, sh * 3.0)
                and abs((inst.centroid or _center(inst.bbox))[0] - scx) <= max(2.0 * median_width, sw * 1.5)
            )
        ]
        if shelf_items:
            return "shelf_edge_tag", shelf_items, min(0.65, sign.confidence * 0.7)

        return "none", [], 0.0

    def associate(
        self,
        instances: list[Instance],
        signs: list[Sign],
        posters: list[PosterRegion],
        *,
        image_width: float | None = None,
    ) -> tuple[list[Sign], list[Evidence]]:
        if instances:
            median_width = max(1.0, median(instance.bbox[2] for instance in instances))
            median_height = max(1.0, median(instance.bbox[3] for instance in instances))
        else:
            median_width = median_height = 1.0

        # Filter out poster-contained signs first.
        active_signs: list[Sign] = []
        updated: list[Sign] = []
        for sign in signs:
            if self._poster_contains_sign(sign, posters):
                scoped = replace(sign, scope=Scope("none", None, 0.99))
                updated.append(scoped)
            else:
                active_signs.append(sign)

        # Step 1: Identify the header band and infer column boundaries.
        header_signs, non_header_signs = self._find_header_band(active_signs, instances, median_height)

        # Only instances below the header band are candidates for column assignment.
        if header_signs:
            header_bottom = max(s.bbox[1] + s.bbox[3] for s in header_signs)
            below_instances = [
                inst for inst in instances
                if (inst.centroid or _center(inst.bbox))[1] >= header_bottom - median_height * 0.25
            ]
        else:
            below_instances = list(instances)

        columns = _infer_columns(header_signs, image_width, median_width)

        # Step 2: Assign instances to columns and emit evidence for header signs.
        # Deduplicate: emit evidence only once per (instance, brand) pair,
        # even when multiple header signs share the same brand.
        evidence: list[Evidence] = []
        instance_assigned: set[str] = set()  # track which instances got C2 evidence
        emitted_pairs: set[tuple[str, str]] = set()  # (instance_id, brand)

        for sign in header_signs:
            if not sign.brand:
                scoped = replace(sign, scope=Scope("none", None, 0.0))
                updated.append(scoped)
                continue

            # Find all columns for this sign's brand (handles merged columns).
            sign_columns = [c for c in columns if c.brand == sign.brand]
            if not sign_columns:
                scoped = replace(sign, scope=Scope("none", None, 0.0))
                updated.append(scoped)
                continue

            # Collect instances that fall within any of this brand's columns.
            column_instances: list[Instance] = []
            for inst in below_instances:
                assigned_col = self._assign_to_column(inst, columns)
                if assigned_col is not None and assigned_col.brand == sign.brand:
                    column_instances.append(inst)

            if not column_instances:
                scoped = replace(sign, scope=Scope("bay_header", None, 0.0))
                updated.append(scoped)
                continue

            region = _union(column_instances)
            # The scope region is the union of all matching columns.
            col_left = min(c.left for c in sign_columns)
            col_right = max(c.right for c in sign_columns)
            scope_region = [col_left, header_bottom, col_right - col_left, 0.0]
            if region:
                # Extend the region to cover the actual instance positions.
                scope_region[3] = region[1] + region[3] - header_bottom

            # Scope confidence: how cleanly the columns separate brands.
            # If there is only one header sign (no column differentiation possible),
            # confidence is moderate.
            if len(columns) <= 1:
                scope_conf = min(0.70, sign.confidence * 0.75)
            else:
                scope_conf = min(0.92, sign.confidence * 0.95)

            scope = Scope("bay_header", scope_region, scope_conf)
            scoped = replace(sign, scope=scope)
            updated.append(scoped)

            for inst in column_instances:
                pair = (inst.id, sign.brand)
                if pair in emitted_pairs:
                    continue
                assigned_col = self._assign_to_column(inst, columns)
                if assigned_col is None:
                    continue
                containment_conf = self._column_containment_confidence(inst, assigned_col)
                evidence.append(
                    Evidence(
                        instance_id=inst.id,
                        brand=sign.brand,
                        confidence=max(0.0, min(1.0, containment_conf)),
                        cue=self.name,
                        support={
                            "sign_id": sign.sign_id,
                            "sign_text": sign.text,
                            "sign_bbox": list(sign.bbox),
                            "sign_confidence": sign.confidence,
                            "scope": scope.to_dict(),
                            "column_left": assigned_col.left,
                            "column_right": assigned_col.right,
                        },
                    )
                )
                emitted_pairs.add(pair)
                instance_assigned.add(inst.id)

        # Step 3: Handle non-header signs with proximity-based scope.
        for sign in non_header_signs:
            if not sign.brand:
                scoped = replace(sign, scope=Scope("none", None, 0.0))
                updated.append(scoped)
                continue

            # Only consider instances not already assigned by header columns.
            unassigned = [inst for inst in instances if inst.id not in instance_assigned]
            scope_type, scope_instances, scope_conf = self._proximity_scope(
                sign, unassigned, median_width, median_height,
            )

            if scope_conf < self.min_scope_confidence or not scope_instances:
                scoped = replace(sign, scope=Scope("none", None, max(0.0, scope_conf)))
                updated.append(scoped)
                continue

            region = _union(scope_instances)
            scope = Scope(scope_type, region, scope_conf)
            scoped = replace(sign, scope=scope)
            updated.append(scoped)
            for inst in scope_instances:
                if region and not _contains(region, tuple(inst.centroid or _center(inst.bbox))):
                    continue
                evidence.append(
                    Evidence(
                        instance_id=inst.id,
                        brand=sign.brand,
                        confidence=max(0.0, min(1.0, sign.confidence * scope_conf)),
                        cue=self.name,
                        support={
                            "sign_id": sign.sign_id,
                            "sign_text": sign.text,
                            "sign_bbox": list(sign.bbox),
                            "sign_confidence": sign.confidence,
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
