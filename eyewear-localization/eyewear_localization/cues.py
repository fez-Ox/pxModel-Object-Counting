"""Cue implementations C1, C2, and C4.

Cues only emit evidence.  They never decide an instance's final brand; that
responsibility belongs to :mod:`eyewear_localization.fusion`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Callable, Protocol

from eyewear_localization.gazetteer import Gazetteer, normalize_text
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
    * A wider retry crop for unresolved logos so temple-arm text just outside
      the localization box can still be read without widening every crop.
    """

    name = "C1"

    def __init__(
        self,
        ocr: Any,
        gazetteer: Gazetteer,
        *,
        margin: float = 0.25,
        wide_margin: float = 0.40,
        scales: tuple[float, ...] | list[float] = (1.0, 2.0, 4.0),
        sharpen: bool = True,
        use_clahe: bool = False,
        dual_polarity: bool = False,
        verbose: bool = False,
        batch_size: int = 1,
    ) -> None:
        self.ocr = ocr
        self.gazetteer = gazetteer
        self.margin = max(0.0, float(margin))
        self.wide_margin = max(self.margin, float(wide_margin))
        self.scales = tuple(sorted(set(float(s) for s in scales)))
        self.sharpen = sharpen
        self.use_clahe = use_clahe
        self.dual_polarity = dual_polarity
        self.verbose = verbose
        # Batching is deliberately opt-in.  Backends without a batch API, and
        # any failed batch call, use the reference single-image path.
        self.batch_size = max(1, int(batch_size))

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

    @staticmethod
    def _enhance_contrast(image: Any) -> Any:
        """Apply tile-based CLAHE to luminance while preserving RGB chroma."""
        import cv2
        import numpy as np
        from PIL import Image

        # Equalize only the L channel in CIELAB. Applying CLAHE independently
        # to RGB channels can create color shifts around small logos.
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_lightness = clahe.apply(lightness)
        enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb, mode="RGB")

    @staticmethod
    def _invert_polarity(image: Any) -> Any:
        """Apply a photographic negative to read light-on-dark branding."""
        from PIL import ImageOps

        # ImageOps.invert performs per-channel inversion: output = 255-input.
        return ImageOps.invert(image.convert("RGB"))

    def _scaled_variants(self, crop: Any, scale: float) -> list[Any]:
        """Create the exact ordered pixel variants used by the reference path."""
        from PIL import Image

        crop_width, crop_height = crop.size
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        if scale != 1.0:
            working = crop.resize(
                (max(1, round(crop_width * scale)), max(1, round(crop_height * scale))),
                resample=resample_filter,
            )
        else:
            working = crop
        if self.sharpen and scale > 1.0:
            working = self._sharpen(working)

        variants = [working]
        if self.use_clahe:
            variants.append(self._enhance_contrast(working))
        if self.dual_polarity:
            inverted = self._invert_polarity(working)
            variants.append(inverted)
            if self.use_clahe:
                variants.append(self._enhance_contrast(inverted))
        return variants

    def _single_detector(self, method: str | None) -> Callable[[Any], Any] | None:
        detector = getattr(self.ocr, method, None) if method else None
        if not callable(detector):
            detector = getattr(self.ocr, "detect_preprocessed", None)
        return detector if callable(detector) else None

    def _batch_detections(
        self,
        images: list[Any],
        *,
        method: str | None,
    ) -> list[list[TextDetection]]:
        """Run a backend batch, falling back per item on any incompatibility.

        The fallback is intentionally local to a chunk.  A malformed result,
        unsupported processor, or GPU OOM therefore cannot turn valid OCR into
        an empty result.
        """
        detector = self._single_detector(method)
        if not images:
            return []

        batch_name = f"{method}_batch" if method else "detect_preprocessed_batch"
        batch_detector = getattr(self.ocr, batch_name, None)
        if self.batch_size <= 1 or len(images) == 1 or not callable(batch_detector):
            return [
                list(detector(image)) if detector is not None else []
                for image in images
            ]

        results: list[list[TextDetection]] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            try:
                raw = list(batch_detector(chunk))
                if len(raw) != len(chunk):
                    raise ValueError(
                        f"{batch_name} returned {len(raw)} results for {len(chunk)} inputs"
                    )
                results.extend([list(item) for item in raw])
            except Exception:
                # Preserve correctness over throughput for this chunk.
                results.extend(
                    [list(detector(image)) if detector is not None else [] for image in chunk]
                )
        return results

    def _ocr_at_scale_batch(
        self,
        crops: list[Any],
        scale: float,
        *,
        method: str | None = None,
    ) -> list[list[TextDetection]]:
        """OCR several crops while retaining per-crop variant early exits."""
        variants_by_crop = [self._scaled_variants(crop, scale) for crop in crops]
        output: list[list[TextDetection]] = [[] for _ in crops]
        active = list(range(len(crops)))
        variant_count = max((len(variants) for variants in variants_by_crop), default=0)

        for variant_index in range(variant_count):
            current = [
                index for index in active if variant_index < len(variants_by_crop[index])
            ]
            if not current:
                continue
            detections_by_crop = self._batch_detections(
                [variants_by_crop[index][variant_index] for index in current],
                method=method,
            )
            next_active: list[int] = []
            for index, detections in zip(current, detections_by_crop):
                output[index].extend(detections)
                # This is deliberately the same direct-match early exit as
                # _ocr_at_scale; confidence is evaluated later by emit().
                if not any(self.gazetteer.match(det.text) is not None for det in detections):
                    next_active.append(index)
            active = next_active
            if not active:
                break

        # Map boxes from the upscaled working image back to crop coordinates.
        rescaled: list[list[TextDetection]] = []
        for detections in output:
            rescaled.append([
                TextDetection(
                    text=detection.text,
                    bbox=[
                        detection.bbox[0] / scale,
                        detection.bbox[1] / scale,
                        detection.bbox[2] / scale,
                        detection.bbox[3] / scale,
                    ],
                    confidence=detection.confidence,
                    source=detection.source,
                )
                for detection in detections
            ])
        return rescaled

    def _ocr_at_scale(
        self,
        crop: Any,
        scale: float,
        *,
        method: str | None = None,
    ) -> list[TextDetection]:
        """Reference single-crop API backed by the batch scheduler."""
        return self._ocr_at_scale_batch([crop], scale, method=method)[0]

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

    @staticmethod
    def _crop_for_instance(
        image: Any,
        instance: Instance,
        width: int,
        height: int,
        margin: float,
    ) -> tuple[float, float, Any]:
        x, y, box_width, box_height = instance.bbox
        margin_x = box_width * margin
        margin_y = box_height * margin
        left = max(0.0, x - margin_x)
        top = max(0.0, y - margin_y)
        right = min(float(width), x + box_width + margin_x)
        bottom = min(float(height), y + box_height + margin_y)
        return left, top, image.crop((round(left), round(top), round(right), round(bottom)))

    @staticmethod
    def _translate_detections(
        detections: list[TextDetection],
        offset_x: float,
        offset_y: float,
    ) -> list[TextDetection]:
        if offset_x == 0.0 and offset_y == 0.0:
            return detections
        return [
            TextDetection(
                text=detection.text,
                bbox=[
                    detection.bbox[0] + offset_x,
                    detection.bbox[1] + offset_y,
                    detection.bbox[2],
                    detection.bbox[3],
                ],
                confidence=detection.confidence,
                source=detection.source,
            )
            for detection in detections
        ]

    def emit(self, image_path: str | Path, instances: list[Instance]) -> list[Evidence]:
        image = self._load(image_path)
        width, height = image.size
        evidence: list[Evidence] = []
        if self.verbose:
            print(f"[C1] start instances={len(instances)}", flush=True)

        # Crop geometry and pixels are prepared once.  The scheduler below
        # batches only independent OCR work; it never removes an instance or
        # changes the crop, scale, variant, threshold, or fallback budget.
        crops: list[tuple[Instance, float, float, Any, float, float, Any]] = []
        for instance in instances:
            left, top, crop = self._crop_for_instance(
                image, instance, width, height, self.margin
            )
            wide_left, wide_top, wide_crop = self._crop_for_instance(
                image, instance, width, height, self.wide_margin
            )
            crops.append((instance, left, top, crop, wide_left, wide_top, wide_crop))

        all_detections: list[list[tuple[TextDetection, float]]] = [
            [] for _ in crops
        ]
        primary_method = (
            "detect_primary_preprocessed"
            if callable(getattr(self.ocr, "detect_primary_preprocessed", None))
            else None
        )

        # Evaluate one scale for all currently eligible crops.  A crop leaves
        # the active set at exactly the same confidence condition as before.
        active = list(range(len(crops)))
        for scale in self.scales:
            if not active:
                break
            scale_results = self._ocr_at_scale_batch(
                [crops[index][3] for index in active],
                scale,
                method=primary_method,
            )
            next_active: list[int] = []
            for index, detections in zip(active, scale_results):
                all_detections[index].extend((detection, scale) for detection in detections)
                if not any(
                    (match := self.gazetteer.match(det.text)) is not None
                    and det.confidence * match.score >= 0.50
                    for det in detections
                ):
                    next_active.append(index)
            active = next_active

        # Paragraph mode is intentionally left as a per-crop operation; it is
        # an EasyOCR-specific compatibility path and does not invoke the VLM.
        if self.scales:
            best_scale = max(self.scales)
            for index, (_instance, _left, _top, crop, _wide_left, _wide_top, _wide_crop) in enumerate(crops):
                for detection in self._ocr_paragraph_pass(crop, best_scale):
                    all_detections[index].append((detection, best_scale))

            fallback_method = getattr(self.ocr, "detect_fallback_preprocessed", None)
            if callable(fallback_method):
                fallback_indices: list[int] = []
                for index, detections_with_scale in enumerate(all_detections):
                    has_reliable_primary_match = any(
                        (match := self.gazetteer.match(detection.text)) is not None
                        and detection.confidence * match.score >= 0.55
                        for detection, _scale in detections_with_scale
                    )
                    if not has_reliable_primary_match:
                        fallback_indices.append(index)

                # SelectiveOCRBackend exposes the exact remaining budget.  The
                # slice preserves the reference instance-order reservation.
                remaining = getattr(self.ocr, "remaining_fallback_calls", None)
                if isinstance(remaining, int):
                    fallback_indices = fallback_indices[:max(0, remaining)]
                if getattr(self.ocr, "fallback_batch_order_sensitive", False):
                    # The selective cascade owns a shared mutable budget. Its
                    # single-image fallback path may consume more than one
                    # variant call for a crop, so replay it in reference order.
                    for index in fallback_indices:
                        _instance, left, top, _crop, wide_left, wide_top, wide_crop = crops[index]
                        detections = self._ocr_at_scale(
                            wide_crop,
                            best_scale,
                            method="detect_fallback_preprocessed",
                        )
                        detections = self._translate_detections(
                            detections,
                            wide_left - left,
                            wide_top - top,
                        )
                        all_detections[index].extend(
                            (detection, best_scale) for detection in detections
                        )
                else:
                    fallback_results = self._ocr_at_scale_batch(
                        [crops[index][6] for index in fallback_indices],
                        best_scale,
                        method="detect_fallback_preprocessed",
                    )
                    for index, detections in zip(fallback_indices, fallback_results):
                        _instance, left, top, _crop, wide_left, wide_top, _wide_crop = crops[index]
                        detections = self._translate_detections(
                            detections,
                            wide_left - left,
                            wide_top - top,
                        )
                        all_detections[index].extend(
                            (detection, best_scale) for detection in detections
                        )

        for index, (instance, left, top, _crop, _wide_left, _wide_top, _wide_crop) in enumerate(crops):
            # For each brand, keep only the highest-confidence match across
            # all scales and modes. This avoids duplicate evidence while
            # allowing different scales to succeed on different text.
            best_per_brand: dict[str, tuple[Evidence, float]] = {}
            for detection, scale in all_detections[index]:
                match = self.gazetteer.match(detection.text)
                if match is None:
                    continue
                local_x, local_y, local_w, local_h = detection.bbox
                global_box = [left + local_x, top + local_y, local_w, local_h]
                spatial_weight = self._spatial_proximity_weight(global_box, instance.bbox)
                combined_confidence = max(
                    0.0,
                    min(1.0, detection.confidence * match.score * spatial_weight),
                )
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
                        "spatial_weight": spatial_weight,
                        "scale_used": scale,
                    },
                )
                existing = best_per_brand.get(match.brand)
                if existing is None or combined_confidence > existing[1]:
                    best_per_brand[match.brand] = (ev, combined_confidence)
            instance_evidence = [
                ev for ev, _ in sorted(
                    best_per_brand.values(), key=lambda item: item[1], reverse=True
                )
            ]
            evidence.extend(instance_evidence)
            if self.verbose:
                print(f"[C1] {instance.id} evidence={len(instance_evidence)}", flush=True)
        return evidence

    @staticmethod
    def _spatial_proximity_weight(text_bbox: list[float], instance_bbox: list[float]) -> float:
        """Weight C1 text detection based on spatial proximity to core instance box."""
        tx, ty, tw, th = text_bbox
        ix, iy, iw, ih = instance_bbox
        if iw <= 0 or ih <= 0:
            return 1.0

        if ty + th < iy:
            dy = iy - (ty + th)
        elif ty > iy + ih:
            dy = ty - (iy + ih)
        else:
            dy = 0.0

        if tx + tw < ix:
            dx = ix - (tx + tw)
        elif tx > ix + iw:
            dx = tx - (ix + iw)
        else:
            dx = 0.0

        if dx == 0.0 and dy == 0.0:
            return 1.0

        vert_penalty = dy / max(1.0, 0.5 * ih)
        horiz_penalty = dx / max(1.0, 0.5 * iw)
        total_penalty = vert_penalty + horiz_penalty

        return max(0.30, min(1.0, 1.0 - total_penalty))


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

    @classmethod
    def _is_poster_header_sign(
        cls,
        sign: Sign,
        posters: list[PosterRegion],
        instances: list[Instance],
        median_height: float,
    ) -> bool:
        """Allow only a poster that is an actual header above the display.

        Campaign banners can carry the only readable brand label while sitting
        immediately above the shelf. This exception does not admit side/bottom
        advertisements: the containing poster must end no lower than the first
        eyewear row (with a small object-height tolerance), and the sign must
        be in its upper display-facing area.
        """
        if not instances or not cls._poster_contains_sign(sign, posters):
            return False
        first_instance_top = min(instance.bbox[1] for instance in instances)
        sign_center_y = _center(sign.bbox)[1]
        for poster in posters:
            if not (_contains(poster.bbox, _center(sign.bbox)) or _iou(sign.bbox, poster.bbox) >= 0.1):
                continue
            poster_bottom = poster.bbox[1] + poster.bbox[3]
            if (
                poster_bottom <= first_instance_top + max(median_height, 40.0)
                and sign_center_y <= poster.bbox[1] + poster.bbox[3] * 0.75
            ):
                return True
        return False

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
            is_above = (min_inst_y == float("inf")) or (sign_y_center < min_inst_y + median_height * 0.25)
            if abs(sign_y_center - min_y_center) <= band_threshold and is_above:
                headers.append(sign)
            else:
                others.append(sign)
        if headers or not instances:
            return headers, others

        # Some displays place brand plates below the last row (for example
        # Oakley/Meta at the foot of a countertop display). Treat the bottom
        # sign band as a bay anchor when no top header was found. This remains
        # geometry-only and leaves unrelated middle/row labels in `others`.
        max_y_center = max(_center(s.bbox)[1] for s in signs)
        max_inst_bottom = max(inst.bbox[1] + inst.bbox[3] for inst in instances)
        bottom_headers: list[Sign] = []
        bottom_others: list[Sign] = []
        for sign in signs:
            sign_y_center = _center(sign.bbox)[1]
            is_below = sign.bbox[1] >= max_inst_bottom - median_height * 0.10
            if abs(sign_y_center - max_y_center) <= band_threshold and is_below:
                bottom_headers.append(sign)
            else:
                bottom_others.append(sign)
        return bottom_headers, bottom_others

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

    @staticmethod
    def _boundary_anchors(
        signs: list[Sign],
        detections: list[TextDetection] | None,
        median_height: float,
    ) -> list[Sign]:
        """Keep high-quality unknown OCR labels as geometry-only separators.

        A target gazetteer may intentionally omit a neighboring label (for
        example Meta beside Oakley). Its OCR box is still useful for finding
        the boundary between bays, but it must never become brand evidence or
        appear as a target sign in the output.
        """
        if not signs or not detections:
            return []
        known_y = median(_center(sign.bbox)[1] for sign in signs)
        anchors: list[Sign] = []
        for index, detection in enumerate(detections, start=1):
            if detection.confidence < 0.45:
                continue
            normalized = normalize_text(detection.text)
            tokens = normalized.split()
            if len("".join(tokens)) < 3 or len(tokens) > 5:
                continue
            if abs(_center(detection.bbox)[1] - known_y) > max(2.0 * median_height, 100.0):
                continue
            # A matched sign already supplies this geometry. Avoid recreating
            # it as an unknown separator when a backend emitted both forms.
            if any(_iou(detection.bbox, sign.bbox) >= 0.25 for sign in signs):
                continue
            try:
                anchors.append(
                    Sign(
                        sign_id=f"anchor_{index:04d}",
                        text=detection.text,
                        brand=None,
                        bbox=list(detection.bbox),
                        scope=Scope(),
                        confidence=detection.confidence,
                    )
                )
            except ValueError:
                continue
        # De-duplicate unknown OCR fragments at the same physical label.
        unique: list[Sign] = []
        for anchor in sorted(anchors, key=lambda item: item.confidence, reverse=True):
            if any(_iou(anchor.bbox, previous.bbox) >= 0.35 for previous in unique):
                continue
            unique.append(anchor)
        return unique

    def associate(
        self,
        instances: list[Instance],
        signs: list[Sign],
        posters: list[PosterRegion],
        *,
        image_width: float | None = None,
        text_detections: list[TextDetection] | None = None,
    ) -> tuple[list[Sign], list[Evidence]]:
        if instances:
            median_width = max(1.0, median(instance.bbox[2] for instance in instances))
            median_height = max(1.0, median(instance.bbox[3] for instance in instances))
        else:
            median_width = median_height = 1.0

        # Unknown, high-confidence OCR is admitted only as a geometry
        # separator. It cannot produce evidence because its brand is None.
        poster_header_present = any(
            self._is_poster_header_sign(sign, posters, instances, median_height)
            for sign in signs
        )
        # Co-branded campaign headers (e.g. Oakley + Meta) are one display
        # header, not adjacent target-brand bays. Do not use their unknown
        # partner word as a separator in that special geometry.
        boundary_anchors = [] if poster_header_present else self._boundary_anchors(
            signs, text_detections, median_height
        )
        all_input_signs = list(signs) + boundary_anchors

        # Filter out poster-contained signs first.
        active_signs: list[Sign] = []
        updated: list[Sign] = []
        known_sign_ids = {sign.sign_id for sign in signs}
        for sign in all_input_signs:
            if self._poster_contains_sign(sign, posters) and not self._is_poster_header_sign(
                sign, posters, instances, median_height
            ):
                if sign.sign_id in known_sign_ids:
                    scoped = replace(sign, scope=Scope("none", None, 0.99))
                    updated.append(scoped)
            else:
                active_signs.append(sign)

        # Step 1: Identify the header band and infer column boundaries.
        header_signs, non_header_signs = self._find_header_band(active_signs, instances, median_height)

        # A sign band can be above or below the display.  Use its relation to
        # the instance centroid distribution to decide which side is scoped.
        bottom_anchor = False
        header_bottom = None
        header_top = None
        if header_signs:
            sign_y = median(_center(sign.bbox)[1] for sign in header_signs)
            instance_y = median(_center(inst.bbox)[1] for inst in instances) if instances else sign_y
            bottom_anchor = sign_y > instance_y
            header_bottom = max(s.bbox[1] + s.bbox[3] for s in header_signs)
            header_top = min(s.bbox[1] for s in header_signs)
            if bottom_anchor:
                edge_instances = [
                    inst for inst in instances
                    if (inst.centroid or _center(inst.bbox))[1] <= header_top + median_height * 0.25
                ]
            else:
                edge_instances = [
                    inst for inst in instances
                    if (inst.centroid or _center(inst.bbox))[1] >= header_bottom - median_height * 0.25
                ]
        else:
            edge_instances = list(instances)

        columns = _infer_columns(header_signs, image_width, median_width)

        # Step 2: Assign instances to columns and emit evidence for header signs.
        # Deduplicate: emit evidence only once per (instance, brand) pair,
        # even when multiple header signs share the same brand.
        evidence: list[Evidence] = []
        instance_assigned: set[str] = set()  # track which instances got C2 evidence
        emitted_pairs: set[tuple[str, str]] = set()  # (instance_id, brand)

        for sign in header_signs:
            if not sign.brand:
                continue

            # Find all columns for this sign's brand (handles merged columns).
            sign_columns = [c for c in columns if c.brand == sign.brand]
            if not sign_columns:
                scoped = replace(sign, scope=Scope("none", None, 0.0))
                updated.append(scoped)
                continue

            # Collect instances that fall within any of this brand's columns.
            column_instances: list[Instance] = []
            for inst in edge_instances:
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
            if bottom_anchor and header_top is not None:
                scope_region = [col_left, region[1] if region else 0.0, col_right - col_left, 0.0]
                if region:
                    scope_region[3] = max(0.0, header_top - region[1])
            else:
                scope_region = [col_left, header_bottom or 0.0, col_right - col_left, 0.0]
                if region:
                    # Extend the region to cover the actual instance positions.
                    scope_region[3] = region[1] + region[3] - (header_bottom or 0.0)

            poster_header = (
                sign.confidence >= 0.45
                and self._is_poster_header_sign(sign, posters, instances, median_height)
            )
            # Scope confidence: how cleanly the columns separate brands.
            # If there is only one header sign (no column differentiation possible),
            # confidence is moderate. A readable closed-set logo in a campaign
            # banner directly above the shelf is a deliberate stronger case.
            if len(columns) <= 1:
                scope_conf = min(0.70, sign.confidence * 0.75)
            else:
                scope_conf = min(0.92, sign.confidence * 0.95)
            if poster_header:
                scope_conf = max(scope_conf, 0.75)

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
                if poster_header:
                    # The geometry is a single campaign header above the full
                    # display; do not let the fuzzy OCR spelling alone force an
                    # abstention when the closed-set logo is clearly scoped.
                    containment_conf = max(containment_conf, 0.80)
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
                            "poster_header": poster_header,
                        },
                    )
                )
                emitted_pairs.add(pair)
                instance_assigned.add(inst.id)

        # Step 3: Handle non-header signs with proximity-based scope.
        for sign in non_header_signs:
            if not sign.brand:
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

        # If two spatially separated signs independently resolve to the same
        # closed-set brand, treat them as corroborated display anchors. This
        # recovers the common two-sided placard layout where column anchors
        # leave center/overlapping eyewear outside either narrow column. It is
        # deliberately disabled when any competing known brand is present.
        known_signs = [sign for sign in signs if sign.brand]
        known_brands = {sign.brand for sign in known_signs}
        separated_same_brand = [
            sign for sign in known_signs
            if sign.brand == next(iter(known_brands), None)
            and all(
                _iou(sign.bbox, other.bbox) < 0.25
                for other in known_signs
                if other is not sign and other.brand == sign.brand
            )
        ]
        if len(known_brands) == 1 and len(separated_same_brand) >= 2 and instances:
            brand = next(iter(known_brands))
            sign_confidence = max(sign.confidence for sign in separated_same_brand)
            corroborated_confidence = min(0.80, max(0.72, sign_confidence * 1.10))
            span = _union(instances)
            sign_ids = [sign.sign_id for sign in separated_same_brand]
            sign_texts = [sign.text for sign in separated_same_brand]
            covered: set[str] = set()
            for index, item in enumerate(evidence):
                if item.cue != "C2" or item.brand != brand:
                    continue
                covered.add(item.instance_id)
                if item.confidence < corroborated_confidence:
                    support = dict(item.support)
                    support.update({
                        "scope_type": "same_brand_display_span",
                        "sign_ids": sign_ids,
                        "sign_texts": sign_texts,
                        "sign_confidence": sign_confidence,
                    })
                    evidence[index] = Evidence(
                        instance_id=item.instance_id,
                        brand=item.brand,
                        confidence=corroborated_confidence,
                        cue=item.cue,
                        support=support,
                    )
            for instance in instances:
                if instance.id in covered:
                    continue
                evidence.append(
                    Evidence(
                        instance_id=instance.id,
                        brand=brand,
                        confidence=corroborated_confidence,
                        cue="C2",
                        support={
                            "scope": Scope("bay_header", span, corroborated_confidence).to_dict(),
                            "scope_type": "same_brand_display_span",
                            "sign_ids": sign_ids,
                            "sign_texts": sign_texts,
                            "sign_confidence": sign_confidence,
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
