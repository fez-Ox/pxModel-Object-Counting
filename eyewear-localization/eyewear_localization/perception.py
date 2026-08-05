"""L0 perception adapters: class-agnostic localization, OCR, and posters.

Optional heavyweight models are loaded lazily.  If one is unavailable, the
frontend returns an explicit low-reliability empty result rather than making
the rest of the pipeline fail.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol

from eyewear_localization.gazetteer import Gazetteer
from eyewear_localization.schemas import (
    Instance,
    InstanceExclusion,
    PerceptionResult,
    PosterRegion,
    Scope,
    Sign,
    TextDetection,
    xyxy_to_xywh,
)


class OCRBackend(Protocol):
    name: str
    reliability: float

    def detect(self, image: Any) -> list[TextDetection]: ...


class PosterBackend(Protocol):
    name: str
    reliability: float

    def detect(self, image: Any) -> list[PosterRegion]: ...


@dataclass(frozen=True)
class LocalizationDetection:
    """Internal XYXY detection returned by a localizer adapter."""

    box: list[float]
    score: float
    prompt: str
    mask_rle: str | None = None


class NullOCRBackend:
    name = "null"
    reliability = 0.1

    def __init__(self, reason: str = "OCR backend unavailable") -> None:
        self.reason = reason

    def detect(self, image: Any) -> list[TextDetection]:
        return []


class EasyOCRBackend:
    """EasyOCR adapter with an explicit upscaling pass for small lettering."""

    name = "easyocr"
    reliability = 1.0

    def __init__(
        self,
        *,
        languages: list[str] | tuple[str, ...] = ("en",),
        gpu: bool | str = False,
        scale: float = 2.0,
        model_storage_directory: str | None = None,
    ) -> None:
        import easyocr

        self.scale = max(1.0, float(scale))
        kwargs: dict[str, Any] = {"gpu": gpu}
        if model_storage_directory:
            kwargs["model_storage_directory"] = model_storage_directory
        self.reader = easyocr.Reader(list(languages), **kwargs)

    @staticmethod
    def _image(value: Any):
        from PIL import Image

        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                return image.convert("RGB")
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        return Image.fromarray(value).convert("RGB")

    def detect(self, image: Any) -> list[TextDetection]:
        import numpy as np

        source = self._image(image)
        width, height = source.size
        if self.scale != 1.0:
            working = source.resize(
                (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
            )
        else:
            working = source
        raw = self.reader.readtext(
            np.asarray(working),
            detail=1,
            paragraph=False,
            mag_ratio=1.0,
        )
        output: list[TextDetection] = []
        for item in raw:
            if len(item) < 3:
                continue
            polygon, text, confidence = item[0], str(item[1]), float(item[2])
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            if self.scale != 1.0:
                xs = [value / self.scale for value in xs]
                ys = [value / self.scale for value in ys]
            try:
                output.append(
                    TextDetection(
                        text=text,
                        bbox=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                        confidence=max(0.0, min(1.0, confidence)),
                        source=self.name,
                    )
                )
            except ValueError:
                continue
        return output


def build_ocr_backend(
    name: str = "easyocr", *, gpu: bool | str = False, scale: float = 2.0
) -> OCRBackend:
    if name == "none":
        return NullOCRBackend("disabled by configuration")
    if name != "easyocr":
        return NullOCRBackend(f"unknown OCR backend: {name}")
    try:
        return EasyOCRBackend(gpu=gpu, scale=scale)
    except Exception as exc:
        return NullOCRBackend(f"EasyOCR unavailable: {exc}")


SAM3_CLASS_PROMPTS = ("sunglasses", "eyeglasses", "glasses", "rimless glasses")


class Localizer(Protocol):
    name: str
    reliability: float

    def detect(self, image_path: Path) -> list[LocalizationDetection]: ...


class HeuristicLocalizer:
    name = "heuristic-empty"
    reliability = 0.1

    def __init__(self, reason: str = "SAM3/open-vocabulary localizer unavailable") -> None:
        self.reason = reason

    def detect(self, image_path: Path) -> list[LocalizationDetection]:
        return []


class SAM3Localizer:
    """Run only the fixed class prompts required by the specification.

    ``predictor`` is injected so the architecture can be tested without model
    weights.  It receives ``(image_path, prompt)`` and may return either a
    mapping with ``boxes``/``scores`` or an iterable of box/score mappings.
    No brand prompt can enter this adapter.
    """

    name = "sam3-class-agnostic"
    reliability = 1.0

    def __init__(
        self,
        predictor: Callable[[Path, str], Any],
        *,
        score_threshold: float = 0.5,
        rimless_threshold: float = 0.35,
        duplicate_iou: float = 0.7,
    ) -> None:
        self.predictor = predictor
        self.score_threshold = float(score_threshold)
        self.rimless_threshold = float(rimless_threshold)
        self.duplicate_iou = float(duplicate_iou)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _predictions(self, raw: Any) -> Iterable[tuple[list[float], float, str | None]]:
        if isinstance(raw, Mapping):
            boxes = raw.get("boxes", raw.get("box", []))
            scores = raw.get("scores", raw.get("score", []))
            boxes = self._as_list(boxes)
            scores = self._as_list(scores)
            if boxes and len(boxes) == 4 and isinstance(boxes[0], (int, float)):
                boxes = [boxes]
            if not scores:
                scores = [1.0] * len(boxes)
            for box, score in zip(boxes, scores):
                yield [float(value) for value in box], float(score), raw.get("mask_rle")
            return
        for item in raw or []:
            if isinstance(item, Mapping):
                box = item.get("box", item.get("bbox"))
                score = item.get("score", item.get("confidence", 1.0))
                mask_rle = item.get("mask_rle")
            else:
                box, score, mask_rle = item[0], item[1], item[2] if len(item) > 2 else None
            yield [float(value) for value in box], float(score), mask_rle

    @staticmethod
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

    def _deduplicate(self, candidates: list[LocalizationDetection]) -> list[LocalizationDetection]:
        kept: list[LocalizationDetection] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if any(self._iou(candidate.box, previous.box) >= self.duplicate_iou for previous in kept):
                continue
            kept.append(candidate)
        return kept

    def detect_prompt(
        self,
        image_path: Path,
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> list[LocalizationDetection]:
        """Run one non-brand prompt through the injected SAM3 predictor."""
        limit = self.score_threshold if threshold is None else float(threshold)
        candidates = [
            LocalizationDetection(box, score, prompt, mask_rle)
            for box, score, mask_rle in self._predictions(self.predictor(image_path, prompt))
            if len(box) == 4 and score >= limit
        ]
        return self._deduplicate(candidates)

    def detect(self, image_path: Path) -> list[LocalizationDetection]:
        candidates: list[LocalizationDetection] = []
        for prompt in SAM3_CLASS_PROMPTS:
            threshold = self.rimless_threshold if prompt == "rimless glasses" else self.score_threshold
            candidates.extend(self.detect_prompt(image_path, prompt, threshold=threshold))
        return self._deduplicate(candidates)


def build_native_sam3_localizer(
    checkpoint: str | Path,
    *,
    device: str | None = None,
    threshold: float = 0.25,
    amp: bool = True,
) -> Localizer:
    """Adapt the existing native SAM3 runtime without importing brand logic."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        return HeuristicLocalizer(f"checkpoint not found: {checkpoint}")
    app_root = Path(__file__).resolve().parents[2] / "sam3-verbose-counting"
    infer_path = app_root / "infer.py"
    if not infer_path.exists():
        return HeuristicLocalizer("native SAM3 application is not present")
    try:
        sys.path.insert(0, str(app_root))
        spec = importlib.util.spec_from_file_location("_native_sam3_infer", infer_path)
        if spec is None or spec.loader is None:
            raise ImportError("could not load native SAM3 inference module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        counter = module.build_counter(
            checkpoint=checkpoint,
            device=device,
            threshold=threshold,
            amp=amp,
        )

        def predictor(image_path: Path, prompt: str) -> dict[str, Any]:
            return counter.infer(
                image_path,
                prompt,
                box_cleanup=False,
                filter_prompt=None,
            )

        return SAM3Localizer(predictor, score_threshold=threshold)
    except Exception as exc:
        return HeuristicLocalizer(f"native SAM3 unavailable: {exc}")


def detections_to_instances(detections: list[LocalizationDetection]) -> list[Instance]:
    ordered = sorted(detections, key=lambda item: (item.box[1], item.box[0]))
    instances: list[Instance] = []
    for index, detection in enumerate(ordered, start=1):
        try:
            instances.append(
                Instance(
                    id=f"inst_{index:04d}",
                    bbox=xyxy_to_xywh(detection.box),
                    mask_rle=detection.mask_rle,
                    localization_score=max(0.0, min(1.0, detection.score)),
                )
            )
        except ValueError:
            continue
    return instances


class NullPosterBackend:
    name = "null"
    reliability = 0.1

    def __init__(self, reason: str = "poster detector unavailable") -> None:
        self.reason = reason

    def detect(self, image: Any) -> list[PosterRegion]:
        return []


class PerceptionFrontend:
    """Compose L0 model adapters and convert their results to JSON schemas."""

    def __init__(
        self,
        *,
        localizer: Localizer,
        ocr: OCRBackend,
        gazetteer: Gazetteer,
        poster_detector: PosterBackend | None = None,
        scene_filter: Any | None = None,
    ) -> None:
        self.localizer = localizer
        self.ocr = ocr
        self.gazetteer = gazetteer
        self.poster_detector = poster_detector or NullPosterBackend()
        self.scene_filter = scene_filter

    def run(self, image_path: str | Path) -> PerceptionResult:
        image_path = Path(image_path)
        try:
            detections = self.localizer.detect(image_path)
        except Exception:
            detections = []
        instances = detections_to_instances(detections)

        try:
            text_detections = self.ocr.detect(image_path)
        except Exception:
            text_detections = []
        signs: list[Sign] = []
        for index, text_detection in enumerate(text_detections, start=1):
            match = self.gazetteer.match(text_detection.text)
            if match is None:
                continue
            signs.append(
                Sign(
                    sign_id=f"s_{index:02d}",
                    text=text_detection.text,
                    brand=match.brand,
                    bbox=text_detection.bbox,
                    scope=Scope(),
                    confidence=max(0.0, min(1.0, text_detection.confidence * match.score)),
                )
            )

        try:
            posters = self.poster_detector.detect(image_path)
        except Exception:
            posters = []

        excluded_instances: list[InstanceExclusion] = []
        if self.scene_filter is not None:
            try:
                instances, excluded_instances = self.scene_filter.filter(image_path, instances)
            except Exception:
                # Filtering is a conservative optional stage: a broken filter
                # must not discard otherwise valid eyewear detections.
                pass
        return PerceptionResult(
            instances,
            signs,
            posters,
            text_detections,
            excluded_instances,
        )
