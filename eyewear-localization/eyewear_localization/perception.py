"""L0 perception adapters: class-agnostic localization, OCR, and posters.

Optional heavyweight models are loaded lazily.  If one is unavailable, the
frontend returns an explicit low-reliability empty result rather than making
the rest of the pipeline fail.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from io import BytesIO, StringIO
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol

from eyewear_localization.gazetteer import Gazetteer, GazetteerMatch, normalize_text
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


class TesseractOCRBackend:
    """Local OCR fallback that returns word boxes from the Tesseract CLI.

    Tesseract is intentionally used as a fallback/ensemble member rather than
    as a brand classifier: the gazetteer still decides which OCR strings can
    become brand signs.  Several polarity-normalized passes are useful for
    illuminated retail headers (dark text on white) and backlit signs (light
    text on dark), while all coordinates remain in the source image space.
    """

    name = "tesseract"
    reliability = 0.85

    def __init__(
        self,
        *,
        executable: str | None = None,
        scale: float = 1.0,
        psm: int = 11,
        threshold_levels: tuple[int, ...] = (160, 190),
        max_dimension: int = 4200,
        timeout: int = 45,
    ) -> None:
        self.executable = executable or shutil.which("tesseract")
        if not self.executable:
            raise RuntimeError("tesseract executable not found")
        self.scale = max(1.0, float(scale))
        self.psm = int(psm)
        self.threshold_levels = tuple(int(level) for level in threshold_levels)
        self.max_dimension = max(512, int(max_dimension))
        self.timeout = max(5, int(timeout))

    @staticmethod
    def _image(value: Any):
        from PIL import Image

        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                return image.convert("RGB")
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        return Image.fromarray(value).convert("RGB")

    def _variants(self, source: Any, *, scale: float | None = None) -> list[tuple[str, Any]]:
        from PIL import ImageOps

        image = source
        working_scale = self.scale if scale is None else max(1.0, float(scale))
        if working_scale != 1.0:
            image = image.resize(
                (max(1, round(image.width * working_scale)),
                 max(1, round(image.height * working_scale)))
            )
        # Tesseract's resolution estimator performs poorly on large phone
        # photos with small shelf lettering. Limit the working image size, but
        # record the scale so TSV coordinates can be mapped back exactly.
        if max(image.size) > self.max_dimension:
            ratio = self.max_dimension / max(image.size)
            image = image.resize((max(1, round(image.width * ratio)),
                                  max(1, round(image.height * ratio))))
        gray = ImageOps.grayscale(image)
        variants: list[tuple[str, Any]] = [
            ("original", image),
            ("autocontrast", ImageOps.autocontrast(gray)),
        ]
        # Both polarities are retained.  This is not super-resolution; it is
        # a deterministic contrast transform for sign OCR.
        for level in self.threshold_levels:
            binary = gray.point(lambda value, threshold=level: 0 if value < threshold else 255)
            variants.append((f"threshold-{level}", binary))
            variants.append((f"inverse-{level}", ImageOps.invert(binary)))
        return variants

    def _run(self, image: Any, *, brand_mode: bool = False) -> list[TextDetection]:
        payload = BytesIO()
        image.save(payload, format="PNG", optimize=False)
        command = [self.executable, "stdin", "stdout", "--psm", str(self.psm)]
        if brand_mode:
            # A second, closed-alphabet pass helps stylized logos where the
            # normal recognizer inserts arbitrary glyphs between letters (for
            # example ``OAK LY``).  The gazetteer still performs the actual
            # brand match; this is not a brand prompt or classifier.
            command.extend([
                "-c",
                "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz&",
            ])
        command.append("tsv")
        try:
            completed = subprocess.run(
                command,
                input=payload.getvalue(),
                capture_output=True,
                check=False,
                timeout=self.timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []
        try:
            text = completed.stdout.decode("utf-8", errors="replace")
        except AttributeError:
            text = str(completed.stdout)
        word_rows: list[tuple[str, float, list[float], tuple[str, str, str]]] = []
        for row in csv.DictReader(StringIO(text), delimiter="\t"):
            if row.get("level") != "5":
                continue
            value = (row.get("text") or "").strip()
            # Malformed TSV rows have occasionally been observed when
            # Tesseract emits a non-UTF8 glyph.  Never let the rest of the TSV
            # become an OCR token (or a false gazetteer match).
            if (
                not value
                or len(value) > 128
                or any(char in value for char in "\r\n\t")
            ):
                continue
            try:
                confidence = float(row.get("conf", "-1")) / 100.0
                left = float(row.get("left", "0"))
                top = float(row.get("top", "0"))
                width = float(row.get("width", "0"))
                height = float(row.get("height", "0"))
            except (TypeError, ValueError):
                continue
            if (
                not value
                or confidence <= 0.0
                or not any(char.isalnum() for char in value)
                or width <= 0
                or height <= 0
            ):
                continue
            line_key = (
                row.get("block_num", ""),
                row.get("par_num", ""),
                row.get("line_num", ""),
            )
            word_rows.append((
                value,
                max(0.0, min(1.0, confidence)),
                [left, top, width, height],
                line_key,
            ))

        detections: list[TextDetection] = []
        for value, confidence, bbox, _line_key in word_rows:
            try:
                detections.append(TextDetection(
                    text=value,
                    bbox=bbox,
                    confidence=confidence,
                    source=self.name,
                ))
            except ValueError:
                continue

        # Add line-level candidates so multi-token brands such as
        # ``MICHAEL KORS`` and ``DOLCE & GABBANA`` survive word-level OCR.
        # They are candidates only; Gazetteer.match remains the closed-set
        # gate.  Line confidence ignores punctuation-only connector tokens.
        grouped: dict[tuple[str, str, str], list[tuple[str, float, list[float]]]] = {}
        for value, confidence, bbox, line_key in word_rows:
            grouped.setdefault(line_key, []).append((value, confidence, bbox))
        for words in grouped.values():
            if len(words) < 2:
                continue
            words.sort(key=lambda item: (item[2][1], item[2][0]))
            line_text = " ".join(item[0] for item in words)
            if len(line_text) > 256:
                continue
            x0 = min(item[2][0] for item in words)
            y0 = min(item[2][1] for item in words)
            x1 = max(item[2][0] + item[2][2] for item in words)
            y1 = max(item[2][1] + item[2][3] for item in words)
            semantic_words = [
                item for item in words
                if any(char.isalnum() for char in item[0])
            ]
            if not semantic_words:
                continue
            line_confidence = sum(item[1] for item in semantic_words) / len(semantic_words)
            try:
                detections.append(TextDetection(
                    text=line_text,
                    bbox=[x0, y0, x1 - x0, y1 - y0],
                    confidence=max(0.0, min(1.0, line_confidence)),
                    source="tesseract-line",
                ))
            except ValueError:
                continue
        return detections

    @staticmethod
    def _iou(left: list[float], right: list[float]) -> float:
        lx0, ly0, lw, lh = left
        rx0, ry0, rw, rh = right
        x0, y0 = max(lx0, rx0), max(ly0, ry0)
        x1, y1 = min(lx0 + lw, rx0 + rw), min(ly0 + lh, ry0 + rh)
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = lw * lh + rw * rh - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _text_key(value: str) -> str:
        return "".join(char.lower() for char in value if char.isalnum())

    def detect(
        self,
        image: Any,
        *,
        scale: float | None = None,
        tile: bool = True,
    ) -> list[TextDetection]:
        source = self._image(image)
        source_width, source_height = source.size
        variants = self._variants(source, scale=scale)
        work_items: list[tuple[Any, float, float, bool]] = []
        for variant_name, variant in variants:
            work_items.append((variant, 0.0, 0.0, False))
            # A full phone photo can contain readable headers that Tesseract
            # misses because the other shelves dominate its page layout.  OCR
            # overlapping horizontal bands as a second pass.  C1 disables this
            # path because it already supplies a focused product crop.
            if tile and max(source.size) >= 2400 and variant_name in {
                "original", "autocontrast", "threshold-160", "inverse-160",
            }:
                band_height = max(512, round(variant.height * 0.42))
                band_tops = (
                    0,
                    round(variant.height * 0.29),
                    max(0, variant.height - band_height),
                )
                seen_bands: set[tuple[int, int]] = set()
                for top in band_tops:
                    top = min(max(0, int(top)), max(0, variant.height - 1))
                    bottom = min(variant.height, top + band_height)
                    band = (top, bottom)
                    if bottom - top < 32 or band in seen_bands:
                        continue
                    seen_bands.add(band)
                    # Use the closed-alphabet pass for focused bands.  The
                    # full-image passes above retain ordinary scene text.
                    if variant_name in {"original", "autocontrast", "threshold-160", "inverse-160"}:
                        work_items.append(
                            (variant.crop((0, top, variant.width, bottom)), 0.0, float(top), True)
                        )

        detections: list[TextDetection] = []
        for variant, offset_x, offset_y, brand_mode in work_items:
            for detection in self._run(variant, brand_mode=brand_mode):
                # `_variants` may have upscaled and then downscaled the image;
                # map TSV boxes back using the actual working dimensions rather
                # than assuming the requested scale survived the max-size cap.
                x_factor = source_width / max(1, variants[0][1].width)
                y_factor = source_height / max(1, variants[0][1].height)
                bbox = [
                    (detection.bbox[0] + offset_x) * x_factor,
                    (detection.bbox[1] + offset_y) * y_factor,
                    detection.bbox[2] * x_factor,
                    detection.bbox[3] * y_factor,
                ]
                try:
                    detections.append(TextDetection(
                        text=detection.text,
                        bbox=bbox,
                        confidence=detection.confidence,
                        source=self.name,
                    ))
                except ValueError:
                    continue

        # Collapse repeated polarity/pass detections, but do not collapse two
        # distinct signs of the same brand that are spatially separated.
        kept: list[TextDetection] = []
        by_text: dict[str, list[TextDetection]] = {}
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            key = self._text_key(candidate.text)
            duplicate = any(
                self._iou(candidate.bbox, previous.bbox) >= 0.45
                for previous in by_text.get(key, [])
            )
            if not duplicate:
                kept.append(candidate)
                by_text.setdefault(key, []).append(candidate)
        return kept

    def detect_preprocessed(self, image: Any) -> list[TextDetection]:
        """Detect an image that the caller has already resized.

        C1 performs its own crop upscaling.  Reapplying the whole-image OCR
        scale here would make every crop needlessly large and can exhaust the
        worker when a shelf contains many instances.
        """
        return self.detect(image, scale=1.0, tile=False)


def build_ocr_backend(
    name: str = "easyocr",
    *,
    gpu: bool | str = False,
    scale: float = 2.0,
    gazetteer: Gazetteer | None = None,
    fallback_budget: int = 6,
) -> OCRBackend:
    if name == "none":
        return NullOCRBackend("disabled by configuration")
    if name == "tesseract":
        try:
            return TesseractOCRBackend(scale=scale)
        except Exception as exc:
            return NullOCRBackend(f"Tesseract unavailable: {exc}")
    if name == "florence2":
        try:
            device = "cuda" if gpu is True or gpu == "cuda" else (gpu if isinstance(gpu, str) else None)
            return Florence2OCRBackend(device=device, scale=scale)
        except Exception as exc:
            return NullOCRBackend(f"Florence2 unavailable: {exc}")
    if name == "tesseract+florence2":
        if gazetteer is None:
            return NullOCRBackend("tesseract+florence2 requires a gazetteer")
        try:
            device = "cuda" if gpu is True or gpu == "cuda" else (gpu if isinstance(gpu, str) else None)
            return SelectiveOCRBackend(
                TesseractOCRBackend(scale=scale),
                Florence2OCRBackend(device=device, scale=scale),
                gazetteer,
                max_fallback_calls=fallback_budget,
            )
        except Exception as exc:
            return NullOCRBackend(f"tesseract+florence2 unavailable: {exc}")
    if name != "easyocr":
        return NullOCRBackend(f"unknown OCR backend: {name}")
    try:
        return EasyOCRBackend(gpu=gpu, scale=scale)
    except Exception as exc:
        return NullOCRBackend(f"EasyOCR unavailable: {exc}")


class Florence2OCRBackend:
    """Florence-2 zero-shot OCR backend using HuggingFace transformers."""

    name = "florence2"
    reliability = 1.0

    def __init__(
        self,
        *,
        model_id: str = "microsoft/Florence-2-base",
        device: str | None = None,
        scale: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.scale = max(1.0, float(scale))
        self._processor = None
        self._model = None
        self._fallback: TesseractOCRBackend | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        import torch
        import transformers.utils.import_utils
        import transformers.dynamic_module_utils
        from transformers import AutoModelForCausalLM, AutoProcessor, PretrainedConfig

        # Patch 1: forced_bos_token_id compatibility for transformers >= 4.45
        if not hasattr(PretrainedConfig, "forced_bos_token_id"):
            setattr(PretrainedConfig, "forced_bos_token_id", None)

        # Patch 2: flash_attn availability check override
        transformers.utils.import_utils.is_flash_attn_2_available = lambda *args, **kwargs: False

        # Patch 3: dynamic check_imports override to prevent flash_attn requirement crash
        orig_check_imports = transformers.dynamic_module_utils.check_imports
        def patched_check_imports(filename):
            try:
                return orig_check_imports(filename)
            except ImportError as exc:
                if "flash_attn" in str(exc):
                    return []
                raise
        transformers.dynamic_module_utils.check_imports = patched_check_imports

        device_str = self.device
        if device_str is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if "cuda" in device_str else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device_str)
        self.actual_device = device_str

    @staticmethod
    def _image(value: Any):
        from PIL import Image

        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                return image.convert("RGB")
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        return Image.fromarray(value).convert("RGB")

    def _detect_fallback(self, image: Any, *, preprocessed: bool = False) -> list[TextDetection]:
        if self._fallback is None:
            try:
                self._fallback = TesseractOCRBackend(scale=min(self.scale, 2.0))
            except Exception:
                self._fallback = TesseractOCRBackend.__new__(TesseractOCRBackend)
                self._fallback.executable = None
        if not getattr(self._fallback, "executable", None):
            return []
        if preprocessed:
            detect_preprocessed = getattr(self._fallback, "detect_preprocessed", None)
            if callable(detect_preprocessed):
                return list(detect_preprocessed(image))
        return list(self._fallback.detect(image))

    def _detect_model(self, source: Any, *, scale: float) -> list[TextDetection]:
        import torch

        working = source
        if scale != 1.0:
            working = source.resize(
                (max(1, round(source.width * scale)),
                 max(1, round(source.height * scale)))
            )
        width, height = working.size

        # OCR_WITH_REGION is the Florence-2 task that returns both text and
        # boxes.  Plain <OCR> returns a string on some model revisions and a
        # region dictionary on others, which previously made the parser
        # silently produce no detections for the Kaggle run.
        prompt = "<OCR_WITH_REGION>"
        inputs = self._processor(text=prompt, images=working, return_tensors="pt").to(self.actual_device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self._processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(width, height),
        )
        if not isinstance(parsed_answer, Mapping):
            parsed_answer = {}
        ocr_data = parsed_answer.get(prompt, parsed_answer.get("<OCR>", {}))
        labels = ocr_data.get("labels", []) if isinstance(ocr_data, Mapping) else []
        bboxes = ocr_data.get("bboxes", []) if isinstance(ocr_data, Mapping) else []

        output: list[TextDetection] = []
        if bboxes and labels and len(bboxes) == len(labels):
            for box, text in zip(bboxes, labels):
                text_str = str(text).strip()
                if not text_str:
                    continue
                try:
                    x0, y0, x1, y1 = (float(value) for value in box[:4])
                except (TypeError, ValueError):
                    continue
                x0 /= scale
                y0 /= scale
                x1 /= scale
                y1 /= scale
                try:
                    output.append(
                        TextDetection(
                            text=text_str,
                            bbox=[x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                            # Florence-2 does not expose a calibrated box
                            # confidence here. Keep it below a high-confidence
                            # exact OCR result so geometry/fusion still matter.
                            confidence=0.80,
                            source=self.name,
                        )
                    )
                except ValueError:
                    continue

        if not output and isinstance(ocr_data, str) and ocr_data.strip():
            for line in ocr_data.splitlines():
                line = line.strip()
                if line:
                    try:
                        output.append(
                            TextDetection(
                                text=line,
                                bbox=[0.0, 0.0, float(source.width), float(source.height)],
                                confidence=0.75,
                                source=self.name,
                            )
                        )
                    except ValueError:
                        continue
        return output

    def detect(self, image: Any) -> list[TextDetection]:
        source = self._image(image)
        try:
            self._ensure_loaded()
            output = self._detect_model(source, scale=self.scale)
        except Exception as exc:
            import sys
            print(f"WARNING Florence2 inference error: {exc}", file=sys.stderr)
            output = []
        if output:
            return output
        # Keep Florence-2 as the preferred zero-shot backend, but never turn
        # an empty model response into a falsely healthy OCR result.  The
        # deterministic local fallback is especially useful for high-contrast
        # retail headers and remains gazetteer-gated by PerceptionFrontend.
        return self._detect_fallback(source)

    def detect_preprocessed(self, image: Any) -> list[TextDetection]:
        """Run OCR on an image already upscaled by the C1 cropper."""
        source = self._image(image)
        try:
            self._ensure_loaded()
            output = self._detect_model(source, scale=1.0)
        except Exception:
            output = []
        return output or self._detect_fallback(source, preprocessed=True)


class SelectiveOCRBackend:
    """Use inexpensive OCR first and a bounded GPU model as a second pass.

    The primary OCR may find one scene brand while missing a sibling bay.  The
    stronger scene pass therefore runs once per image; C1 may use remaining
    budget for unmatched product crops.  The gazetteer remains the closed-set
    gate, not an open-ended brand prompt, and the budget bounds GPU calls.
    """

    name = "tesseract+florence2-selective"

    def __init__(
        self,
        primary: OCRBackend,
        fallback: OCRBackend,
        gazetteer: Gazetteer,
        *,
        max_fallback_calls: int = 6,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.gazetteer = gazetteer
        self.max_fallback_calls = max(0, int(max_fallback_calls))
        self._fallback_calls = 0
        self._scene_fallback_done = False
        self.reliability = min(
            1.0,
            max(float(getattr(primary, "reliability", 0.0)), float(getattr(fallback, "reliability", 0.0))),
        )

    def reset_budget(self) -> None:
        """Start an independent fallback budget for the next source image."""
        self._fallback_calls = 0
        self._scene_fallback_done = False

    def _should_fallback(self, detections: list[TextDetection]) -> bool:
        """Retain the fallback budget for scene and crop stages.

        A scene can contain several brands.  Finding one Tesseract match must
        not suppress the stronger scene pass, otherwise a missed sibling bay
        (for example Michael Kors beside Burberry) can never be recovered.
        """
        return self._fallback_calls < self.max_fallback_calls

    @staticmethod
    def _deduplicate(detections: list[TextDetection]) -> list[TextDetection]:
        """Suppress only equivalent text at materially overlapping locations."""
        kept: list[TextDetection] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            candidate_key = TesseractOCRBackend._text_key(candidate.text)
            if any(
                candidate_key == TesseractOCRBackend._text_key(previous.text)
                and TesseractOCRBackend._iou(candidate.bbox, previous.bbox) >= 0.5
                for previous in kept
            ):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _call(backend: OCRBackend, method: str, image: Any) -> list[TextDetection]:
        try:
            return list(getattr(backend, method)(image))
        except Exception:
            return []

    def _detect(self, image: Any, method: str) -> list[TextDetection]:
        primary = self._call(self.primary, method, image)
        if not self._should_fallback(primary):
            return primary
        stronger = self._fallback_detect(image, method)
        return self._deduplicate(primary + stronger)

    def _fallback_detect(self, image: Any, method: str) -> list[TextDetection]:
        if self._fallback_calls >= self.max_fallback_calls:
            return []
        self._fallback_calls += 1
        return self._call(self.fallback, method, image)

    def detect(self, image: Any) -> list[TextDetection]:
        primary = self._call(self.primary, "detect", image)
        if not self._scene_fallback_done:
            self._scene_fallback_done = True
            stronger = self._fallback_detect(image, "detect")
            return self._deduplicate(primary + stronger)
        return primary

    def detect_preprocessed(self, image: Any) -> list[TextDetection]:
        # C1 uses the explicit primary/fallback methods below, so a direct
        # caller still receives the safe default cascade behavior.
        return self._detect(image, "detect_preprocessed")

    def detect_primary_preprocessed(self, image: Any) -> list[TextDetection]:
        """Run only cheap OCR; used by C1 across all of its resize scales."""
        return self._call(self.primary, "detect_preprocessed", image)

    def detect_fallback_preprocessed(self, image: Any) -> list[TextDetection]:
        """Spend one bounded stronger-OCR call on a C1 crop if warranted."""
        return self._fallback_detect(image, "detect_preprocessed")


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
        verbose: bool = False,
        release_callback: Callable[[], None] | None = None,
    ) -> None:
        self.predictor = predictor
        self.score_threshold = float(score_threshold)
        self.rimless_threshold = float(rimless_threshold)
        self.duplicate_iou = float(duplicate_iou)
        self.verbose = bool(verbose)
        self._release_callback = release_callback

    def release(self) -> None:
        """Release an optional native model after L0 is complete."""
        if self._release_callback is not None:
            self._release_callback()
            self._release_callback = None

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
        if self.verbose:
            print(f"[SAM3] start prompt={prompt!r} image={image_path.name}", flush=True)
        candidates = [
            LocalizationDetection(box, score, prompt, mask_rle)
            for box, score, mask_rle in self._predictions(self.predictor(image_path, prompt))
            if len(box) == 4 and score >= limit
        ]
        result = self._deduplicate(candidates)
        if self.verbose:
            print(f"[SAM3] done prompt={prompt!r} detections={len(result)}", flush=True)
        return result

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
        print(f"[SAM3] loading checkpoint={checkpoint}", flush=True)
        counter = module.build_counter(
            checkpoint=checkpoint,
            device=device,
            threshold=threshold,
            amp=amp,
        )
        print("[SAM3] model ready", flush=True)

        def predictor(image_path: Path, prompt: str) -> dict[str, Any]:
            return counter.infer(
                image_path,
                prompt,
                box_cleanup=False,
                filter_prompt=None,
            )

        return SAM3Localizer(
            predictor,
            score_threshold=threshold,
            verbose=True,
            release_callback=counter.release,
        )
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

    @staticmethod
    def _intersection(left: list[float], right: list[float]) -> float:
        x0 = max(left[0], right[0])
        y0 = max(left[1], right[1])
        x1 = min(left[0] + left[2], right[0] + right[2])
        y1 = min(left[1] + left[3], right[1] + right[3])
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @classmethod
    def _is_plausible_sign(cls, detection: TextDetection, match: GazetteerMatch) -> bool:
        """Reject broad OCR sentences that merely contain a brand token.

        Scene OCR can return a product line, advertisement caption, or several
        adjacent words as one box.  Treating ``HSTN Oakley Mahomes Oakley
        Mbappé`` as an Oakley header would create a false column spanning the
        whole shelf.  Short compounds such as ``RAY BAN EYEWEAR`` remain
        allowed, while long token-rich lines are left as raw OCR only.
        """
        tokens = normalize_text(detection.text).split()
        brand_tokens = normalize_text(match.brand).split()
        if not tokens or not brand_tokens:
            return False
        if match.method == "token_containment":
            return len(tokens) <= len(brand_tokens) + 1
        if match.method == "token_containment_without_connector":
            return len(tokens) <= max(2, len(brand_tokens) - 1) + 1
        return True

    @classmethod
    def _matched_signs(cls, detections: list[TextDetection], gazetteer: Gazetteer) -> list[Sign]:
        """Create one spatially deduplicated sign per physical label."""
        candidates: list[tuple[Sign, float]] = []
        for index, detection in enumerate(detections, start=1):
            match = gazetteer.match(detection.text)
            if match is None or not cls._is_plausible_sign(detection, match):
                continue
            # A zero/near-zero OCR confidence is useful in raw diagnostics but
            # is not strong enough to define a display zone.  This prevents
            # malformed words such as low-confidence ``Cartier``/``MICHAEL``
            # fragments from becoming C2 anchors.
            combined_confidence = detection.confidence * match.score
            if combined_confidence < 0.25:
                continue
            try:
                candidates.append((
                    Sign(
                        sign_id=f"candidate_{index:04d}",
                        text=detection.text,
                        brand=match.brand,
                        bbox=detection.bbox,
                        scope=Scope(),
                        confidence=max(0.0, min(1.0, combined_confidence)),
                    ),
                    match.score,
                ))
            except ValueError:
                continue

        # Keep separate occurrences of the same brand, but collapse polarity,
        # word/line, and OCR-backend duplicates that describe one label.
        kept: list[Sign] = []
        for candidate, _match_score in sorted(
            candidates,
            key=lambda item: (item[0].confidence, item[0].bbox[1], item[0].bbox[0]),
            reverse=True,
        ):
            duplicate_index: int | None = None
            for previous_index, previous in enumerate(kept):
                if candidate.brand != previous.brand:
                    continue
                intersection = cls._intersection(candidate.bbox, previous.bbox)
                min_area = min(
                    candidate.bbox[2] * candidate.bbox[3],
                    previous.bbox[2] * previous.bbox[3],
                )
                union = (
                    candidate.bbox[2] * candidate.bbox[3]
                    + previous.bbox[2] * previous.bbox[3]
                    - intersection
                )
                iou = intersection / union if union > 0 else 0.0
                contained = intersection / min_area if min_area > 0 else 0.0
                if iou >= 0.35 or contained >= 0.75:
                    duplicate_index = previous_index
                    break
            if duplicate_index is None:
                kept.append(candidate)
            elif candidate.confidence > kept[duplicate_index].confidence:
                kept[duplicate_index] = candidate

        kept.sort(key=lambda sign: (sign.bbox[1], sign.bbox[0], sign.brand or ""))
        return [replace(sign, sign_id=f"s_{index:02d}") for index, sign in enumerate(kept, start=1)]

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

    def run(
        self,
        image_path: str | Path,
        *,
        text_detections_override: list[TextDetection] | None = None,
    ) -> PerceptionResult:
        image_path = Path(image_path)
        try:
            detections = self.localizer.detect(image_path)
        except Exception:
            detections = []
        instances = detections_to_instances(detections)

        if text_detections_override is not None:
            text_detections = list(text_detections_override)
        else:
            try:
                text_detections = self.ocr.detect(image_path)
            except Exception:
                text_detections = []
        signs = self._matched_signs(text_detections, self.gazetteer)

        try:
            posters = self.poster_detector.detect(image_path)
        except Exception:
            posters = []

        excluded_instances: list[InstanceExclusion] = []
        if self.scene_filter is not None:
            try:
                instances, excluded_instances = self.scene_filter.filter(image_path, instances)
                # SAM3SceneFilter already detects poster regions while
                # filtering. Reuse those boxes for C2 so advertisement text
                # cannot become a display-zone sign, while retaining the
                # independently injected poster backend when one is present.
                if not posters:
                    posters = list(getattr(self.scene_filter, "last_poster_regions", []))
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
