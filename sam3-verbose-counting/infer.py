#!/usr/bin/env python3
"""Count objects in images with native SAM3 and verbose text prompts.

CLI examples:
  uv run python infer.py image.jpg "all pairs of black sunglasses displayed on the retail rack"
  uv run python infer.py images/ "all pairs of black sunglasses displayed on the retail rack"
  uv run python infer.py images/ "all pairs of black sunglasses displayed on the retail rack" --recursive
  uv run python infer.py https://example.com/image.jpg "the red cars parked beside the building"

Notebook / library usage (run from a project root that has `sam3-verbose-counting/`
on ``sys.path``, with the ``sam3.pt`` checkpoint downloaded):

    from infer import build_counter, annotate

    counter = build_counter(threshold=0.5)          # loads sam3.pt on cuda (or cpu)
    result = counter.infer("image.jpg", "the red cars parked beside the building")
    annotated = annotate("image.jpg", "the red cars parked beside the building",
                         result["boxes"], result["scores"])
    print(result["count"])
    annotated.save("annotated.jpg")
"""

from __future__ import annotations

import argparse
import mimetypes
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import unquote, urlparse

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = APP_ROOT / "checkpoints" / "sam3.pt"
DEFAULT_OUTPUT = APP_ROOT / "outputs" / "sam3_verbose_counting"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

__all__ = [
    "Sam3VerboseCounter",
    "build_counter",
    "annotate",
    "filter_overlapping",
    "resolve_checkpoint",
    "cuda_available",
    "image_path_to_pil",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_OUTPUT",
]


def cuda_available() -> bool:
    """Return True when a CUDA-capable device is available to torch."""
    import torch

    return torch.cuda.is_available()


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_image(url: str, work_dir: str, timeout: int) -> Path:
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path)).suffix
    if not suffix:
        content_type = mimetypes.guess_type(url)[0] or ""
        suffix = mimetypes.guess_extension(content_type) or ".jpg"

    with tempfile.NamedTemporaryFile(
        prefix="sam3_url_", suffix=suffix, dir=work_dir, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)

    request = urllib.request.Request(url, headers={"User-Agent": "sam3-verbose-counting/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (urllib.error.URLError, TimeoutError, OSError):
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _expand_inputs(inputs: list[str], recursive: bool) -> list[str]:
    expanded: list[str] = []
    for value in inputs:
        if _is_url(value):
            expanded.append(value)
            continue

        path = Path(value).expanduser()
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            images = sorted(item for item in iterator if _is_image_file(item))
            if not images:
                print(f"Skip: {value} — no supported image files found", file=sys.stderr)
                continue
            expanded.extend(str(item) for item in images)
        else:
            expanded.append(value)
    return expanded


def _display_name(value: str) -> str:
    if _is_url(value):
        parsed = urlparse(value)
        return Path(unquote(parsed.path)).name or parsed.netloc
    return Path(value).name


def _slugify(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return (safe.strip("_") or "image")[:160]


def resolve_checkpoint(value: str | None = None) -> Path:
    """Resolve a checkpoint path relative to this app, verifying it exists."""
    if value is None:
        value = str(DEFAULT_CHECKPOINT)
    checkpoint = Path(value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = APP_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found at {checkpoint}. "
            "Run: uv run python download_model.py"
        )
    return checkpoint


def _next_output_path(output_dir: Path, stem: str, suffix: str) -> Path:
    candidate = output_dir / f"{stem}{suffix}"
    index = 2
    while candidate.exists():
        candidate = output_dir / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def annotate(image_path: Path, prompt: str, boxes: list[list[float]], scores: list[float]):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    line_width = max(2, round(min(width, height) / 300))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, width // 55))
    except Exception:
        font = ImageFont.load_default()

    for index, (box, score) in enumerate(zip(boxes, scores), start=1):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y1), outline=(235, 45, 35, 255), width=line_width)
        label = f"{index}: {score:.2f}"
        text_box = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle(text_box, fill=(235, 45, 35, 220))
        draw.text((x0, y0), label, fill=(255, 255, 255, 255), font=font)

    label = f"{len(boxes)} detected | {prompt}"
    text_box = draw.textbbox((0, 0), label, font=font)
    padding = 8
    draw.rectangle(
        (0, 0, text_box[2] + padding * 2, text_box[3] + padding * 2),
        fill=(0, 0, 0, 180),
    )
    draw.text((padding, padding), label, fill=(255, 255, 255, 255), font=font)
    return image


def _box_area(box: list[float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _box_center_in_box(target: list[float], ref: list[float]) -> bool:
    cx = (target[0] + target[2]) / 2.0
    cy = (target[1] + target[3]) / 2.0
    return ref[0] <= cx <= ref[2] and ref[1] <= cy <= ref[3]


def filter_overlapping(
    target_boxes: list[list[float]],
    target_scores: list[float],
    filter_boxes: list[list[float]],
    *,
    center: bool = True,
    iou: float = 0.0,
) -> tuple[list[list[float]], list[float], list[int]]:
    """Drop target boxes that overlap any filter box.

    A target box is removed when its center lies inside a filter box
    (``center=True``) or its IoU with any filter box reaches ``iou``.
    Both criteria may be active at once; a match on either removes the box.

    Returns ``(kept_boxes, kept_scores, removed_indices)``. All boxes are in
    ``[x0, y0, x1, y1]`` pixel coordinates.
    """
    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    removed: list[int] = []
    for index, (box, score) in enumerate(zip(target_boxes, target_scores)):
        overlap = any(
            (center and _box_center_in_box(box, ref))
            or (iou > 0 and _box_iou(box, ref) >= iou)
            for ref in filter_boxes
        )
        if overlap:
            removed.append(index)
        else:
            kept_boxes.append(box)
            kept_scores.append(score)
    return kept_boxes, kept_scores, removed


class Sam3VerboseCounter:
    """Persistent native SAM3 image counter."""

    def __init__(self, checkpoint: Path, device: str, threshold: float, amp: bool = True):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.threshold = float(threshold)
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            device=str(self.device),
            eval_mode=True,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.threshold,
        )

    def _autocast(self):
        if not self.amp:
            return nullcontext()
        capability = self.torch.cuda.get_device_capability(self.device)
        dtype = self.torch.bfloat16 if capability[0] >= 8 else self.torch.float16
        return self.torch.autocast(device_type="cuda", dtype=dtype)

    def infer(
        self,
        image_path: Path,
        prompt: str,
        *,
        filter_prompt: str | None = None,
        filter_center: bool = True,
        filter_iou: float = 0.0,
    ) -> dict:
        """Count objects matching ``prompt``, optionally filtered.

        When ``filter_prompt`` is given, a second SAM3 pass runs on the same
        image to localize the filter objects (e.g. ``"faces of people"``), and
        every target box whose center lands inside a filter box
        (``filter_center``) or whose IoU with one reaches ``filter_iou`` is
        dropped from the count.

        The unfiltered detections are kept on the result under ``raw_boxes`` /
        ``raw_scores`` / ``raw_count`` so the filtering decision can be audited.
        """
        torch = self.torch
        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()

        try:
            with torch.inference_mode(), self._autocast():
                state = self.processor.set_image(image_path_to_pil(image_path))
                state = self.processor.set_text_prompt(prompt, state)
            if is_cuda:
                torch.cuda.synchronize(self.device)
        except Exception:
            if is_cuda:
                torch.cuda.synchronize(self.device)
            raise

        target_boxes = state.get("boxes", torch.empty((0, 4))).detach().cpu().tolist()
        target_scores = state.get("scores", torch.empty((0,))).detach().cpu().tolist()

        filter_boxes: list[list[float]] = []
        if filter_prompt:
            try:
                with torch.inference_mode(), self._autocast():
                    state = self.processor.set_text_prompt(filter_prompt, state)
                if is_cuda:
                    torch.cuda.synchronize(self.device)
            except Exception:
                if is_cuda:
                    torch.cuda.synchronize(self.device)
                raise
            filter_boxes = state.get("boxes", torch.empty((0, 4))).detach().cpu().tolist()

        elapsed = time.perf_counter() - start
        if is_cuda:
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        else:
            peak_allocated = None
            peak_reserved = None

        if filter_boxes:
            kept_boxes, kept_scores, removed = filter_overlapping(
                target_boxes,
                target_scores,
                filter_boxes,
                center=filter_center,
                iou=filter_iou,
            )
        else:
            kept_boxes, kept_scores, removed = target_boxes, target_scores, []

        # Release image features before processing the next image.
        state.clear()

        result = {
            "count": len(kept_boxes),
            "boxes": [[float(value) for value in box] for box in kept_boxes],
            "scores": [float(value) for value in kept_scores],
            "inference_time_seconds": elapsed,
            "peak_vram_mb": peak_allocated,
            "peak_reserved_vram_mb": peak_reserved,
        }
        if filter_prompt:
            result.update(
                {
                    "raw_count": len(target_boxes),
                    "raw_boxes": [[float(value) for value in box] for box in target_boxes],
                    "raw_scores": [float(value) for value in target_scores],
                    "filtered_count": len(removed),
                    "filter_prompt": filter_prompt,
                    "filter_object_count": len(filter_boxes),
                }
            )
        return result


def build_counter(
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    threshold: float = 0.5,
    amp: bool = True,
) -> Sam3VerboseCounter:
    """Build a persistent SAM3 counter, resolved for the current environment.

    Args:
        checkpoint: Path to ``sam3.pt``. Defaults to ``checkpoints/sam3.pt``.
        device: Torch device string. Defaults to ``cuda`` when available, else ``cpu``.
        threshold: Detection confidence threshold in ``[0, 1]``.
        amp: Enable CUDA automatic mixed precision (ignored on CPU).

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If a CUDA device is requested but unavailable.

    Returns:
        A ready-to-use :class:`Sam3VerboseCounter`.
    """
    resolved = resolve_checkpoint(str(checkpoint) if checkpoint is not None else None)
    if device is None:
        device = "cuda" if cuda_available() else "cpu"
    if device.startswith("cuda") and not cuda_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    return Sam3VerboseCounter(
        checkpoint=resolved,
        device=device,
        threshold=threshold,
        amp=amp,
    )


def image_path_to_pil(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count objects with native SAM3 using a verbose text prompt."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image path(s), folder(s), or http(s) URL(s); folders contain images",
    )
    parser.add_argument(
        "prompt",
        help="Verbose natural-language description of the objects to count",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to sam3.pt (default: checkpoints/sam3.pt)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=str(DEFAULT_OUTPUT),
        help="Output directory for visualizations and JSON files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Include images in subfolders",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for image URLs (default: 30)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Display each visualization if a display is available",
    )
    parser.add_argument(
        "--filter-prompt",
        default=None,
        help="Second prompt whose detections suppress target boxes "
        "(e.g. 'faces of people'); any target box overlapping a "
        "detection is dropped from the count",
    )
    parser.add_argument(
        "--filter-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop a target box when its center lies inside a filter detection "
        "(default: on; use --no-filter-center to disable)",
    )
    parser.add_argument(
        "--filter-iou",
        type=float,
        default=0.0,
        help="Also drop a target box when its IoU with any filter detection "
        "reaches this value (default: 0.0 = disabled)",
    )
    args = parser.parse_args()

    if args.filter_prompt and not (args.filter_center or args.filter_iou > 0):
        parser.error("--filter-prompt needs --filter-center and/or --filter-iou > 0")

    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    image_inputs = _expand_inputs(args.images, recursive=args.recursive)
    if not image_inputs:
        parser.error("no images to process")

    try:
        counter = build_counter(
            checkpoint=checkpoint,
            device=args.device,
            threshold=args.threshold,
            amp=not args.no_amp,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(f"could not initialize SAM3: {exc}")

    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sam3_verbose_counting_") as temporary_dir:
        for value in image_inputs:
            if _is_url(value):
                print(f"Downloading: {value}")
                try:
                    image_path = _download_image(value, temporary_dir, args.download_timeout)
                except Exception as exc:
                    print(f"Skip: {value} — download failed: {exc}", file=sys.stderr)
                    continue
            else:
                image_path = Path(value).expanduser()
                if not image_path.exists():
                    print(f"Skip: {value} — file not found", file=sys.stderr)
                    continue

            print(f"Processing: {_display_name(value)}")
            try:
                result = counter.infer(
                    image_path.resolve(),
                    args.prompt,
                    filter_prompt=args.filter_prompt,
                    filter_center=args.filter_center,
                    filter_iou=args.filter_iou,
                )
            except Exception as exc:
                print(f"  Error: {exc}", file=sys.stderr)
                continue

            source_stem = Path(_display_name(value)).stem
            output_stem = _slugify(f"{source_stem}__{args.prompt}")
            json_path = _next_output_path(output_dir, output_stem, ".json")
            image_output_path = json_path.with_suffix(".jpg")
            payload = {
                "image_path": str(image_path.resolve()),
                "prompt": args.prompt,
                "threshold": args.threshold,
                **result,
            }
            import json

            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            visualization = annotate(
                image_path,
                args.prompt,
                result["boxes"],
                result["scores"],
            )
            visualization.save(image_output_path, quality=95)

            print(f"  Count: {result['count']}")
            print(f"  Boxes: {len(result['boxes'])}")
            if "raw_count" in result:
                print(
                    f"  (after filter: raw={result['raw_count']}, "
                    f"removed={result['filtered_count']}, "
                    f"filter objects={result['filter_object_count']})"
                )
            print(f"  Inference time: {result['inference_time_seconds']:.2f}s")
            if result["peak_vram_mb"] is not None:
                print(f"  Peak VRAM allocated: {result['peak_vram_mb']:.1f} MiB")
                print(f"  Peak VRAM reserved: {result['peak_reserved_vram_mb']:.1f} MiB")
            else:
                print("  Peak VRAM: unavailable (CPU inference)")
            print(f"  JSON: {json_path}")
            print(f"  Visualization: {image_output_path}")
            if args.show:
                visualization.show()
            print()


if __name__ == "__main__":
    main()
