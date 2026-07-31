#!/usr/bin/env python3
"""Count objects in images with native SAM3 and verbose text prompts.

Examples:
  uv run python infer.py image.jpg "all pairs of black sunglasses displayed on the retail rack"
  uv run python infer.py images/ "all pairs of black sunglasses displayed on the retail rack"
  uv run python infer.py images/ "all pairs of black sunglasses displayed on the retail rack" --recursive
  uv run python infer.py https://example.com/image.jpg "the red cars parked beside the building"
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


def _resolve_checkpoint(value: str) -> Path:
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


def _draw_result(image_path: Path, prompt: str, boxes: list[list[float]], scores: list[float]):
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

    def infer(self, image_path: Path, prompt: str) -> dict:
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

        elapsed = time.perf_counter() - start
        if is_cuda:
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        else:
            peak_allocated = None
            peak_reserved = None

        boxes = state.get("boxes", torch.empty((0, 4))).detach().cpu().tolist()
        scores = state.get("scores", torch.empty((0,))).detach().cpu().tolist()
        # Release image features before processing the next image.
        state.clear()
        return {
            "count": len(boxes),
            "boxes": [[float(value) for value in box] for box in boxes],
            "scores": [float(value) for value in scores],
            "inference_time_seconds": elapsed,
            "peak_vram_mb": peak_allocated,
            "peak_reserved_vram_mb": peak_reserved,
        }


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
    args = parser.parse_args()

    try:
        checkpoint = _resolve_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    image_inputs = _expand_inputs(args.images, recursive=args.recursive)
    if not image_inputs:
        parser.error("no images to process")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but CUDA is unavailable: {device}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    try:
        counter = Sam3VerboseCounter(
            checkpoint=checkpoint,
            device=device,
            threshold=args.threshold,
            amp=not args.no_amp,
        )
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
                result = counter.infer(image_path.resolve(), args.prompt)
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
            visualization = _draw_result(
                image_path,
                args.prompt,
                result["boxes"],
                result["scores"],
            )
            visualization.save(image_output_path, quality=95)

            print(f"  Count: {result['count']}")
            print(f"  Boxes: {len(result['boxes'])}")
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
