#!/usr/bin/env python3
"""CLI for CountAnything text-guided object counting.

Usage:
  python infer.py image.jpg "red cars"
  python infer.py images/ "red cars"
  python infer.py images/ "red cars" --recursive
  python infer.py https://example.com/image.jpg "red cars"
  python infer.py image1.jpg https://example.com/image2.png "people" --out results/
  python infer.py --checkpoint checkpoints/count_anything.pt image.jpg "dogs"
"""

import argparse
import mimetypes
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "count_anything.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def _is_url(value):
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_image(url, work_dir, timeout):
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path)).suffix
    if not suffix:
        guessed = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or "")
        suffix = guessed or ".jpg"

    with tempfile.NamedTemporaryFile(
        prefix="count_anything_url_", suffix=suffix, dir=work_dir, delete=False
    ) as tmp:
        tmp_name = tmp.name

    headers = {"User-Agent": "count-anything-infer/1.0"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp_name, "wb") as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, TimeoutError, OSError):
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return Path(tmp_name)


def _display_name(value):
    if _is_url(value):
        parsed = urlparse(value)
        return Path(unquote(parsed.path)).name or parsed.netloc
    return Path(value).name


def _is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _expand_image_inputs(inputs, recursive=False):
    expanded = []
    for value in inputs:
        if _is_url(value):
            expanded.append(value)
            continue

        path = Path(value).expanduser()
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            images = sorted(p for p in iterator if _is_image_file(p))
            if not images:
                print(f"Skip: {value} — no supported image files found", file=sys.stderr)
                continue
            expanded.extend(str(p) for p in images)
            continue

        expanded.append(value)
    return expanded


def _resolve_checkpoint(path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    if not p.exists():
        print(f"Error: checkpoint not found at {p}", file=sys.stderr)
        print("Download it with: python download_checkpoint.py", file=sys.stderr)
        print("Source: https://huggingface.co/MengqiLei/count-anything/resolve/main/count_anything.pt", file=sys.stderr)
        sys.exit(1)
    return str(p)


def main():
    parser = argparse.ArgumentParser(
        description="Count objects in an image using a text query (CountAnything)."
    )
    parser.add_argument("images", nargs="+", help="Path(s), folder(s), or http(s) URL(s) to image(s)")
    parser.add_argument("query", help="Text description of objects to count (e.g. 'red cars')")
    parser.add_argument(
        "--checkpoint", "-c",
        default=str(DEFAULT_CKPT),
        help="Path to count_anything.pt checkpoint",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output directory for visualizations (default: exp/count_anything_inference/)",
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the result in a window (if DISPLAY is available)",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for downloading image URLs (default: 30)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="When an image argument is a folder, include images from subfolders too",
    )
    args = parser.parse_args()

    ckpt = _resolve_checkpoint(args.checkpoint)

    from count_anything import CountAnything

    model = CountAnything(checkpoint=ckpt, output_dir=args.out or str(REPO_ROOT / "exp" / "count_anything_inference"))

    image_inputs = _expand_image_inputs(args.images, recursive=args.recursive)
    if not image_inputs:
        print("Error: no images to process", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="count_anything_infer_") as tmp_dir:
        for img_path in image_inputs:
            if _is_url(img_path):
                print(f"Downloading: {img_path}")
                try:
                    p = _download_image(img_path, tmp_dir, args.download_timeout)
                except Exception as e:
                    print(f"Skip: {img_path} — download failed: {e}", file=sys.stderr)
                    continue
            else:
                p = Path(img_path)
                if not p.exists():
                    print(f"Skip: {img_path} — file not found", file=sys.stderr)
                    continue

            print(f"Processing: {_display_name(img_path)}")
            start_time = time.perf_counter()
            try:
                results = model(str(p.resolve()), args.query)
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                print(f"  Error after {elapsed:.2f}s: {e}", file=sys.stderr)
                continue
            elapsed = time.perf_counter() - start_time

            for r in results:
                saved = r.save()
                print(f"  Count: {r.count}")
                print(f"  Points: {len(r.pred_points)}")
                print(f"  Inference time: {elapsed:.2f}s")
                print(f"  Saved: {saved}")
                if args.show:
                    r.show()
            print()


if __name__ == "__main__":
    main()
