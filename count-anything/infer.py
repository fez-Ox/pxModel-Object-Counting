#!/usr/bin/env python3
"""CLI for CountAnything text-guided object counting.

Usage:
  python infer.py image.jpg "red cars"
  python infer.py image1.jpg image2.jpg "people" --out results/
  python infer.py --checkpoint checkpoints/count_anything.pt image.jpg "dogs"
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "count_anything.pt"


def _resolve_checkpoint(path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    if not p.exists():
        print(f"Error: checkpoint not found at {p}", file=sys.stderr)
        print("Download from: https://huggingface.co/MengqiLei/count-anything", file=sys.stderr)
        sys.exit(1)
    return str(p)


def main():
    parser = argparse.ArgumentParser(
        description="Count objects in an image using a text query (CountAnything)."
    )
    parser.add_argument("images", nargs="+", help="Path(s) to image(s)")
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
    args = parser.parse_args()

    ckpt = _resolve_checkpoint(args.checkpoint)

    from count_anything import CountAnything

    model = CountAnything(checkpoint=ckpt, output_dir=args.out or str(REPO_ROOT / "exp" / "count_anything_inference"))

    for img_path in args.images:
        p = Path(img_path)
        if not p.exists():
            print(f"Skip: {img_path} — file not found", file=sys.stderr)
            continue

        print(f"Processing: {p.name}")
        try:
            results = model(str(p.resolve()), args.query)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            continue

        for r in results:
            saved = r.save()
            print(f"  Count: {r.count}")
            print(f"  Points: {len(r.pred_points)}")
            print(f"  Saved: {saved}")
            if args.show:
                r.show()
        print()


if __name__ == "__main__":
    main()
