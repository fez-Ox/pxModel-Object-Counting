"""Command-line entry point for the staged eyewear localization pipeline."""

from __future__ import annotations

import argparse
import json as _json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from eyewear_localization.config import load_config
from eyewear_localization.cues import StylePriorCue
from eyewear_localization.gazetteer import Gazetteer, normalize_text
from eyewear_localization.io import materialize_inputs, write_json
from eyewear_localization.perception import (
    HeuristicLocalizer,
    NullPosterBackend,
    PerceptionFrontend,
    build_native_sam3_localizer,
    build_ocr_backend,
)
from eyewear_localization.pipeline import LocalizationPipeline
from eyewear_localization.scene_filter import build_scene_filter
from eyewear_localization.visualization import annotate

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = APP_ROOT / "config.yaml"


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem)
    return value.strip("_") or "image"


def _parse_value(raw: str) -> Any:
    """Parse a CLI value as JSON (numbers/bools/strings) or return raw string."""
    try:
        return _json.loads(raw)
    except _json.JSONDecodeError:
        return raw


def set_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the nested config overrides dict from --set and sugar flags.

    Scene thresholds are gathered under a ``scene`` key (consumed by the
    scene filter, not the fusion config).
    """
    overrides: dict[str, Any] = {}

    def assign(dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    for value in args.set:
        if "=" not in value:
            raise ValueError(f"--set expects DOTTED.KEY=VALUE, got {value!r}")
        dotted, _, raw = value.partition("=")
        assign(dotted.strip(), _parse_value(raw))

    # Convenience sugar flags → dotted keys
    for key, value in (
        ("fusion.tau", args.tau),
        ("fusion.margin", args.margin),
        ("fusion.temperature", args.temperature),
        ("fusion.unknown_prior", args.unknown_prior),
        ("smoothing.lambda", args.smooth_lambda),
        ("smoothing.gate_punknown", args.smooth_gate),
        ("scene.person_threshold", args.person_threshold),
        ("scene.poster_threshold", args.poster_threshold),
        ("scene.shelf_threshold", args.shelf_threshold),
    ):
        if value is not None:
            assign(key, value)

    for value in args.cue_reliability:
        if "=" not in value:
            raise ValueError(f"--cue-reliability expects CUE=RELIABILITY, got {value!r}")
        cue, _, raw = value.partition("=")
        assign(f"cue_reliability.{cue.strip()}", _parse_value(raw))

    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Class-agnostic eyewear localization plus independent OCR/fusion attribution."
    )
    parser.add_argument("items", nargs="+", help="image paths, folders, or HTTP(S) URLs")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML configuration")
    parser.add_argument("--brand", action="append", default=[], help="Add one gazetteer brand; repeatable")
    parser.add_argument("--brand-file", help="Text file containing one gazetteer brand per line")
    parser.add_argument("--ocr-backend", default="easyocr", choices=("easyocr", "none"))
    parser.add_argument("--ocr-scale", type=float, default=2.0, help="OCR upscale factor")
    parser.add_argument("--sam3-checkpoint", help="Optional native SAM3 checkpoint")
    parser.add_argument("--device", default=None, help="SAM3 device, when enabled")
    parser.add_argument("--sam3-threshold", type=float, default=0.25)
    parser.add_argument(
        "--shelf-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter detections outside detected retail shelves when shelf boxes exist",
    )

    # --- config.yaml overrides -------------------------------------------------
    parser.add_argument(
        "--set", action="append", default=[], metavar="DOTTED.KEY=VALUE",
        help="Override any config key, e.g. --set fusion.tau=0.5 "
        "--set smoothing.lambda=0.4 --set cue_reliability.C2=0.8. Repeatable.",
    )
    for flag, _key, kind in (
        ("--tau", "fusion.tau", float),
        ("--margin", "fusion.margin", float),
        ("--temperature", "fusion.temperature", float),
        ("--unknown-prior", "fusion.unknown_prior", float),
        ("--smooth-lambda", "smoothing.lambda", float),
        ("--smooth-gate", "smoothing.gate_punknown", float),
        ("--person-threshold", "scene.person_threshold", float),
        ("--poster-threshold", "scene.poster_threshold", float),
        ("--shelf-threshold", "scene.shelf_threshold", float),
    ):
        parser.add_argument(flag, type=kind, default=None, help=f"Override {_key}")
    parser.add_argument(
        "--cue-reliability", action="append", default=[], metavar="CUE=RELIABILITY",
        help="Override one cue reliability, e.g. --cue-reliability C2=0.85. Repeatable.",
    )
    # ---

    parser.add_argument("--recursive", "-r", action="store_true")
    parser.add_argument("--out", "-o", default=str(APP_ROOT / "outputs"))
    parser.add_argument("--download-timeout", type=int, default=30)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-vlm-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)

    # Apply overrides
    overrides = set_overrides_from_args(args)
    config = config.with_overrides(overrides)

    brands = list(config.gazetteer) + list(args.brand)
    if args.brand_file:
        brands.extend(
            line.strip()
            for line in Path(args.brand_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    config.gazetteer = sorted({normalize_text(brand) for brand in brands if normalize_text(brand)})

    # Scene thresholds from overrides (or config.yaml defaults)
    scene = overrides.get("scene", {})
    ocr = build_ocr_backend(args.ocr_backend, gpu=args.device == "cuda", scale=args.ocr_scale)
    if args.sam3_checkpoint:
        localizer = build_native_sam3_localizer(
            args.sam3_checkpoint,
            device=args.device,
            threshold=args.sam3_threshold,
        )
    else:
        localizer = HeuristicLocalizer("no SAM3 checkpoint supplied")
    frontend = PerceptionFrontend(
        localizer=localizer,
        ocr=ocr,
        gazetteer=Gazetteer(config.gazetteer),
        poster_detector=NullPosterBackend("poster regions supplied by scene filter"),
        scene_filter=build_scene_filter(
            localizer,
            require_shelf=args.shelf_filter,
            person_threshold=float(scene.get("person_threshold", 0.25)),
            poster_threshold=float(scene.get("poster_threshold", 0.25)),
            shelf_threshold=float(scene.get("shelf_threshold", 0.20)),
        ),
    )
    if args.no_vlm_audit:
        config.use_vlm_audit = False
    pipeline = LocalizationPipeline(frontend, config=config, c4=StylePriorCue())
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with materialize_inputs(args.items, recursive=args.recursive, timeout=args.download_timeout) as sources:
            for source in sources:
                stem = _slug(source.source)
                try:
                    result = pipeline.run(source.path, source=source.source)
                except Exception as exc:
                    print(f"ERROR {source.source}: {exc}", file=sys.stderr)
                    continue
                json_path = output_dir / f"{stem}.json"
                write_json(json_path, result)
                if not args.no_visualization:
                    try:
                        image = annotate(source.path, result)
                        image.save(output_dir / f"{stem}.jpg")
                        if args.show:
                            image.show()
                    except Exception as exc:
                        print(f"WARNING {source.source}: visualization skipped: {exc}", file=sys.stderr)
                summary = ", ".join(
                    f'{item["instance_id"]}={item["brand"]}' for item in result["outputs"]
                ) or "no instances"
                print(f"{source.source}: {summary} -> {json_path}")
    except Exception as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
