#!/usr/bin/env python3
"""Standalone prototype test for Single-Pass Full-Native-Resolution / Tiled OCR + SAM3 + Overlap Attribution.

Profiles and scene modes:
  --sam3-profile {full,fast}   full: 4 class + 4 signage + 9 scene-filter prompts (legacy)
                               fast: 2 class + 0 signage + 3 scene-filter prompts (~5 prompts)
  --sam3-prompt-batch-size N   batch SAM3 prompts through set_text_prompts
  --florence-scene {full,quick,off}
                               full: one full-frame pass + lower crop + up to 4 column crops
                               quick: one full-frame pass only
                               off: primary OCR only (no Florence scene pass)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from PIL import Image

# Ensure eyewear-localization package is in Python path
repo_root = Path(__file__).resolve().parents[1]
app_dir = repo_root / "eyewear-localization"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.gazetteer import Gazetteer
from eyewear_localization.perception import (
    Florence2OCRBackend,
    RapidOCRBackend,
    SAM3_CLASS_PROMPTS,
    SAM3_SIGNAGE_PROMPTS,
    SAM3Localizer,
    TextDetection,
    build_native_sam3_localizer,
    build_ocr_backend,
    detections_to_instances,
)
from eyewear_localization.scene_filter import SAM3SceneFilter
from eyewear_localization.cues import Evidence, SignageScopeCue
from eyewear_localization.fusion import decide, fuse_evidence, smooth_continuity
from eyewear_localization.schemas import Instance, Sign, Scope

DEFAULT_PERSON_PROMPTS = ("people", "person", "faces of people")
DEFAULT_POSTER_PROMPTS = ("advertisements", "posters", "billboards")
DEFAULT_SHELF_PROMPTS = ("retail shelves", "display shelves", "shelf")

SAM3_PROFILES = {
    "full": {
        "class_prompts": SAM3_CLASS_PROMPTS,
        "signage_prompts": SAM3_SIGNAGE_PROMPTS,
        "person_prompts": DEFAULT_PERSON_PROMPTS,
        "poster_prompts": DEFAULT_POSTER_PROMPTS,
        "shelf_prompts": DEFAULT_SHELF_PROMPTS,
    },
    "fast": {
        "class_prompts": ("sunglasses", "eyeglasses", "glasses"),
        "signage_prompts": (),
        "person_prompts": ("people",),
        "poster_prompts": ("advertisements",),
        "shelf_prompts": ("retail shelves",),
    },
    # Signage-assisted fast profiles: re-enable the SAM3 signage placeholder
    # pass so display text (countertop base plates, tray labels) is OCR'd on
    # upscaled crops instead of only the full frame.
    "fasts": {
        "class_prompts": ("sunglasses", "eyeglasses", "glasses"),
        "signage_prompts": SAM3_SIGNAGE_PROMPTS,
        "person_prompts": ("people",),
        "poster_prompts": ("advertisements",),
        "shelf_prompts": ("retail shelves",),
    },
    # fasts without the scene filter (scene prompts removed nothing on the GT
    # set; keeps SAM3 ~1.4s cheaper when the budget is tight).
    "fastn": {
        "class_prompts": ("sunglasses", "eyeglasses", "glasses"),
        "signage_prompts": SAM3_SIGNAGE_PROMPTS,
        "person_prompts": (),
        "poster_prompts": (),
        "shelf_prompts": (),
    },
}


def tile_image(image: Image.Image, tile_size: int = 1600, overlap: int = 400) -> list[tuple[Image.Image, int, int]]:
    """Split image into overlapping full-resolution sub-tiles ensuring full image coverage."""
    width, height = image.size
    if width <= tile_size and height <= tile_size:
        return [(image, 0, 0)]

    x_coords = []
    x = 0
    while True:
        x_coords.append(min(x, max(0, width - tile_size)))
        if x + tile_size >= width:
            break
        x += tile_size - overlap
    x_coords = sorted(list(set(x_coords)))

    y_coords = []
    y = 0
    while True:
        y_coords.append(min(y, max(0, height - tile_size)))
        if y + tile_size >= height:
            break
        y += tile_size - overlap
    y_coords = sorted(list(set(y_coords)))

    tiles = []
    for top in y_coords:
        for left in x_coords:
            right = min(width, left + tile_size)
            bottom = min(height, top + tile_size)
            tile = image.crop((left, top, right, bottom))
            tiles.append((tile, left, top))
    return tiles


def bbox_intersect(box1: list[float], box2: list[float]) -> bool:
    """Check if box1 [x, y, w, h] intersects or touches box2 [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def process_ocr_batches(
    items: list[tuple[Image.Image, int, int]],
    ocr_backend: Any,
    batch_size: int = 4,
) -> list[TextDetection]:
    """Process PIL image items (tiles or crops) in GPU micro-batches."""
    if not items:
        return []

    detections: list[TextDetection] = []
    batch_size = max(1, int(batch_size))

    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        chunk_images = [img for img, _, _ in chunk]

        if hasattr(ocr_backend, "detect_batch"):
            try:
                chunk_results = ocr_backend.detect_batch(chunk_images)
            except Exception:
                chunk_results = [ocr_backend.detect(img) for img in chunk_images]
        else:
            chunk_results = [ocr_backend.detect(img) for img in chunk_images]

        for (img, offset_x, offset_y), det_list in zip(chunk, chunk_results):
            for det in det_list:
                gx = det.bbox[0] + offset_x
                gy = det.bbox[1] + offset_y
                detections.append(TextDetection(
                    text=det.text,
                    bbox=[gx, gy, det.bbox[2], det.bbox[3]],
                    confidence=det.confidence,
                    source=det.source,
                ))

    return detections


def run_single_pass_prototype(
    image_path: Path,
    localizer: Any,
    ocr_backend: Any,
    gazetteer: Gazetteer,
    config: LocalizationConfig,
    tile_threshold: int = 2500,
    ocr_batch_size: int = 4,
    class_prompts: tuple[str, ...] = SAM3_CLASS_PROMPTS,
    signage_prompts: tuple[str, ...] = SAM3_SIGNAGE_PROMPTS,
    person_prompts: tuple[str, ...] = DEFAULT_PERSON_PROMPTS,
    poster_prompts: tuple[str, ...] = DEFAULT_POSTER_PROMPTS,
    shelf_prompts: tuple[str, ...] = DEFAULT_SHELF_PROMPTS,
    retry_threshold: float = 0.15,
    ocr_mode: str = "tiled",
    crop_ocr_backend: Any = None,
) -> dict:
    start_time = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    # --- Stage 1: SAM3 Instance Detection, Signage Placard Segmentation & Scene Filtering ---
    sam3_start = time.perf_counter()

    class_list = list(class_prompts)
    if hasattr(localizer, "detect_classes"):
        raw_detections = localizer.detect_classes(image_path, class_list)
        if not raw_detections:
            raw_detections = localizer.detect_classes(
                image_path, class_list, threshold=retry_threshold
            )
    else:
        raw_detections = localizer.detect(image_path)
    raw_instances = detections_to_instances(raw_detections)
    sam3_class_time = time.perf_counter() - sam3_start

    signage_start = time.perf_counter()
    signage_crops: list[tuple[Image.Image, int, int]] = []
    signage_boxes: list[list[float]] = []
    if signage_prompts and hasattr(localizer, "detect_prompts"):
        try:
            signage_per_prompt = localizer.detect_prompts(
                image_path,
                signage_prompts,
                thresholds=[0.20] * len(signage_prompts),
            )
            for prompt_dets in signage_per_prompt:
                for det in prompt_dets:
                    bx, by, bw, bh = [int(v) for v in det.box]
                    # Clamp bounding box
                    x1 = max(0, bx)
                    y1 = max(0, by)
                    x2 = min(img_width, bx + bw)
                    y2 = min(img_height, by + bh)
                    if x2 - x1 > 20 and y2 - y1 > 10:
                        crop = image.crop((x1, y1, x2, y2))
                        signage_crops.append((crop, x1, y1))
                        signage_boxes.append([x1, y1, x2 - x1, y2 - y1])
        except Exception:
            pass
    sam3_signage_time = time.perf_counter() - signage_start

    scene_start = time.perf_counter()
    scene_filter = SAM3SceneFilter(
        localizer,
        person_prompts=person_prompts,
        poster_prompts=poster_prompts,
        shelf_prompts=shelf_prompts,
    )
    instances, _ = scene_filter.filter(image_path, raw_instances)
    poster_regions = getattr(scene_filter, "last_poster_regions", [])
    sam3_scene_time = time.perf_counter() - scene_start
    sam3_time = time.perf_counter() - sam3_start

    # --- Stage 2: Full-Resolution / Tiled OCR + SAM3 Signage Crop Pass ---
    ocr_start = time.perf_counter()

    if ocr_mode == "single":
        # One native-resolution full-frame pass: RapidOCR gains nothing from
        # tiling (no upscale beyond max_dimension), and a single detect also
        # gives the SelectiveOCR scene pass the whole image instead of a tile.
        tiles = [(image, 0, 0)]
        tile_detections: list[TextDetection] = process_ocr_batches(
            tiles, ocr_backend, batch_size=1
        )
    else:
        tiles = tile_image(image, tile_size=tile_threshold, overlap=400)
        # Process tiled full-resolution passes in GPU micro-batches
        tile_detections = process_ocr_batches(
            tiles, ocr_backend, batch_size=ocr_batch_size
        )
    ocr_tiles_time = time.perf_counter() - ocr_start

    # Process SAM3 segmented signage placard crops in GPU micro-batches.
    # Signage crops are OCR'd at 2x (crop_ocr_backend) so small display text
    # like countertop base-plate logos is legible; the upscale stays under the
    # RapidOCR max_dimension cap for crop-sized images.
    all_text_detections: list[TextDetection] = list(tile_detections)
    if signage_crops:
        crop_backend = crop_ocr_backend if crop_ocr_backend is not None else ocr_backend
        placard_detections = process_ocr_batches(
            signage_crops, crop_backend, batch_size=ocr_batch_size
        )
        all_text_detections.extend(placard_detections)
    ocr_signage_time = time.perf_counter() - ocr_start - ocr_tiles_time
    ocr_time = time.perf_counter() - ocr_start

    # --- Stage 3: Decoupled Spatial Attribution (C1 On-Product vs C2 Signage) ---
    c1_evidence: list[Evidence] = []
    c2_signs: list[Sign] = []

    for idx, det in enumerate(all_text_detections, start=1):
        match = gazetteer.match(det.text)
        if match is None:
            continue

        matched_instance = None
        for inst in instances:
            if bbox_intersect(det.bbox, inst.bbox):
                matched_instance = inst
                break

        if matched_instance is not None:
            c1_evidence.append(Evidence(
                instance_id=matched_instance.id,
                brand=match.brand,
                confidence=det.confidence * match.score,
                cue="C1",
            ))
        else:
            c2_signs.append(Sign(
                sign_id=f"s_{idx:02d}",
                text=det.text,
                brand=match.brand,
                bbox=det.bbox,
                scope=Scope(),
                confidence=det.confidence * match.score,
            ))

    # --- Stage 4: C2 Signage Scope & Precision Cascade ---
    fusion_start = time.perf_counter()
    c2_cue = SignageScopeCue()
    scoped_signs, c2_evidence = c2_cue.associate(
        instances,
        c2_signs,
        poster_regions,
        image_width=float(img_width),
        text_detections=all_text_detections,
    )

    total_evidence = list(c1_evidence) + list(c2_evidence)
    probs = fuse_evidence(instances, total_evidence, config)
    probs = smooth_continuity(probs, instances, config)
    outputs = decide(instances, probs, total_evidence, config)
    fusion_time = time.perf_counter() - fusion_start

    total_time = time.perf_counter() - start_time

    return {
        "schema_version": "1.0",
        "image": str(image_path),
        "instances": [inst.to_dict() for inst in instances],
        "signs": [sign.to_dict() for sign in scoped_signs],
        "evidence": [ev.to_dict() for ev in total_evidence],
        "outputs": [out.to_dict() for out in outputs],
        "timings": {
            "sam3_time_seconds": round(sam3_time, 3),
            "sam3_class_seconds": round(sam3_class_time, 3),
            "sam3_signage_seconds": round(sam3_signage_time, 3),
            "sam3_scene_seconds": round(sam3_scene_time, 3),
            "single_pass_ocr_seconds": round(ocr_time, 3),
            "ocr_tiles_seconds": round(ocr_tiles_time, 3),
            "ocr_signage_crops_seconds": round(ocr_signage_time, 3),
            "c2_fusion_seconds": round(fusion_time, 3),
            "brand_association_seconds": round(ocr_time + fusion_time, 3),
            "total_pipeline_seconds": round(total_time, 3),
        },
        "counts": {
            "instances": len(instances),
            "physical_signs": len(c2_signs),
            "c1_evidence": len(c1_evidence),
            "c2_evidence": len(c2_evidence),
            "sam3_prompts": len(class_list)
            + len(signage_prompts)
            + len(person_prompts)
            + len(poster_prompts)
            + len(shelf_prompts),
        },
    }


def _parse_profile_specs(raw: str) -> list[dict]:
    """Parse '<sam3>[:<florence>[:<ocr_mode>[:<ocr_scale>[:<backend>]]]]' tokens.

    ':'-separated fields with defaults for the omitted parts, e.g.
    'fast:quick:single:1.0', 'fast:off', or 'full/full' (legacy '/' form).
    A 5th field overrides --ocr-backend for that spec.
    """
    specs: list[dict] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        token = token.replace("/", ":")
        parts = [part.strip() for part in token.split(":")]
        sam3_profile = parts[0] or "full"
        florence = parts[1] if len(parts) > 1 and parts[1] else "full"
        ocr_mode = parts[2] if len(parts) > 2 and parts[2] else "tiled"
        ocr_scale = parts[3] if len(parts) > 3 and parts[3] else None
        backend = parts[4] if len(parts) > 4 and parts[4] else None
        if sam3_profile not in SAM3_PROFILES:
            raise SystemExit(f"unknown --benchmark-profiles sam3 profile: {sam3_profile!r} "
                             f"(choose from {sorted(SAM3_PROFILES)})")
        if ocr_mode not in ("tiled", "single"):
            raise SystemExit(f"unknown ocr_mode in profile spec: {ocr_mode!r}")
        if florence not in ("full", "quick", "off"):
            raise SystemExit(f"unknown florence scene in profile spec: {florence!r}")
        specs.append({
            "sam3_profile": sam3_profile,
            "florence": florence,
            "ocr_mode": ocr_mode,
            "ocr_scale": float(ocr_scale) if ocr_scale is not None else None,
            "backend": backend,
        })
    return specs


def run_images(
    images: list[Path],
    localizer: Any,
    ocr_backend: Any,
    gazetteer: Gazetteer,
    config: LocalizationConfig,
    profile: dict,
    ocr_batch_size: int,
    json_dir: Path,
    vis_dir: Path,
    save_outputs: bool,
    out_tag: str | None = None,
    ocr_mode: str = "tiled",
    crop_ocr_backend: Any = None,
) -> dict:
    results: dict = {}
    total_sam3 = 0.0
    total_ocr = 0.0
    total_pipeline = 0.0
    if out_tag:
        json_dir = json_dir / out_tag
        vis_dir = vis_dir / out_tag
        json_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        # Fresh fallback budget per image so the bounded Florence scene pass
        # runs once per image (matching pipeline.py), not only on the first.
        if hasattr(ocr_backend, "reset_budget"):
            ocr_backend.reset_budget()
        res = run_single_pass_prototype(
            img_path,
            localizer,
            ocr_backend,
            gazetteer,
            config,
            ocr_batch_size=ocr_batch_size,
            class_prompts=profile["class_prompts"],
            signage_prompts=profile["signage_prompts"],
            person_prompts=profile["person_prompts"],
            poster_prompts=profile["poster_prompts"],
            shelf_prompts=profile["shelf_prompts"],
            ocr_mode=ocr_mode,
            crop_ocr_backend=crop_ocr_backend,
        )
        results[img_path.stem] = res

        if save_outputs:
            out_file = json_dir / f"{img_path.stem}.json"
            out_file.write_text(json.dumps(res, indent=2))
            try:
                from eyewear_localization.visualization import annotate

                annotated_img = annotate(img_path, res)
                vis_file = vis_dir / f"{img_path.stem}_annotated.jpg"
                annotated_img.save(vis_file)
            except Exception as exc:
                print(f"  Warning: could not render visualization for {img_path.stem}: {exc}")

        t = res["timings"]
        c = res["counts"]
        total_sam3 += t["sam3_time_seconds"]
        total_ocr += t["single_pass_ocr_seconds"]
        total_pipeline += t["total_pipeline_seconds"]

        print(
            f"  {img_path.stem:12s} | Inst: {c['instances']:2d} | C1 Ev: {c['c1_evidence']:2d} | "
            f"C2 Signs: {c['physical_signs']:2d} | Prompts: {c['sam3_prompts']:2d} | "
            f"SAM3: {t['sam3_time_seconds']:6.2f}s | OCR: {t['single_pass_ocr_seconds']:6.2f}s | "
            f"Total: {t['total_pipeline_seconds']:6.2f}s"
        )

    count = max(1, len(images))
    return {
        "results": results,
        "total_sam3_seconds": total_sam3,
        "total_ocr_seconds": total_ocr,
        "total_pipeline_seconds": total_pipeline,
        "avg_sam3_seconds": total_sam3 / count,
        "avg_ocr_seconds": total_ocr / count,
        "avg_total_seconds": total_pipeline / count,
    }


def main():
    parser = argparse.ArgumentParser(description="Prototype Single-Pass OCR + SAM3 + Overlap Attribution.")
    parser.add_argument("items", nargs="+", help="Image paths or directory")
    parser.add_argument("--sam3-checkpoint", required=True, help="Path to SAM3 checkpoint")
    parser.add_argument("--brand-file", required=True, help="Gazetteer brand file")
    parser.add_argument("--ocr-backend", default="rapidocr+florence2")
    parser.add_argument("--ocr-batch-size", type=int, default=4, help="Micro-batch size for GPU OCR sub-tile/placard inference (default: 4)")
    parser.add_argument("--ocr-scale", type=float, default=1.0, help="Upscale factor before OCR (lower = faster; default: 1.0)")
    parser.add_argument("--ocr-mode", choices=["tiled", "single"], default="tiled",
                        help="tiled: overlapping 2500px tiles; single: one full-frame pass (default: tiled)")
    parser.add_argument("--florence-scene", choices=["full", "quick", "off"], default="full",
                        help="full: full-frame + lower + column passes; quick: single full pass; off: no Florence scene pass")
    parser.add_argument("--sam3-profile", choices=list(SAM3_PROFILES), default="full",
                        help="full: 4 class + 4 signage + 9 scene; fast: 3 class + 0 signage + 3 scene; "
                             "fasts: fast + signage pass; fastn: fasts without scene filter")
    parser.add_argument("--sam3-prompt-batch-size", type=int, default=4,
                        help="Batch SAM3 prompts through set_text_prompts (default: 4)")
    parser.add_argument("--benchmark-batch-sizes", type=str, default=None, help="Comma-separated batch sizes to benchmark (e.g. '1,2,4,8')")
    parser.add_argument("--max-benchmark-images", type=int, default=None, help="Limit number of images used for batch size benchmarking (e.g. 3)")
    parser.add_argument("--benchmark-profiles", type=str, default=None,
                        help="Comma-separated '<sam3>:<florence>:<mode>:<scale>' specs, e.g. 'fast:quick:single:1.0,fast:off:single:1.0'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/prototype_results")
    args = parser.parse_args()

    brands = [line.strip() for line in Path(args.brand_file).read_text().splitlines() if line.strip()]
    gazetteer = Gazetteer(brands)
    config = LocalizationConfig(gazetteer=brands, enable_highest_confidence_fallback=True)

    localizer = build_native_sam3_localizer(
        args.sam3_checkpoint,
        device=args.device,
        prompt_batch_size=args.sam3_prompt_batch_size,
    )

    from eyewear_localization.visualization import annotate  # noqa: F401

    base_out_dir = Path(args.out)
    json_dir = base_out_dir / "json"
    vis_dir = base_out_dir / "visualizations"
    json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for item in args.items:
        p = Path(item)
        if p.is_dir():
            images.extend(sorted(p.glob("*.jpg")) + sorted(p.glob("*.png")))
        elif p.is_file():
            images.append(p)

    def _build_ocr(florence: str, ocr_scale: float, backend_name: str | None = None) -> Any:
        backend = build_ocr_backend(
            backend_name or args.ocr_backend,
            gpu=args.device,
            gazetteer=gazetteer,
            scale=ocr_scale,
            florence_scene=florence,
        )
        if hasattr(backend, "max_fallback_calls"):
            backend.max_fallback_calls = 99
        return backend

    def _build_crop_ocr_backend() -> Any:
        """OCR backend for SAM3 signage-detect crops.

        RapidOCR at 2x upscale: crop-sized inputs stay under the max_dimension
        cap so display text gets a true 2x resolution boost with no VLM cost.
        """
        backend = globals().get("_crop_ocr_backend_cache")
        if backend is not None:
            return backend
        try:
            from eyewear_localization.perception import RapidOCRBackend

            backend = RapidOCRBackend(scale=2.0)
        except Exception:
            backend = None
        globals()["_crop_ocr_backend_cache"] = backend
        return backend


    if args.benchmark_profiles:
        specs = _parse_profile_specs(args.benchmark_profiles)
        print(f"\n=== PROFILE BENCHMARK ({len(images)} images | {len(specs)} specs) ===")
        report: dict[str, dict] = {}
        ocr_backend = None
        prev_key = None
        crop_ocr_backend = _build_crop_ocr_backend()
        for spec in specs:
            sam3_profile = spec["sam3_profile"]
            florence = spec["florence"]
            ocr_mode = spec["ocr_mode"]
            ocr_scale = spec["ocr_scale"] if spec["ocr_scale"] is not None else args.ocr_scale
            backend_spec = spec["backend"]
            key = (backend_spec, florence, ocr_mode, ocr_scale)
            if key != prev_key:
                ocr_backend = _build_ocr(florence, ocr_scale, backend_name=backend_spec)
                prev_key = key
            profile = SAM3_PROFILES[sam3_profile]
            label = f"{sam3_profile}/{florence}/{ocr_mode}@{ocr_scale}"
            if backend_spec and backend_spec != args.ocr_backend:
                label = f"{label}+{backend_spec}"
            n_prompts = sum(len(profile[key]) for key in
                            ("class_prompts", "signage_prompts", "person_prompts", "poster_prompts", "shelf_prompts"))
            print(f"\n--- profile {label} | {n_prompts} prompts | mode={ocr_mode} scale={ocr_scale} ---")
            run = run_images(
                images, localizer, ocr_backend, gazetteer, config, profile,
                args.ocr_batch_size, json_dir, vis_dir, save_outputs=True,
                out_tag=label.replace("/", "__").replace("@", "x"), ocr_mode=ocr_mode,
                crop_ocr_backend=crop_ocr_backend,
            )
            report[label] = {
                "sam3_profile": sam3_profile,
                "florence_scene": florence,
                "ocr_mode": ocr_mode,
                "ocr_scale": ocr_scale,
                "ocr_backend": backend_spec or args.ocr_backend,
                "images": len(images),
                "sam3_prompts": n_prompts,
                "avg_sam3_seconds": round(run["avg_sam3_seconds"], 3),
                "avg_ocr_seconds": round(run["avg_ocr_seconds"], 3),
                "avg_total_seconds": round(run["avg_total_seconds"], 3),
                "total_sam3_seconds": round(run["total_sam3_seconds"], 3),
                "total_ocr_seconds": round(run["total_ocr_seconds"], 3),
                "total_pipeline_seconds": round(run["total_pipeline_seconds"], 3),
            }

        if len(report) > 1:
            print("\n" + "=" * 80)
            print("=== PROFILE BENCHMARK SUMMARY ===")
            print("=" * 80)
            labels = list(report)
            baseline_total = report[labels[0]]["avg_total_seconds"]
            baseline_sam3 = report[labels[0]]["avg_sam3_seconds"]
            print(f"{'Profile':<16} | {'SAM3':<8} | {'OCR':<8} | {'Total':<8} | {'Total Speedup':<16}")
            print("-" * 70)
            for label in labels:
                r = report[label]
                speedup = baseline_total / max(0.001, r["avg_total_seconds"])
                print(f"{label:<16} | {r['avg_sam3_seconds']:<8.2f}s | {r['avg_ocr_seconds']:<8.2f}s | "
                      f"{r['avg_total_seconds']:<8.2f}s | {speedup:<16.2f}x")
            bench_file = base_out_dir / "profile_benchmark_results.json"
            bench_file.write_text(json.dumps(report, indent=2))
            print(f"\nBenchmark metrics saved to: {bench_file}")
        return

    if args.benchmark_batch_sizes:
        batch_sizes_to_test = [int(b.strip()) for b in args.benchmark_batch_sizes.split(",") if b.strip()]
        benchmark_images = images
        if args.max_benchmark_images and len(images) > args.max_benchmark_images:
            benchmark_images = images[:args.max_benchmark_images]
    else:
        batch_sizes_to_test = [args.ocr_batch_size]
        benchmark_images = images

    profile = SAM3_PROFILES[args.sam3_profile]
    ocr_backend = _build_ocr(args.florence_scene, args.ocr_scale)
    crop_ocr_backend = _build_crop_ocr_backend()
    benchmark_report = {}

    for bs in batch_sizes_to_test:
        current_images = benchmark_images if len(batch_sizes_to_test) > 1 else images
        print(f"\n=== PROTOTYPE OCR RUN ({len(current_images)} images | ocr_batch_size={bs} | "
              f"sam3={args.sam3_profile} | florence={args.florence_scene} | scale={args.ocr_scale}) ===")
        run = run_images(
            current_images, localizer, ocr_backend, gazetteer, config, profile,
            bs, json_dir, vis_dir, save_outputs=True,
            ocr_mode=args.ocr_mode,
            crop_ocr_backend=crop_ocr_backend,
        )
        benchmark_report[bs] = {
            "batch_size": bs,
            "avg_ocr_seconds": run["avg_ocr_seconds"],
            "avg_total_seconds": run["avg_total_seconds"],
            "total_ocr_seconds": run["total_ocr_seconds"],
            "total_pipeline_seconds": run["total_pipeline_seconds"],
        }

    if len(benchmark_report) > 1:
        print("\n" + "=" * 80)
        print("=== BATCH SIZE BENCHMARK SUMMARY ===")
        print("=" * 80)
        baseline_ocr = benchmark_report[batch_sizes_to_test[0]]["avg_ocr_seconds"]
        print(f"{'Batch Size (B)':<15} | {'Avg OCR Time':<15} | {'Avg Total Time':<15} | {'OCR Speedup':<15}")
        print("-" * 70)
        for bs, r in benchmark_report.items():
            speedup = baseline_ocr / max(0.001, r['avg_ocr_seconds'])
            print(f"{bs:<15d} | {r['avg_ocr_seconds']:<15.2f}s | {r['avg_total_seconds']:<15.2f}s | {speedup:<15.2f}x")

        bench_file = base_out_dir / "ocr_batch_benchmark_results.json"
        bench_file.write_text(json.dumps(benchmark_report, indent=2))
        print(f"\nBenchmark metrics saved to: {bench_file}")

    print("\nSummary Results Saved to:", base_out_dir)


if __name__ == "__main__":
    main()
