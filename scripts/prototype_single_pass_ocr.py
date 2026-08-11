#!/usr/bin/env python3
"""Standalone prototype test for Single-Pass Full-Native-Resolution / Tiled OCR + SAM3 + Overlap Attribution."""

import argparse
import json
import sys
import time
from pathlib import Path
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


def run_single_pass_prototype(
    image_path: Path,
    localizer: any,
    ocr_backend: any,
    gazetteer: Gazetteer,
    config: LocalizationConfig,
    tile_threshold: int = 2500,
) -> dict:
    start_time = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    # --- Stage 1: SAM3 Instance Detection & Scene Filtering ---
    sam3_start = time.perf_counter()
    raw_detections = localizer.detect(image_path)
    raw_instances = detections_to_instances(raw_detections)
    scene_filter = SAM3SceneFilter(localizer)
    instances, _ = scene_filter.filter(image_path, raw_instances)
    poster_regions = getattr(scene_filter, "last_poster_regions", [])
    sam3_time = time.perf_counter() - sam3_start

    # --- Stage 2: Native Full-Resolution / Tiled OCR Pass ---
    ocr_start = time.perf_counter()
    tiles = tile_image(image, tile_size=tile_threshold, overlap=500)
    all_text_detections: list[TextDetection] = []

    for tile, offset_x, offset_y in tiles:
        if hasattr(ocr_backend, "detect_preprocessed"):
            tile_detections = ocr_backend.detect_preprocessed(tile)
        else:
            tile_detections = ocr_backend.detect(tile)
        
        for det in tile_detections:
            gx = det.bbox[0] + offset_x
            gy = det.bbox[1] + offset_y
            all_text_detections.append(TextDetection(
                text=det.text,
                bbox=[gx, gy, det.bbox[2], det.bbox[3]],
                confidence=det.confidence,
                source=det.source,
            ))
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
            "single_pass_ocr_seconds": round(ocr_time, 3),
            "c2_fusion_seconds": round(fusion_time, 3),
            "brand_association_seconds": round(ocr_time + fusion_time, 3),
            "total_pipeline_seconds": round(total_time, 3),
        },
        "counts": {
            "instances": len(instances),
            "physical_signs": len(c2_signs),
            "c1_evidence": len(c1_evidence),
            "c2_evidence": len(c2_evidence),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prototype Single-Pass OCR + SAM3 + Overlap Attribution.")
    parser.add_argument("items", nargs="+", help="Image paths or directory")
    parser.add_argument("--sam3-checkpoint", required=True, help="Path to SAM3 checkpoint")
    parser.add_argument("--brand-file", required=True, help="Gazetteer brand file")
    parser.add_argument("--ocr-backend", default="rapidocr+florence2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/prototype_results")
    args = parser.parse_args()

    brands = [line.strip() for line in Path(args.brand_file).read_text().splitlines() if line.strip()]
    gazetteer = Gazetteer(brands)
    config = LocalizationConfig(gazetteer=brands, enable_highest_confidence_fallback=True)

    localizer = build_native_sam3_localizer(args.sam3_checkpoint, device=args.device)
    ocr_backend = build_ocr_backend(args.ocr_backend, gpu=args.device, gazetteer=gazetteer)

    from eyewear_localization.visualization import annotate

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

    print(f"\n=== PROTOTYPE SINGLE-PASS OCR RUN ({len(images)} images) ===")
    results = {}
    for img_path in images:
        res = run_single_pass_prototype(img_path, localizer, ocr_backend, gazetteer, config)
        results[img_path.stem] = res
        
        # Save JSON output
        out_file = json_dir / f"{img_path.stem}.json"
        out_file.write_text(json.dumps(res, indent=2))
        
        # Render and save visual bounding-box overlay image
        try:
            annotated_img = annotate(img_path, res)
            vis_file = vis_dir / f"{img_path.stem}_annotated.jpg"
            annotated_img.save(vis_file)
        except Exception as exc:
            print(f"  Warning: could not render visualization for {img_path.stem}: {exc}")

        t = res["timings"]
        c = res["counts"]
        print(
            f"  {img_path.stem:12s} | Inst: {c['instances']:2d} | C1 Ev: {c['c1_evidence']:2d} | "
            f"C2 Signs: {c['physical_signs']:2d} | SAM3: {t['sam3_time_seconds']:6.2f}s | "
            f"OCR: {t['single_pass_ocr_seconds']:6.2f}s | Total: {t['total_pipeline_seconds']:6.2f}s"
        )

    print("\nSummary Results Saved to:", base_out_dir)


if __name__ == "__main__":
    main()
