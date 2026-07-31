from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return value or "query"


@dataclass
class CountAnythingResult:
    image_path: str
    text_query: str
    count: int
    pred_points: List[Dict[str, float]]
    run_dir: str
    peak_vram_mb: float | None = None
    peak_reserved_vram_mb: float | None = None

    def _render(self):
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError("Pillow is required for visualization.") from exc

        image = Image.open(self.image_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        radius = max(4, round(min(width, height) / 180))
        for point in self.pred_points:
            x = max(0.0, min(float(point["x"]), float(width - 1)))
            y = max(0.0, min(float(point["y"]), float(height - 1)))
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(230, 40, 35, 230),
                outline=(255, 255, 255, 255),
                width=max(1, radius // 2),
            )
        label = f"{self.text_query}: {self.count}"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, width // 45))
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        pad = 10
        draw.rectangle(
            [0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2],
            fill=(0, 0, 0, 170),
        )
        draw.text((pad, pad), label, fill=(255, 255, 255, 255), font=font)
        return image

    def show(self) -> None:
        self._render().show()

    def _json_payload(self) -> Dict:
        return {
            "image_path": self.image_path,
            "text_query": self.text_query,
            "count": self.count,
            "points": self.pred_points,
            "peak_vram_mb": self.peak_vram_mb,
            "peak_reserved_vram_mb": self.peak_reserved_vram_mb,
        }

    def save_json(self, path: str | os.PathLike | None = None) -> str:
        if path is None:
            filename = f"{Path(self.image_path).stem}__{_slugify(self.text_query)}.json"
            path = Path(self.run_dir) / filename
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._json_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def save(self, path: str | os.PathLike | None = None) -> str:
        if path is None:
            filename = f"{Path(self.image_path).stem}__{_slugify(self.text_query)}.jpg"
            path = Path(self.run_dir) / filename
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._render().save(path)
        self.save_json(path.with_suffix(".json"))
        return str(path)


class CountAnything:
    def __init__(
        self,
        checkpoint: str | os.PathLike = "checkpoints/count_anything.pt",
        *,
        config: str | os.PathLike = "config/count_anything_test_cloc.yaml",
        output_dir: str | os.PathLike = "exp/count_anything_inference",
        num_gpus: int = 1,
        python_executable: str = sys.executable,
        device: str | None = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.checkpoint = self._resolve_path(checkpoint)
        self.config = self._resolve_path(config)
        self.output_dir = self._resolve_path(output_dir)
        self.num_gpus = int(num_gpus)
        self.python_executable = python_executable
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._runtime = None

    def __call__(self, image_path: str | os.PathLike, text_query: str) -> List[CountAnythingResult]:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        text_query = str(text_query).strip()
        if not text_query:
            raise ValueError("text_query must be a non-empty string")

        run_dir = self._make_run_dir(image_path, text_query)
        annotation_path = run_dir / "temporary_inference_annotation.json"

        self._write_annotation(annotation_path, image_path, text_query)
        record = self._run_direct(annotation_path)
        return [
            CountAnythingResult(
                image_path=str(image_path),
                text_query=text_query,
                count=int(record["pred_count"]),
                pred_points=record.get("points", []),
                run_dir=str(run_dir),
                peak_vram_mb=record.get("peak_vram_mb"),
                peak_reserved_vram_mb=record.get("peak_reserved_vram_mb"),
            )
        ]

    def _resolve_path(self, path: str | os.PathLike) -> Path:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _make_run_dir(self, image_path: Path, text_query: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{image_path.stem}__{_slugify(text_query)}__{stamp}"
        run_dir = self.output_dir / name
        index = 2
        while run_dir.exists():
            run_dir = self.output_dir / f"{name}_{index}"
            index += 1
        run_dir.mkdir(parents=True)
        return run_dir

    def _write_annotation(self, path: Path, image_path: Path, text_query: str) -> None:
        annotation = {
            f"0__{_slugify(text_query)}": {
                "idx": 0,
                "image_path": str(image_path),
                "image_from": "custom",
                "classes": [text_query],
                "annotation": {text_query: {"point": []}},
                "split": "inference",
                "selected_classes": [text_query],
                "selected_annotation": {text_query: {"point": []}},
            }
        }
        path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_runtime(self):
        if self._runtime is not None:
            return self._runtime

        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        with self.config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_conf = config["trainer"]["model"]
        model_conf["device"] = self.device
        model_conf["eval_mode"] = True
        model_conf["checkpoint_path"] = str(self.checkpoint)
        model_conf["load_from_HF"] = False

        postprocessor_conf = config["trainer"]["meters"]["val"]["all"]["counting"]["postprocessor"]
        dataset_conf = config["trainer"]["data"]["val"]["dataset"]
        collate_conf = config["trainer"]["data"]["val"]["collate_fn"]

        model = instantiate(OmegaConf.create(model_conf), _convert_="all")
        model.eval()
        postprocessor = instantiate(OmegaConf.create(postprocessor_conf), _convert_="all")
        collate_fn = instantiate(OmegaConf.create(collate_conf), _convert_="all")

        self._runtime = {
            "model": model,
            "postprocessor": postprocessor,
            "dataset_conf": dataset_conf,
            "collate_fn": collate_fn,
        }
        return self._runtime

    def _run_direct(self, annotation_path: Path) -> Dict:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam3.model.utils.misc import copy_data_to_device

        runtime = self._load_runtime()
        dataset_conf = dict(runtime["dataset_conf"])
        dataset_conf["ann_file"] = str(annotation_path)
        dataset_conf["img_folder"] = "/"
        dataset_conf["training"] = False
        dataset_conf["use_caching"] = False
        dataset = instantiate(OmegaConf.create(dataset_conf), _convert_="all")
        batch_dict = runtime["collate_fn"]([dataset[0]])
        _, batch = batch_dict.popitem()
        batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

        device = torch.device(self.device)
        is_cuda = device.type == "cuda"
        if is_cuda:
            # Synchronize so allocations from preprocessing are not counted as
            # asynchronous work from the forward pass, then start a fresh peak.
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=is_cuda,
                dtype=torch.bfloat16,
            ):
                find_stages = runtime["model"](batch)

        if is_cuda:
            # CUDA kernels are asynchronous; synchronize before reading peak stats.
            torch.cuda.synchronize(device)
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved_vram_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            peak_vram_mb = None
            peak_reserved_vram_mb = None

        for stage_outputs, stage_meta in zip(find_stages, batch.find_metadatas):
            predictions = runtime["postprocessor"](
                outputs=stage_outputs,
                original_sizes=stage_meta.original_size,
                processed_scales_xy=getattr(stage_meta, "processed_scale_xy", None),
                processed_offsets_xy=getattr(stage_meta, "processed_offset_xy", None),
            )
            if predictions:
                pred = predictions[0]
                raw_points = pred["pred_count_points"].detach().cpu().tolist()
                raw_scores = pred["pred_count_scores"].detach().cpu().tolist()
                points = [
                    {"x": float(point[0]), "y": float(point[1]), "score": float(score)}
                    for point, score in zip(raw_points, raw_scores)
                ]
                return {
                    "pred_count": int(pred["pred_count"]),
                    "points": points,
                    "scores": raw_scores,
                    "peak_vram_mb": peak_vram_mb,
                    "peak_reserved_vram_mb": peak_reserved_vram_mb,
                }
        raise RuntimeError("No prediction records were produced")


__all__ = ["CountAnything", "CountAnythingResult"]
