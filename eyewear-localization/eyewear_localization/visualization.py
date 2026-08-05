"""Auditable image annotations for perception and attribution outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def annotate(image_path: str | Path, result: dict[str, Any]):
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, image.width // 70))
    except Exception:
        font = ImageFont.load_default()

    outputs = {item["instance_id"]: item for item in result.get("outputs", [])}
    for instance in result.get("instances", []):
        x, y, width, height = instance["bbox"]
        decision = outputs.get(instance["id"], {})
        brand = decision.get("brand", "unknown")
        color = (35, 180, 75, 230) if brand != "unknown" else (230, 155, 30, 230)
        draw.rectangle((x, y, x + width, y + height), outline=color, width=max(2, image.width // 500))
        label = f'{instance["id"]}: {brand}'
        if decision.get("abstained"):
            label += " (abstain)"
        text_box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)

    for sign in result.get("signs", []):
        x, y, width, height = sign["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline=(35, 110, 235, 230), width=2)
        label = f'{sign["text"]} → {sign.get("brand") or "candidate"}'
        text_box = draw.textbbox((x, y + height), label, font=font)
        draw.rectangle(text_box, fill=(35, 110, 235, 210))
        draw.text((x, y + height), label, fill=(255, 255, 255, 255), font=font)

    for excluded in result.get("excluded_instances", []):
        x, y, width, height = excluded["bbox"]
        draw.rectangle(
            (x, y, x + width, y + height),
            outline=(190, 60, 60, 230),
            width=max(1, image.width // 700),
        )
        label = f'{excluded["instance_id"]}: {",".join(excluded.get("reasons", []))}'
        text_box = draw.textbbox((x, y + height), label, font=font)
        draw.rectangle(text_box, fill=(190, 60, 60, 200))
        draw.text((x, y + height), label, fill=(255, 255, 255, 255), font=font)

    return image
