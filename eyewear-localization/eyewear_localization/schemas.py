"""JSON-compatible data contracts for every localization stage.

Coordinates use ``[x, y, width, height]`` throughout the public schemas.  The
native SAM3 adapter may use XYXY internally, but conversion happens at the L0
boundary so modules do not silently disagree about geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping


def _numbers(values: Any, *, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{name} must contain exactly four numbers")
    result = [float(value) for value in values]
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _point(values: Any, *, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    result = [float(value) for value in values]
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _confidence(value: float, *, name: str = "confidence") -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0 or not isfinite(result):
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _bbox_xywh(value: Any, *, name: str = "bbox") -> list[float]:
    result = _numbers(value, name=name)
    if result[2] < 0 or result[3] < 0:
        raise ValueError(f"{name} width and height must be non-negative")
    return result


def xyxy_to_xywh(box: Any) -> list[float]:
    values = _numbers(box, name="XYXY box")
    x0, y0, x1, y1 = values
    if x1 < x0 or y1 < y0:
        raise ValueError("XYXY box must have non-decreasing corners")
    return [x0, y0, x1 - x0, y1 - y0]


def xywh_to_xyxy(box: Any) -> list[float]:
    x, y, width, height = _bbox_xywh(box)
    return [x, y, x + width, y + height]


@dataclass
class Instance:
    """An eyewear localization emitted by L0."""

    id: str
    bbox: list[float]
    centroid: list[float] | None = None
    mask_rle: str | None = None
    lens_tint_score: float | None = None
    localization_score: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("instance id must not be empty")
        self.bbox = _bbox_xywh(self.bbox)
        if self.centroid is None:
            self.centroid = [
                self.bbox[0] + self.bbox[2] / 2.0,
                self.bbox[1] + self.bbox[3] / 2.0,
            ]
        else:
            self.centroid = _point(self.centroid, name="centroid")
        if self.lens_tint_score is not None:
            self.lens_tint_score = _confidence(self.lens_tint_score, name="lens_tint_score")
        if self.localization_score is not None:
            self.localization_score = _confidence(
                self.localization_score, name="localization_score"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid or []),
            "mask_rle": self.mask_rle,
            "lens_tint_score": self.lens_tint_score,
            "localization_score": self.localization_score,
        }


@dataclass
class TextDetection:
    """A raw OCR detection before gazetteer matching or scope attribution."""

    text: str
    bbox: list[float]
    confidence: float
    source: str = "ocr"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("OCR text must not be empty")
        self.bbox = _bbox_xywh(self.bbox)
        self.confidence = _confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "source": self.source,
        }


_SCOPE_TYPES = {"bay_header", "row_label", "shelf_edge_tag", "hanging_sign", "none"}


@dataclass
class Scope:
    """The region that a sign is hypothesized to label."""

    type: str = "none"
    region_bbox: list[float] | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.type not in _SCOPE_TYPES:
            raise ValueError(f"unsupported scope type: {self.type!r}")
        if self.region_bbox is not None:
            self.region_bbox = _bbox_xywh(self.region_bbox, name="scope.region_bbox")
        self.confidence = _confidence(self.confidence, name="scope.confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "region_bbox": list(self.region_bbox) if self.region_bbox is not None else None,
            "confidence": self.confidence,
        }


@dataclass
class Sign:
    """A gazetteer-matched or candidate brand sign detected by L0 OCR."""

    sign_id: str
    text: str
    brand: str | None
    bbox: list[float]
    scope: Scope = field(default_factory=Scope)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.sign_id:
            raise ValueError("sign_id must not be empty")
        if not self.text.strip():
            raise ValueError("sign text must not be empty")
        self.brand = self.brand.strip().lower() if self.brand else None
        self.bbox = _bbox_xywh(self.bbox)
        self.confidence = _confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sign_id": self.sign_id,
            "text": self.text,
            "brand": self.brand,
            "bbox": list(self.bbox),
            "scope": self.scope.to_dict(),
            "confidence": self.confidence,
        }


@dataclass
class PosterRegion:
    """A region likely to be an advertisement/poster rather than signage."""

    bbox: list[float]
    confidence: float
    source: str = "poster_detector"

    def __post_init__(self) -> None:
        self.bbox = _bbox_xywh(self.bbox)
        self.confidence = _confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class Evidence:
    """A cue-level belief; it is not a final brand decision."""

    instance_id: str
    brand: str
    confidence: float
    cue: str
    support: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("evidence instance_id must not be empty")
        self.brand = self.brand.strip().lower()
        if not self.brand or self.brand == "unknown":
            raise ValueError("evidence must name a concrete brand")
        self.confidence = _confidence(self.confidence)
        if not self.cue:
            raise ValueError("evidence cue must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "brand": self.brand,
            "confidence": self.confidence,
            "cue": self.cue,
            "support": self.support,
        }


@dataclass
class AttributionOutput:
    """A final post-fusion decision for one instance."""

    instance_id: str
    brand: str
    abstained: bool
    probabilities: dict[str, float]
    evidence: list[Evidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("output instance_id must not be empty")
        self.brand = self.brand.strip().lower()
        if not self.brand:
            raise ValueError("output brand must not be empty")
        self.probabilities = {
            str(key).lower(): _confidence(value, name=f"probabilities[{key}]")
            for key, value in self.probabilities.items()
        }
        if "unknown" not in self.probabilities:
            raise ValueError("output probabilities must include unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "brand": self.brand,
            "abstained": bool(self.abstained),
            "probabilities": dict(self.probabilities),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass
class InstanceExclusion:
    """An L0 instance removed before attribution, retained for auditability."""

    instance_id: str
    bbox: list[float]
    reasons: list[str]
    support: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("excluded instance_id must not be empty")
        self.bbox = _bbox_xywh(self.bbox)
        self.reasons = [str(reason) for reason in self.reasons if str(reason)]
        if not self.reasons:
            raise ValueError("an exclusion must have at least one reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "bbox": list(self.bbox),
            "reasons": list(self.reasons),
            "support": self.support,
        }


@dataclass
class PerceptionResult:
    """The complete JSON boundary emitted by L0."""

    instances: list[Instance]
    signs: list[Sign]
    poster_regions: list[PosterRegion]
    text_detections: list[TextDetection] = field(default_factory=list)
    excluded_instances: list[InstanceExclusion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instances": [item.to_dict() for item in self.instances],
            "excluded_instances": [item.to_dict() for item in self.excluded_instances],
            "signs": [item.to_dict() for item in self.signs],
            "poster_regions": [item.to_dict() for item in self.poster_regions],
            "text_detections": [item.to_dict() for item in self.text_detections],
        }


def evidence_from_dict(value: Mapping[str, Any]) -> Evidence:
    """Parse a schema-level evidence mapping, useful for model adapters."""
    return Evidence(
        instance_id=str(value["instance_id"]),
        brand=str(value["brand"]),
        confidence=float(value["confidence"]),
        cue=str(value["cue"]),
        support=dict(value.get("support", {})),
    )
