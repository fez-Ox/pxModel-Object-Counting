"""Configuration for the attribution pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from eyewear_localization.gazetteer import normalize_text


@dataclass
class LocalizationConfig:
    gazetteer: list[str] = field(default_factory=list)
    cue_reliability: dict[str, float] = field(
        default_factory=lambda: {"C1": 0.95, "C2": 0.30, "C4": 0.35, "L3": 0.70}
    )
    temperature: float = 0.6
    unknown_prior: float = 0.35
    tau: float = 0.6
    margin: float = 0.15
    max_per_evidence: float | None = None
    smoothing_lambda: float = 0.25
    smoothing_gate_punknown: float = 0.5
    use_vlm_audit: bool = True
    uncertainty_band: tuple[float, float] = (0.45, 0.70)
    cascade_t1: float = 0.70
    cascade_t2: float = 0.75
    cascade_t4: float = 0.85
    c1_margin: float = 0.25
    c1_wide_margin: float = 0.40
    c1_scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    c1_use_clahe: bool = True
    c1_dual_polarity: bool = True
    ocr_scale: float = 2.0
    ocr_fallback_budget: int = 12
    florence2_model_id: str = "microsoft/Florence-2-large"
    # Batched deployment defaults retain per-chunk sequential fallback when a
    # backend cannot safely batch, but use the approved larger micro-batches.
    c1_batch_size: int = 4
    sam3_prompt_batch_size: int = 4
    sam3_compile: bool = False

    def __post_init__(self) -> None:
        self.gazetteer = sorted(
            {normalize_text(str(item)) for item in self.gazetteer if normalize_text(str(item))}
        )
        self.c1_batch_size = _positive_int(self.c1_batch_size, "c1_batch_size")
        self.sam3_prompt_batch_size = _positive_int(
            self.sam3_prompt_batch_size, "sam3_prompt_batch_size"
        )
        self.sam3_compile = bool(self.sam3_compile)
        self.cascade_t1 = _probability(self.cascade_t1, "cascade.c1_threshold")
        self.cascade_t2 = _probability(self.cascade_t2, "cascade.c2_threshold")
        self.cascade_t4 = _probability(self.cascade_t4, "cascade.c4_threshold")
        self.c1_margin = _nonnegative(self.c1_margin, "c1.margin")
        self.c1_wide_margin = _nonnegative(self.c1_wide_margin, "c1.wide_margin")
        if self.c1_wide_margin < self.c1_margin:
            raise ValueError("c1.wide_margin must be at least c1.margin")
        self.c1_scales = tuple(sorted({float(scale) for scale in self.c1_scales}))
        if not self.c1_scales or any(scale <= 0 for scale in self.c1_scales):
            raise ValueError("c1.scales must contain positive values")
        self.ocr_scale = float(self.ocr_scale)
        if self.ocr_scale <= 0:
            raise ValueError("ocr.scale must be greater than zero")
        self.ocr_fallback_budget = _positive_int(
            self.ocr_fallback_budget, "ocr.fallback_budget"
        )
        for name, value in self.cue_reliability.items():
            self.cue_reliability[name] = _probability(value, name=f"cue_reliability.{name}")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.unknown_prior = _nonnegative(self.unknown_prior, "unknown_prior")
        self.tau = _probability(self.tau, "tau")
        self.margin = _probability(self.margin, "margin")
        if self.max_per_evidence is not None:
            self.max_per_evidence = _nonnegative(self.max_per_evidence, "max_per_evidence")
        self.smoothing_lambda = _probability(self.smoothing_lambda, "smoothing_lambda")
        self.smoothing_gate_punknown = _probability(
            self.smoothing_gate_punknown, "smoothing_gate_punknown"
        )
        if len(self.uncertainty_band) != 2:
            raise ValueError("uncertainty_band must contain two values")
        low, high = (float(value) for value in self.uncertainty_band)
        if not 0 <= low <= high <= 1:
            raise ValueError("uncertainty_band must be ordered probabilities")
        self.uncertainty_band = (low, high)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gazetteer": list(self.gazetteer),
            "cue_reliability": dict(self.cue_reliability),
            "fusion": {
                "temperature": self.temperature,
                "unknown_prior": self.unknown_prior,
                "tau": self.tau,
                "margin": self.margin,
                "max_per_evidence": self.max_per_evidence,
            },
            "smoothing": {
                "lambda": self.smoothing_lambda,
                "gate_punknown": self.smoothing_gate_punknown,
            },
            "cascade": {
                "use_vlm_audit": self.use_vlm_audit,
                "uncertainty_band": list(self.uncertainty_band),
                "c1_threshold": self.cascade_t1,
                "c2_threshold": self.cascade_t2,
                "c4_threshold": self.cascade_t4,
            },
            "c1": {
                "margin": self.c1_margin,
                "wide_margin": self.c1_wide_margin,
                "scales": list(self.c1_scales),
                "use_clahe": self.c1_use_clahe,
                "dual_polarity": self.c1_dual_polarity,
            },
            "ocr": {
                "scale": self.ocr_scale,
                "fallback_budget": self.ocr_fallback_budget,
            },
            "performance": {
                "c1_batch_size": self.c1_batch_size,
                "sam3_prompt_batch_size": self.sam3_prompt_batch_size,
                "sam3_compile": self.sam3_compile,
            },
        }

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "LocalizationConfig":
        """Return a copy with nested overrides applied on top of this config.

        ``overrides`` uses the same nested shape as the YAML file, e.g.
        ``{"fusion": {"tau": 0.5}, "cue_reliability": {"C2": 0.8}}``.
        Values are deep-merged: unspecified keys keep their current value.
        """
        if not overrides:
            return self
        merged = _deep_merge(self.to_dict(), dict(overrides))
        return config_from_mapping(merged)


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def config_from_mapping(raw: Mapping[str, Any]) -> LocalizationConfig:
    fusion = _mapping(raw.get("fusion"), "fusion")
    smoothing = _mapping(raw.get("smoothing"), "smoothing")
    cascade = _mapping(raw.get("cascade"), "cascade")
    performance = _mapping(raw.get("performance"), "performance")
    c1 = _mapping(raw.get("c1"), "c1")
    ocr = _mapping(raw.get("ocr"), "ocr")
    band = cascade.get("uncertainty_band", (0.45, 0.70))
    return LocalizationConfig(
        gazetteer=list(raw.get("gazetteer", [])),
        cue_reliability=dict(
            raw.get(
                "cue_reliability",
                {"C1": 0.95, "C2": 0.30, "C4": 0.35, "L3": 0.70},
            )
        ),
        temperature=float(fusion.get("temperature", 0.6)),
        unknown_prior=float(fusion.get("unknown_prior", 0.35)),
        tau=float(fusion.get("tau", 0.6)),
        margin=float(fusion.get("margin", 0.15)),
        max_per_evidence=None if fusion.get("max_per_evidence") in (None, "none") else float(fusion.get("max_per_evidence")),
        smoothing_lambda=float(smoothing.get("lambda", 0.25)),
        smoothing_gate_punknown=float(smoothing.get("gate_punknown", 0.5)),
        use_vlm_audit=bool(cascade.get("use_vlm_audit", True)),
        uncertainty_band=(float(band[0]), float(band[1])),
        cascade_t1=float(cascade.get("c1_threshold", 0.70)),
        cascade_t2=float(cascade.get("c2_threshold", 0.75)),
        cascade_t4=float(cascade.get("c4_threshold", 0.85)),
        c1_margin=float(c1.get("margin", 0.25)),
        c1_wide_margin=float(c1.get("wide_margin", 0.40)),
        c1_scales=tuple(float(scale) for scale in c1.get("scales", (1.0, 2.0, 4.0))),
        c1_use_clahe=bool(c1.get("use_clahe", True)),
        c1_dual_polarity=bool(c1.get("dual_polarity", True)),
        ocr_scale=float(ocr.get("scale", 2.0)),
        ocr_fallback_budget=_positive_int(ocr.get("fallback_budget", 12), "ocr.fallback_budget"),
        c1_batch_size=_positive_int(performance.get("c1_batch_size", 4), "performance.c1_batch_size"),
        sam3_prompt_batch_size=_positive_int(
            performance.get("sam3_prompt_batch_size", 4),
            "performance.sam3_prompt_batch_size",
        ),
        sam3_compile=bool(performance.get("sam3_compile", False)),
    )


def load_config(path: str | Path | None = None) -> LocalizationConfig:
    """Load YAML/JSON config, or return specification defaults."""
    if path is None:
        return LocalizationConfig()
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        import yaml

        raw = yaml.safe_load(handle) or {}
    return config_from_mapping(raw)
