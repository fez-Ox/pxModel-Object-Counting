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
        default_factory=lambda: {"C1": 0.95, "C2": 0.85, "C4": 0.35, "L3": 0.70}
    )
    temperature: float = 0.6
    unknown_prior: float = 0.35
    tau: float = 0.6
    margin: float = 0.15
    smoothing_lambda: float = 0.25
    smoothing_gate_punknown: float = 0.5
    use_vlm_audit: bool = True
    uncertainty_band: tuple[float, float] = (0.45, 0.70)

    def __post_init__(self) -> None:
        self.gazetteer = sorted(
            {normalize_text(str(item)) for item in self.gazetteer if normalize_text(str(item))}
        )
        for name, value in self.cue_reliability.items():
            self.cue_reliability[name] = _probability(value, name=f"cue_reliability.{name}")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.unknown_prior = _nonnegative(self.unknown_prior, "unknown_prior")
        self.tau = _probability(self.tau, "tau")
        self.margin = _probability(self.margin, "margin")
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
            },
            "smoothing": {
                "lambda": self.smoothing_lambda,
                "gate_punknown": self.smoothing_gate_punknown,
            },
            "cascade": {
                "use_vlm_audit": self.use_vlm_audit,
                "uncertainty_band": list(self.uncertainty_band),
            },
        }


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
    band = cascade.get("uncertainty_band", (0.45, 0.70))
    return LocalizationConfig(
        gazetteer=list(raw.get("gazetteer", [])),
        cue_reliability=dict(
            raw.get(
                "cue_reliability",
                {"C1": 0.95, "C2": 0.85, "C4": 0.35, "L3": 0.70},
            )
        ),
        temperature=float(fusion.get("temperature", 0.6)),
        unknown_prior=float(fusion.get("unknown_prior", 0.35)),
        tau=float(fusion.get("tau", 0.6)),
        margin=float(fusion.get("margin", 0.15)),
        smoothing_lambda=float(smoothing.get("lambda", 0.25)),
        smoothing_gate_punknown=float(smoothing.get("gate_punknown", 0.5)),
        use_vlm_audit=bool(cascade.get("use_vlm_audit", True)),
        uncertainty_band=(float(band[0]), float(band[1])),
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
