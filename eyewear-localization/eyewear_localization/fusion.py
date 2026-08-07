"""Belief fusion, abstention, and gated planogram continuity smoothing."""

from __future__ import annotations

from collections import defaultdict
from math import exp
from statistics import median
from typing import Callable, Iterable

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.schemas import AttributionOutput, Evidence, Instance


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    scaled = {key: value / temperature for key, value in scores.items()}
    maximum = max(scaled.values()) if scaled else 0.0
    exponentials = {key: exp(value - maximum) for key, value in scaled.items()}
    total = sum(exponentials.values())
    if total <= 0:
        return {"unknown": 1.0}
    return {key: value / total for key, value in exponentials.items()}


def fuse_evidence(
    instances: Iterable[Instance],
    evidence: list[Evidence],
    config: LocalizationConfig,
) -> dict[str, dict[str, float]]:
    """Apply the default reliability-weighted fusion equation."""
    instances = list(instances)
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    valid_ids = {instance.id for instance in instances}
    for item in evidence:
        if item.instance_id in valid_ids:
            grouped[item.instance_id].append(item)

    probabilities: dict[str, dict[str, float]] = {}
    for instance in instances:
        # The gazetteer is a matching vocabulary, not a set of active
        # hypotheses for every object. Including every zero-evidence brand
        # here would dilute a valid cue as the catalog grows. Each instance
        # competes only among brands supported by its own evidence and the
        # explicit unknown alternative.
        scores: dict[str, float] = {"unknown": config.unknown_prior}
        for item in grouped[instance.id]:
            contribution = config.cue_reliability.get(item.cue, 0.0) * item.confidence
            if config.max_per_evidence is not None:
                contribution = min(contribution, config.max_per_evidence)
            scores[item.brand] = scores.get(item.brand, 0.0) + contribution
        probabilities[instance.id] = _softmax(scores, config.temperature)
    return probabilities


def _same_shelf_neighbors(
    instances: list[Instance],
    *,
    appearance_similarity: Callable[[Instance, Instance], float] | None = None,
    similarity_threshold: float = 0.0,
) -> dict[str, list[tuple[str, float]]]:
    if not instances:
        return {}
    median_height = max(1.0, median(instance.bbox[3] for instance in instances))
    median_width = max(1.0, median(instance.bbox[2] for instance in instances))
    neighbors: dict[str, list[tuple[str, float]]] = {instance.id: [] for instance in instances}
    for left_index, left in enumerate(instances):
        for right in instances[left_index + 1 :]:
            left_center = left.centroid or [left.bbox[0] + left.bbox[2] / 2, left.bbox[1] + left.bbox[3] / 2]
            right_center = right.centroid or [right.bbox[0] + right.bbox[2] / 2, right.bbox[1] + right.bbox[3] / 2]
            y_gap = abs(left_center[1] - right_center[1])
            x_gap = abs(left_center[0] - right_center[0])
            same_shelf = y_gap <= max(1.5 * median_height, 0.25 * max(left.bbox[3], right.bbox[3]))
            nearby = x_gap <= 6.0 * median_width
            if not (same_shelf and nearby):
                continue
            similarity = 1.0 if appearance_similarity is None else float(appearance_similarity(left, right))
            if similarity < similarity_threshold:
                continue
            weight = max(1e-6, similarity) / (1.0 + x_gap / median_width)
            neighbors[left.id].append((right.id, weight))
            neighbors[right.id].append((left.id, weight))
    return neighbors


def smooth_continuity(
    probabilities: dict[str, dict[str, float]],
    instances: list[Instance],
    config: LocalizationConfig,
    *,
    appearance_similarity: Callable[[Instance, Instance], float] | None = None,
    similarity_threshold: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Apply C3 only through neighbors that are not themselves uncertain."""
    if not probabilities or config.smoothing_lambda == 0:
        return {key: dict(value) for key, value in probabilities.items()}
    neighbors = _same_shelf_neighbors(
        instances,
        appearance_similarity=appearance_similarity,
        similarity_threshold=similarity_threshold,
    )
    all_brands = {brand for value in probabilities.values() for brand in value}
    result: dict[str, dict[str, float]] = {}
    for instance in instances:
        current = dict(probabilities.get(instance.id, {"unknown": 1.0}))
        eligible = [
            (neighbor_id, weight)
            for neighbor_id, weight in neighbors.get(instance.id, [])
            if probabilities.get(neighbor_id, {}).get("unknown", 1.0) < config.smoothing_gate_punknown
        ]
        if not eligible:
            result[instance.id] = current
            continue
        weight_total = sum(weight for _, weight in eligible)
        neighbor_average = {
            brand: sum(
                weight * probabilities[neighbor_id].get(brand, 0.0)
                for neighbor_id, weight in eligible
            )
            / weight_total
            for brand in all_brands
        }
        mixed = {
            brand: (1.0 - config.smoothing_lambda) * current.get(brand, 0.0)
            + config.smoothing_lambda * neighbor_average.get(brand, 0.0)
            for brand in all_brands
        }
        total = sum(mixed.values())
        result[instance.id] = (
            {brand: value / total for brand, value in mixed.items()}
            if total > 0
            else {"unknown": 1.0}
        )
    return result


def decide(
    instances: Iterable[Instance],
    probabilities: dict[str, dict[str, float]],
    evidence: list[Evidence],
    config: LocalizationConfig,
) -> list[AttributionOutput]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        grouped[item.instance_id].append(item)
    outputs: list[AttributionOutput] = []

    t1 = getattr(config, "cascade_t1", 0.80)
    t2 = getattr(config, "cascade_t2", 0.70)
    t4 = getattr(config, "cascade_t4", 0.85)

    for instance in instances:
        values = dict(probabilities.get(instance.id, {"unknown": 1.0}))
        values.setdefault("unknown", 1.0)
        inst_evidence = grouped[instance.id]

        c1_ev = next((ev for ev in inst_evidence if ev.cue == "C1" and ev.confidence >= t1), None)
        c2_ev = next((ev for ev in inst_evidence if ev.cue == "C2" and ev.confidence >= t2), None)
        c4_ev = next((ev for ev in inst_evidence if ev.cue == "C4" and ev.confidence >= t4), None)

        product_brand = c1_ev.brand if c1_ev else (c4_ev.brand if c4_ev else None)
        zone_brand = c2_ev.brand if c2_ev else None

        # Precision cascade decision list
        if c1_ev:
            outputs.append(
                AttributionOutput(
                    instance_id=instance.id,
                    brand=c1_ev.brand,
                    abstained=False,
                    probabilities=values,
                    evidence=inst_evidence,
                    decision_debug={
                        "gate": "precision_cascade",
                        "path": "C1",
                        "c1_confidence": c1_ev.confidence,
                        "gates": {"tau": True, "margin": True, "beats_unknown": True},
                    },
                    product_brand=c1_ev.brand,
                    zone_brand=zone_brand,
                    decision_path="C1",
                )
            )
            continue

        if c2_ev:
            outputs.append(
                AttributionOutput(
                    instance_id=instance.id,
                    brand=c2_ev.brand,
                    abstained=False,
                    probabilities=values,
                    evidence=inst_evidence,
                    decision_debug={
                        "gate": "precision_cascade",
                        "path": "C2",
                        "c2_confidence": c2_ev.confidence,
                        "gates": {"tau": True, "margin": True, "beats_unknown": True},
                    },
                    product_brand=product_brand,
                    zone_brand=c2_ev.brand,
                    decision_path="C2",
                )
            )
            continue

        if c4_ev:
            outputs.append(
                AttributionOutput(
                    instance_id=instance.id,
                    brand=c4_ev.brand,
                    abstained=False,
                    probabilities=values,
                    evidence=inst_evidence,
                    decision_debug={
                        "gate": "precision_cascade",
                        "path": "C4",
                        "c4_confidence": c4_ev.confidence,
                        "gates": {"tau": True, "margin": True, "beats_unknown": True},
                    },
                    product_brand=c4_ev.brand,
                    zone_brand=zone_brand,
                    decision_path="C4",
                )
            )
            continue

        candidates = sorted(
            ((brand, probability) for brand, probability in values.items() if brand != "unknown"),
            key=lambda item: item[1],
            reverse=True,
        )
        if not candidates:
            outputs.append(
                AttributionOutput(
                    instance_id=instance.id,
                    brand="unknown",
                    abstained=True,
                    probabilities=values,
                    evidence=inst_evidence,
                    decision_debug={"reason": "no_brand_candidates"},
                    product_brand=None,
                    zone_brand=None,
                    decision_path="none",
                )
            )
            continue

        best_brand, best_probability = candidates[0]
        runner_up = max(
            [values.get("unknown", 0.0)]
            + [probability for _, probability in candidates[1:]]
        )
        runner_label = (
            "unknown"
            if runner_up == values.get("unknown", 0.0)
            else next(
                (brand for brand, probability in candidates[1:] if probability == runner_up),
                "unknown",
            )
        )
        gate_tau = best_probability >= config.tau
        gate_margin = best_probability - runner_up >= config.margin
        gate_unknown = best_probability > values.get("unknown", 0.0)
        accepted = gate_tau and gate_margin and gate_unknown
        decision_debug = {
            "best_brand": best_brand,
            "best_probability": best_probability,
            "runner_up_label": runner_label,
            "runner_up_probability": runner_up,
            "gates": {
                "tau": gate_tau,
                "margin": gate_margin,
                "beats_unknown": gate_unknown,
                "thresholds": {"tau": config.tau, "margin": config.margin},
            },
        }
        outputs.append(
            AttributionOutput(
                instance_id=instance.id,
                brand=best_brand if accepted else "unknown",
                abstained=not accepted,
                probabilities=values,
                evidence=inst_evidence,
                decision_debug=decision_debug,
                product_brand=product_brand,
                zone_brand=zone_brand,
                decision_path="softmax" if accepted else "none",
            )
        )
    return outputs


def is_uncertain(output: AttributionOutput, config: LocalizationConfig) -> bool:
    non_unknown = [
        probability for brand, probability in output.probabilities.items() if brand != "unknown"
    ]
    top = max(non_unknown, default=0.0)
    low, high = config.uncertainty_band
    values = sorted(non_unknown, reverse=True)
    conflict = len(values) >= 2 and values[0] - values[1] < config.margin
    return (low <= top <= high) or conflict
