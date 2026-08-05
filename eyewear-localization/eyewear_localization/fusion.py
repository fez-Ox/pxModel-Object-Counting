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
    known_brands = set(config.gazetteer)
    known_brands.update(item.brand for item in evidence)
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    valid_ids = {instance.id for instance in instances}
    for item in evidence:
        if item.instance_id in valid_ids:
            grouped[item.instance_id].append(item)

    probabilities: dict[str, dict[str, float]] = {}
    for instance in instances:
        scores = {brand: 0.0 for brand in sorted(known_brands)}
        scores["unknown"] = config.unknown_prior
        for item in grouped[instance.id]:
            scores[item.brand] = scores.get(item.brand, 0.0) + (
                config.cue_reliability.get(item.cue, 0.0) * item.confidence
            )
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
    for instance in instances:
        values = dict(probabilities.get(instance.id, {"unknown": 1.0}))
        values.setdefault("unknown", 1.0)
        candidates = sorted(
            ((brand, probability) for brand, probability in values.items() if brand != "unknown"),
            key=lambda item: item[1],
            reverse=True,
        )
        if not candidates:
            outputs.append(AttributionOutput(instance.id, "unknown", True, values, grouped[instance.id]))
            continue
        best_brand, best_probability = candidates[0]
        runner_up = max(
            [values.get("unknown", 0.0)]
            + [probability for _, probability in candidates[1:]]
        )
        accepted = (
            best_probability >= config.tau
            and best_probability - runner_up >= config.margin
            and best_probability > values.get("unknown", 0.0)
        )
        outputs.append(
            AttributionOutput(
                instance.id,
                best_brand if accepted else "unknown",
                not accepted,
                values,
                grouped[instance.id],
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
