#!/usr/bin/env python3
"""Offline audit of an eyewear-localization result JSON.

Pure standard library: no torch, no EasyOCR, no model downloads. Point it at
any ``*.json`` produced by ``infer.py`` (or paste one via stdin) and it prints
exactly why each instance was attributed or abstained.

Usage:
  python diagnose.py outputs/image.json
  python diagnose.py < image.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt_box(box) -> str:
    if not box:
        return "-"
    return f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"


def diagnose(result: dict) -> list[str]:
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(line)

    backends = result.get("backends", {})
    emit("=" * 72)
    emit(f"IMAGE: {result.get('image')}")
    emit(
        "BACKENDS: localizer={name}{reason} | ocr={oname}{oreason}".format(
            name=backends.get("localizer", {}).get("name", "?"),
            reason=(
                f" ({backends['localizer']['reason']})"
                if backends.get("localizer", {}).get("reason")
                else ""
            ),
            oname=backends.get("ocr", {}).get("name", "?"),
            oreason=(
                f" ({backends['ocr']['reason']})"
                if backends.get("ocr", {}).get("reason")
                else ""
            ),
        )
    )
    instances = result.get("instances", [])
    excluded = result.get("excluded_instances", [])
    signs = result.get("signs", [])
    evidence = result.get("evidence", [])
    outputs = result.get("outputs", [])
    emit(
        f"COUNTS: instances={len(instances)} excluded={len(excluded)} "
        f"signs={len(signs)} evidence={len(evidence)} outputs={len(outputs)}"
    )
    emit("=" * 72)

    if excluded:
        emit("EXCLUDED BY SCENE FILTER (never reached attribution):")
        for item in excluded:
            emit(
                f"  {item['instance_id']} {_fmt_box(item['bbox'])} "
                f"reasons={item['reasons']}"
            )
        emit()

    if signs:
        emit("SIGNS (OCR + gazetteer + C2 scope):")
        for sign in signs:
            scope = sign.get("scope", {})
            emit(
                f"  {sign['sign_id']} {sign['text']!r} -> {sign['brand']} "
                f"ocr_conf={sign['confidence']:.2f} "
                f"scope={scope.get('type')} scope_conf={scope.get('confidence', 0):.2f} "
                f"region={_fmt_box(scope.get('region_bbox'))}"
            )
        emit()

    outputs_by_id = {item["instance_id"]: item for item in outputs}
    evidence_by_instance: dict[str, list] = {}
    for item in evidence:
        evidence_by_instance.setdefault(item["instance_id"], []).append(item)

    emit("PER-INSTANCE AUDIT:")
    if not instances:
        emit("  (no instances — the localizer found nothing or everything was filtered)")
    for instance in instances:
        instance_id = instance["id"]
        output = outputs_by_id.get(instance_id, {})
        brand = output.get("brand", "?")
        abstained = output.get("abstained", True)
        if not output:
            emit(f"  {instance_id} {_fmt_box(instance['bbox'])} -> no decision (not in outputs)")
            continue
        emit(
            f"  {instance_id} {_fmt_box(instance['bbox'])} "
            f"-> {brand}{' (ABSTAINED)' if abstained else ''}"
        )
        probs = output.get("probabilities", {})
        if probs:
            top = sorted(
                probs.items(), key=lambda item: item[1], reverse=True
            )[:3]
            emit("      probabilities: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
        debug = output.get("decision_debug")
        if debug:
            gates = debug.get("gates", {})
            emit(
                f"      gates: tau={gates.get('tau')} "
                f"margin={gates.get('margin')} "
                f"beats_unknown={gates.get('beats_unknown')} "
                f"(tau>={gates.get('thresholds', {}).get('tau')}, "
                f"margin>={gates.get('thresholds', {}).get('margin')})"
            )
        for item in evidence_by_instance.get(instance_id, []):
            support = item.get("support", {})
            scope = support.get("scope", {})
            emit(
                f"      evidence[{item['cue']}] {item['brand']} "
                f"conf={item['confidence']:.2f} "
                + (
                    f"(ocr={support.get('ocr_confidence', 0):.2f} "
                    f"match={support.get('match_method', '?')})"
                    if item["cue"] == "C1"
                    else f"(scope_conf={scope.get('confidence', 0):.2f} "
                    f"sign_ocr={support.get('sign_confidence') or '?'})"
                )
            )
        if not evidence_by_instance.get(instance_id):
            emit("      evidence: (none) — no cue reached this instance")
        emit()

    # Highlight the most common failure signatures.
    no_evidence = [
        item["id"] for item in instances if not evidence_by_instance.get(item["id"])
    ]
    weak_gate = [
        item["instance_id"]
        for item in outputs
        if item.get("decision_debug", {}).get("gates", {}).get("tau") is False
    ]
    if no_evidence:
        emit(f"SUMMARY: {len(no_evidence)} instance(s) had NO evidence: {no_evidence}")
        emit("  -> check signs[].scope above: C2 assigned no bay/row region to them.")
    if weak_gate:
        emit(f"SUMMARY: {len(weak_gate)} instance(s) failed the tau gate (evidence too weak): {weak_gate}")
        emit("  -> check evidence confidences; lower fusion.tau in config.yaml to test.")
    if excluded:
        emit(
            f"SUMMARY: {len(excluded)} instance(s) were removed by the scene filter; "
            "try --no-shelf-filter to compare."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args:
        with Path(args[0]).open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    else:
        result = json.load(sys.stdin)
    print("\n".join(diagnose(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
