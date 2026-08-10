"""End-to-end orchestration for the staged attribution architecture."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.cues import (
    NullAuditor,
    OnProductBrandingCue,
    SignageScopeCue,
    StylePriorCue,
)
from eyewear_localization.fusion import decide, fuse_evidence, is_uncertain, smooth_continuity
from eyewear_localization.perception import PerceptionFrontend


class LocalizationPipeline:
    """Run L0 → C1/C2/C4 → C3 → fusion/decision → optional L3."""

    def __init__(
        self,
        frontend: PerceptionFrontend,
        *,
        config: LocalizationConfig | None = None,
        c1: OnProductBrandingCue | None = None,
        c2: SignageScopeCue | None = None,
        c4: StylePriorCue | None = None,
        auditor: Any | None = None,
    ) -> None:
        self.frontend = frontend
        self.config = config or LocalizationConfig(gazetteer=list(frontend.gazetteer.brands))
        self.c1 = c1 or OnProductBrandingCue(frontend.ocr, frontend.gazetteer)
        self.c2 = c2 or SignageScopeCue()
        self.c4 = c4 or StylePriorCue()
        self.auditor = auditor or NullAuditor()

    @staticmethod
    def _safe(callable_: Any, default: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return callable_(*args, **kwargs)
        except Exception:
            return default

    def run(
        self,
        image_path: str | Path,
        *,
        source: str | None = None,
        text_detections_override: list[Any] | None = None,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        # A selective OCR cascade shares one bounded stronger-model budget
        # across L0 signage and C1 product crops for this source image.
        reset_ocr_budget = getattr(self.frontend.ocr, "reset_budget", None)
        if callable(reset_ocr_budget):
            reset_ocr_budget()
        verbose = bool(getattr(self.frontend.localizer, "verbose", False))
        started = time.perf_counter()
        if verbose:
            print(f"[PIPE] start image={image_path.name}", flush=True)
        perception = self.frontend.run(
            image_path,
            text_detections_override=text_detections_override,
        )
        l0_elapsed = time.perf_counter() - started
        if verbose:
            print(
                f"[PIPE] L0 done instances={len(perception.instances)} "
                f"signs={len(perception.signs)} elapsed={l0_elapsed:.1f}s",
                flush=True,
            )

        # Get image dimensions for C2 column-boundary inference.
        image_width: float | None = None
        try:
            from PIL import Image

            with Image.open(image_path) as _img:
                image_width = float(_img.width)
        except Exception:
            pass

        release_localizer = getattr(self.frontend.localizer, "release", None)
        if callable(release_localizer):
            if verbose:
                print("[SAM3] releasing model before C1", flush=True)
            self._safe(release_localizer, None)

        c1_started = time.perf_counter()
        c1_evidence = self._safe(
            self.c1.emit, [], image_path, perception.instances
        )
        c1_elapsed = time.perf_counter() - c1_started
        if verbose:
            print(
                f"[PIPE] C1 done evidence={len(c1_evidence)} "
                f"elapsed={time.perf_counter() - c1_started:.1f}s",
                flush=True,
            )
        scoped_signs, c2_evidence = self._safe(
            lambda: self.c2.associate(
                perception.instances,
                perception.signs,
                perception.poster_regions,
                image_width=image_width,
                text_detections=perception.text_detections,
            ),
            (perception.signs, []),
        )
        c4_evidence = self._safe(
            self.c4.emit, [], image_path, perception.instances
        )
        if verbose:
            print(
                f"[PIPE] C2/C4 done c2={len(c2_evidence)} c4={len(c4_evidence)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        evidence = list(c1_evidence) + list(c2_evidence) + list(c4_evidence)

        probabilities = fuse_evidence(perception.instances, evidence, self.config)
        probabilities = smooth_continuity(
            probabilities,
            perception.instances,
            self.config,
        )
        outputs = decide(perception.instances, probabilities, evidence, self.config)

        # L3 is deliberately one optional pass.  A missing auditor is a
        # normal deployment mode, not an exception.
        if self.config.use_vlm_audit:
            uncertain = [output for output in outputs if is_uncertain(output, self.config)]
            extra = self._safe(
                self.auditor.emit,
                [],
                image_path,
                perception.instances,
                uncertain,
            )
            if extra:
                evidence.extend(extra)
                probabilities = fuse_evidence(perception.instances, evidence, self.config)
                probabilities = smooth_continuity(
                    probabilities,
                    perception.instances,
                    self.config,
                )
                outputs = decide(perception.instances, probabilities, evidence, self.config)

        def backend_info(backend: Any) -> dict[str, Any]:
            info = {
                "name": backend.name,
                "reliability": backend.reliability,
            }
            reason = getattr(backend, "reason", None)
            if reason:
                info["reason"] = str(reason)
            return info

        total_elapsed = time.perf_counter() - started
        c2_c4_fusion_elapsed = total_elapsed - l0_elapsed - c1_elapsed
        timings = {
            "l0_perception_seconds": round(l0_elapsed, 3),
            "c1_onproduct_seconds": round(c1_elapsed, 3),
            "c2_c4_fusion_seconds": round(max(0.0, c2_c4_fusion_elapsed), 3),
            "brand_association_total_seconds": round(c1_elapsed + max(0.0, c2_c4_fusion_elapsed), 3),
            "total_pipeline_seconds": round(total_elapsed, 3),
        }
        if verbose:
            print(
                f"[PIPE] finished image={image_path.name} "
                f"brand_association={timings['brand_association_total_seconds']}s "
                f"total={timings['total_pipeline_seconds']}s",
                flush=True,
            )

        result: dict[str, Any] = {
            "schema_version": "1.0",
            "image": source or str(image_path),
            "instances": [item.to_dict() for item in perception.instances],
            "excluded_instances": [item.to_dict() for item in perception.excluded_instances],
            "signs": [item.to_dict() for item in scoped_signs],
            "poster_regions": [item.to_dict() for item in perception.poster_regions],
            "text_detections": [item.to_dict() for item in perception.text_detections],
            "evidence": [item.to_dict() for item in evidence],
            "outputs": [item.to_dict() for item in outputs],
            "backends": {
                "localizer": backend_info(self.frontend.localizer),
                "ocr": backend_info(self.frontend.ocr),
                "poster_detector": backend_info(self.frontend.poster_detector),
            },
            "timings": timings,
            "effective_config": self.config.to_dict(),
        }
        return result
