"""End-to-end orchestration for the staged attribution architecture."""

from __future__ import annotations

from pathlib import Path
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

    def run(self, image_path: str | Path, *, source: str | None = None) -> dict[str, Any]:
        image_path = Path(image_path)
        perception = self.frontend.run(image_path)

        c1_evidence = self._safe(
            self.c1.emit, [], image_path, perception.instances
        )
        scoped_signs, c2_evidence = self._safe(
            self.c2.associate,
            (perception.signs, []),
            perception.instances,
            perception.signs,
            perception.poster_regions,
        )
        c4_evidence = self._safe(
            self.c4.emit, [], image_path, perception.instances
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
        }
        return result
