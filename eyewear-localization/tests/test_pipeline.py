from pathlib import Path
import tempfile
import unittest

from PIL import Image

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.gazetteer import Gazetteer
from eyewear_localization.perception import (
    LocalizationDetection,
    NullPosterBackend,
    PerceptionFrontend,
    SAM3Localizer,
)
from eyewear_localization.pipeline import LocalizationPipeline
from eyewear_localization.schemas import TextDetection


class FakeOCR:
    name = "fake-ocr"
    reliability = 1.0

    def detect(self, image):
        return [TextDetection("Cartier", [2, 2, 30, 8], 0.95, source=self.name)]


class NoopLocalizer:
    name = "noop-localizer"
    reliability = 0.1

    def detect(self, image_path):
        return []


class SignOCR:
    name = "sign-ocr"
    reliability = 1.0

    def detect(self, image):
        return [
            TextDetection("HSTN Oakley Mahomes Oakley Mbappé", [0, 0, 100, 20], 0.95, source=self.name),
            TextDetection("MICHAEL KORS", [10, 10, 50, 10], 0.90, source=self.name),
            TextDetection("KORS MICHAEL", [11, 11, 49, 9], 0.92, source=self.name),
            TextDetection("MICHAEL KORS", [200, 10, 50, 10], 0.90, source=self.name),
        ]


class PipelineTests(unittest.TestCase):
    def test_cached_ocr_bypasses_live_ocr_backend(self):
        class RaisingOCR:
            name = "raising-ocr"
            reliability = 1.0

            def detect(self, image):
                raise AssertionError("live OCR must not run when cache is supplied")

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            frontend = PerceptionFrontend(
                localizer=NoopLocalizer(),
                ocr=RaisingOCR(),
                gazetteer=Gazetteer(["cartier"]),
                poster_detector=NullPosterBackend(),
            )
            result = frontend.run(
                image_path,
                text_detections_override=[
                    TextDetection("Cartier", [2, 2, 30, 8], 0.95, source="ocr-cache")
                ],
            )

        self.assertEqual([sign.brand for sign in result.signs], ["cartier"])

    def test_sign_aggregation_rejects_broad_lines_and_keeps_separate_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (300, 100), "white").save(image_path)
            frontend = PerceptionFrontend(
                localizer=NoopLocalizer(),
                ocr=SignOCR(),
                gazetteer=Gazetteer(["oakley", "michael kors"]),
                poster_detector=NullPosterBackend(),
            )
            result = frontend.run(image_path)

        self.assertEqual([sign.brand for sign in result.signs], ["michael kors", "michael kors"])
        self.assertEqual(len(result.text_detections), 4)

    def test_ocr_evidence_is_fused_but_instance_scope_is_not_assumed(self):
        def predictor(path: Path, prompt: str):
            if "sunglasses" in prompt:
                return {"boxes": [[20, 45, 60, 75]], "scores": [0.9]}
            return {"boxes": [], "scores": []}

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            frontend = PerceptionFrontend(
                localizer=SAM3Localizer(predictor),
                ocr=FakeOCR(),
                gazetteer=Gazetteer(["cartier"]),
                poster_detector=NullPosterBackend(),
            )
            config = LocalizationConfig(gazetteer=["cartier"], use_vlm_audit=False)
            result = LocalizationPipeline(frontend, config=config).run(image_path)

        self.assertEqual(len(result["instances"]), 1)
        self.assertTrue(result["evidence"])
        self.assertIn(result["outputs"][0]["brand"], {"cartier", "unknown"})
        self.assertEqual(result["backends"]["localizer"]["name"], "sam3-class-agnostic")
