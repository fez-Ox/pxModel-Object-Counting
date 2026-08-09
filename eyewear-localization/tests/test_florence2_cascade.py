from pathlib import Path
import tempfile
import unittest

from PIL import Image

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.cues import OnProductBrandingCue
from eyewear_localization.fusion import decide
from eyewear_localization.gazetteer import Gazetteer
from eyewear_localization.perception import (
    Florence2OCRBackend,
    SelectiveOCRBackend,
    build_ocr_backend,
)
from eyewear_localization.schemas import Evidence, Instance, TextDetection


class StubOCR:
    name = "stub"
    reliability = 1.0

    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return list(self.detections)

    def detect_preprocessed(self, image):
        self.calls += 1
        return list(self.detections)


class C1CascadeStub:
    name = "c1-cascade-stub"
    reliability = 1.0

    def __init__(self):
        self.primary_calls = 0
        self.fallback_calls = 0

    def detect_primary_preprocessed(self, image):
        self.primary_calls += 1
        return [TextDetection("unrelated", [1, 1, 10, 5], 0.9, source=self.name)]

    def detect_fallback_preprocessed(self, image):
        self.fallback_calls += 1
        return [TextDetection("Cartier", [1, 1, 10, 5], 0.9, source=self.name)]


class Florence2AndCascadeTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalizationConfig(gazetteer=["cartier", "gucci", "ray-ban"], use_vlm_audit=False)
        self.instance = Instance("inst_0001", [10, 10, 30, 20])

    def test_build_florence2_ocr_backend(self):
        backend = build_ocr_backend("florence2")
        self.assertEqual(backend.name, "florence2")

    def test_selective_ocr_calls_stronger_backend_only_when_primary_is_unmatched(self):
        primary = StubOCR([TextDetection("unrelated", [1, 1, 10, 5], 0.9, source="primary")])
        fallback = StubOCR([TextDetection("Cartier", [2, 2, 12, 5], 0.95, source="fallback")])
        backend = SelectiveOCRBackend(primary, fallback, Gazetteer(["cartier"]), max_fallback_calls=1)

        output = backend.detect(object())
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual({item.text for item in output}, {"unrelated", "Cartier"})

        # The budget prevents per-instance OCR from expanding without bound.
        backend.detect_preprocessed(object())
        self.assertEqual(fallback.calls, 1)
        backend.reset_budget()
        backend.detect_preprocessed(object())
        self.assertEqual(fallback.calls, 2)

    def test_c1_uses_fallback_once_after_all_primary_scales_are_unmatched(self):
        ocr = C1CascadeStub()
        cue = OnProductBrandingCue(
            ocr,
            Gazetteer(["cartier"]),
            margin=0.0,
            scales=(1.0, 2.0),
            sharpen=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            evidence = cue.emit(image_path, [Instance("inst_0001", [10, 10, 30, 20])])

        self.assertEqual(ocr.primary_calls, 2)
        self.assertEqual(ocr.fallback_calls, 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].brand, "cartier")

    def test_selective_ocr_runs_scene_fallback_even_when_primary_finds_one_brand(self):
        primary = StubOCR([TextDetection("Cartier", [1, 1, 10, 5], 0.9, source="primary")])
        fallback = StubOCR([TextDetection("Gucci", [2, 2, 12, 5], 0.95, source="fallback")])
        backend = SelectiveOCRBackend(primary, fallback, Gazetteer(["cartier", "gucci"]), max_fallback_calls=1)

        output = backend.detect(object())
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual({item.text for item in output}, {"Cartier", "Gucci"})

    def test_precision_cascade_c1_overrides_c2(self):
        evidence = [
            Evidence("inst_0001", "ray-ban", 0.95, "C1"),
            Evidence("inst_0001", "gucci", 0.90, "C2"),
        ]
        probabilities = {"inst_0001": {"ray-ban": 0.5, "gucci": 0.4, "unknown": 0.1}}
        output = decide([self.instance], probabilities, evidence, self.config)[0]

        self.assertEqual(output.final_brand, "ray-ban")
        self.assertEqual(output.product_brand, "ray-ban")
        self.assertEqual(output.zone_brand, "gucci")
        self.assertEqual(output.decision_path, "C1")
        self.assertFalse(output.abstained)

    def test_precision_cascade_c2_used_when_no_c1(self):
        evidence = [
            Evidence("inst_0001", "gucci", 0.90, "C2"),
        ]
        probabilities = {"inst_0001": {"gucci": 0.8, "unknown": 0.2}}
        output = decide([self.instance], probabilities, evidence, self.config)[0]

        self.assertEqual(output.final_brand, "gucci")
        self.assertIsNone(output.product_brand)
        self.assertEqual(output.zone_brand, "gucci")
        self.assertEqual(output.decision_path, "C2")
        self.assertFalse(output.abstained)

    def test_precision_cascade_abstains_when_no_evidence(self):
        probabilities = {"inst_0001": {"unknown": 1.0}}
        output = decide([self.instance], probabilities, [], self.config)[0]

        self.assertEqual(output.final_brand, "unknown")
        self.assertIsNone(output.product_brand)
        self.assertIsNone(output.zone_brand)
        self.assertEqual(output.decision_path, "none")
        self.assertTrue(output.abstained)
