import unittest

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.fusion import decide
from eyewear_localization.perception import Florence2OCRBackend, build_ocr_backend
from eyewear_localization.schemas import Evidence, Instance


class Florence2AndCascadeTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalizationConfig(gazetteer=["cartier", "gucci", "ray-ban"], use_vlm_audit=False)
        self.instance = Instance("inst_0001", [10, 10, 30, 20])

    def test_build_florence2_ocr_backend(self):
        backend = build_ocr_backend("florence2")
        self.assertEqual(backend.name, "florence2")

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
