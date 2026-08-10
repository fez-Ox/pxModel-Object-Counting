from pathlib import Path
import tempfile
import unittest

import torch
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


class DeterministicBatchOCR:
    name = "deterministic-batch-ocr"
    reliability = 1.0

    def __init__(self):
        self.single_calls = 0
        self.batch_calls = 0

    @staticmethod
    def _result():
        return [TextDetection("Cartier", [1, 1, 10, 5], 0.9, source="deterministic")]

    def detect_preprocessed(self, image):
        self.single_calls += 1
        return self._result()

    def detect_preprocessed_batch(self, images):
        self.batch_calls += 1
        return [self._result() for _ in images]


class FakeFlorenceInputs(dict):
    def to(self, device):
        return self


class FakeFlorenceProcessor:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, *, text, images, return_tensors, padding=False):
        batch_size = len(images) if isinstance(images, list) else 1
        self.batch_sizes.append(batch_size)
        return FakeFlorenceInputs(
            input_ids=torch.zeros((batch_size, 1), dtype=torch.long),
            pixel_values=torch.zeros((batch_size, 3, 4, 4), dtype=torch.float32),
        )

    def batch_decode(self, generated_ids, skip_special_tokens=False):
        return [f"response-{index}" for index in range(generated_ids.shape[0])]

    def post_process_generation(self, generated_text, *, task, image_size):
        index = int(generated_text.rsplit("-", 1)[1])
        return {
            task: {
                "labels": ["Cartier" if index == 0 else "Gucci"],
                "bboxes": [[0, 0, 20, 10]],
            }
        }


class FakeFlorenceModel:
    def __init__(self):
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._parameter

    def generate(self, *, input_ids, pixel_values, max_new_tokens, num_beams, do_sample):
        return torch.zeros((input_ids.shape[0], 1), dtype=torch.long)


class Florence2AndCascadeTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalizationConfig(gazetteer=["cartier", "gucci", "ray-ban"], use_vlm_audit=False)
        self.instance = Instance("inst_0001", [10, 10, 30, 20])

    def test_build_florence2_ocr_backend(self):
        backend = build_ocr_backend("florence2")
        self.assertEqual(backend.name, "florence2")

    def test_florence_batch_parser_keeps_one_result_per_input(self):
        backend = Florence2OCRBackend(device="cpu")
        processor = FakeFlorenceProcessor()
        backend._processor = processor
        backend._model = FakeFlorenceModel()
        backend.actual_device = "cpu"
        output = backend.detect_preprocessed_batch([
            Image.new("RGB", (40, 20), "white"),
            Image.new("RGB", (40, 20), "white"),
        ])

        self.assertEqual(processor.batch_sizes, [2])
        self.assertEqual([[item.text for item in detections] for detections in output], [["Cartier"], ["Gucci"]])
        self.assertEqual(output[0][0].bbox, [0.0, 0.0, 20.0, 10.0])

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

    def test_selective_fallback_keeps_order_sensitive_budget_sequential(self):
        primary = StubOCR([TextDetection("unrelated", [1, 1, 10, 5], 0.9, source="primary")])
        fallback = StubOCR([TextDetection("unrelated", [1, 1, 10, 5], 0.9, source="fallback")])
        ocr = SelectiveOCRBackend(
            primary,
            fallback,
            Gazetteer(["cartier"]),
            max_fallback_calls=2,
        )
        cue = OnProductBrandingCue(
            ocr,
            Gazetteer(["cartier"]),
            margin=0.0,
            scales=(1.0,),
            sharpen=False,
            use_clahe=True,
            dual_polarity=True,
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            cue.emit(
                image_path,
                [
                    Instance("inst_0001", [10, 10, 30, 20]),
                    Instance("inst_0002", [50, 10, 30, 20]),
                ],
            )

        # The first unmatched crop owns both available variant calls, exactly
        # as in the reference loop; the later crop cannot steal a slot.
        self.assertEqual(fallback.calls, 2)

    def test_c1_batch_scheduler_matches_reference_evidence(self):
        instances = [
            Instance("inst_0001", [10, 10, 30, 20]),
            Instance("inst_0002", [50, 10, 30, 20]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            sequential_ocr = DeterministicBatchOCR()
            batched_ocr = DeterministicBatchOCR()
            sequential = OnProductBrandingCue(
                sequential_ocr,
                Gazetteer(["cartier"]),
                margin=0.0,
                scales=(1.0, 2.0),
                sharpen=False,
                batch_size=1,
            ).emit(image_path, instances)
            batched = OnProductBrandingCue(
                batched_ocr,
                Gazetteer(["cartier"]),
                margin=0.0,
                scales=(1.0, 2.0),
                sharpen=False,
                batch_size=2,
            ).emit(image_path, instances)

        self.assertEqual([item.to_dict() for item in batched], [item.to_dict() for item in sequential])
        self.assertEqual(batched_ocr.batch_calls, 1)
        self.assertEqual(batched_ocr.single_calls, 0)

    def test_selective_ocr_runs_scene_fallback_even_when_primary_finds_one_brand(self):
        primary = StubOCR([TextDetection("Cartier", [1, 1, 10, 5], 0.9, source="primary")])
        fallback = StubOCR([TextDetection("Gucci", [2, 2, 12, 5], 0.95, source="fallback")])
        backend = SelectiveOCRBackend(primary, fallback, Gazetteer(["cartier", "gucci"]), max_fallback_calls=1)

        output = backend.detect(object())
        self.assertEqual({item.text for item in output}, {"Cartier", "Gucci"})

    def test_spatial_proximity_weighting_favors_on_frame_brand(self):
        # Instance: x=10, y=50, w=50, h=40. Crop margin=0.25 -> top=40.
        # Text 1: Burberry at local_y=0 -> global_y=40..45 (fully above instance y=50)
        # Text 2: Michael Kors at local_y=15 -> global_y=55..65 (inside instance y=50..90)
        ocr = StubOCR([
            TextDetection("BURBERRY", [10, 0, 30, 5], 0.88, source="primary"),
            TextDetection("MICHAEL KORS", [10, 15, 30, 10], 0.72, source="primary"),
        ])
        cue = OnProductBrandingCue(
            ocr,
            Gazetteer(["burberry", "michael kors"]),
            margin=0.25,
            scales=(1.0,),
            sharpen=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            evidence = cue.emit(image_path, [Instance("inst_0001", [10, 50, 50, 40])])

        # Michael Kors (on frame, 0.72) ranks higher than Burberry (outside frame, 0.88 * 0.75 = 0.66)
        self.assertTrue(len(evidence) >= 2)
        self.assertEqual(evidence[0].brand, "michael kors")

    def test_build_got_ocr2_backend(self):
        backend = build_ocr_backend("got-ocr2")
        self.assertEqual(backend.name, "got-ocr2")

    def test_c1_clahe_and_dual_polarity(self):
        ocr = StubOCR([TextDetection("Oakley", [2, 2, 12, 5], 0.95)])
        cue = OnProductBrandingCue(
            ocr,
            Gazetteer(["oakley"]),
            use_clahe=True,
            dual_polarity=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "black").save(image_path)
            evidence = cue.emit(image_path, [Instance("inst_0001", [10, 10, 30, 20])])
        self.assertTrue(len(evidence) >= 1)
        self.assertEqual(evidence[0].brand, "oakley")

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
