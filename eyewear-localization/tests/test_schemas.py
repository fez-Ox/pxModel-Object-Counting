import unittest

from eyewear_localization.schemas import AttributionOutput, Evidence, Instance, Scope, xyxy_to_xywh


class SchemaTests(unittest.TestCase):
    def test_instance_serializes_spec_coordinates(self):
        instance = Instance("inst_0001", [10, 20, 30, 40])
        self.assertEqual(instance.centroid, [25.0, 40.0])
        self.assertEqual(instance.to_dict()["bbox"], [10.0, 20.0, 30.0, 40.0])

    def test_xyxy_conversion(self):
        self.assertEqual(xyxy_to_xywh([1, 2, 11, 22]), [1.0, 2.0, 10.0, 20.0])

    def test_invalid_evidence_cannot_emit_unknown(self):
        with self.assertRaises(ValueError):
            Evidence("inst_0001", "unknown", 1.0, "C1")

    def test_output_requires_unknown_probability(self):
        with self.assertRaises(ValueError):
            AttributionOutput("inst_0001", "unknown", True, {"cartier": 1.0})

    def test_scope_rejects_unknown_layout_type(self):
        with self.assertRaises(ValueError):
            Scope("column_guess", [0, 0, 1, 1], 0.5)
