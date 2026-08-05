import unittest

from eyewear_localization.config import LocalizationConfig
from eyewear_localization.fusion import decide, fuse_evidence, smooth_continuity
from eyewear_localization.schemas import Evidence, Instance


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalizationConfig(gazetteer=["cartier", "oakley"], use_vlm_audit=False)
        self.instance = Instance("inst_0001", [10, 10, 20, 20])

    def test_zero_evidence_abstains_as_unknown(self):
        probabilities = fuse_evidence([self.instance], [], self.config)
        output = decide([self.instance], probabilities, [], self.config)[0]
        self.assertEqual(output.brand, "unknown")
        self.assertTrue(output.abstained)
        self.assertGreater(output.probabilities["unknown"], output.probabilities["cartier"])

    def test_strong_cue_is_accepted(self):
        evidence = [Evidence("inst_0001", "cartier", 1.0, "C1")]
        probabilities = fuse_evidence([self.instance], evidence, self.config)
        output = decide([self.instance], probabilities, evidence, self.config)[0]
        self.assertEqual(output.brand, "cartier")
        self.assertFalse(output.abstained)

    def test_smoothing_does_not_propagate_from_uncertain_neighbor(self):
        right = Instance("inst_0002", [40, 10, 20, 20])
        probabilities = {
            "inst_0001": {"cartier": 0.9, "oakley": 0.05, "unknown": 0.05},
            "inst_0002": {"cartier": 0.1, "oakley": 0.1, "unknown": 0.8},
        }
        smoothed = smooth_continuity(probabilities, [self.instance, right], self.config)
        self.assertEqual(smoothed["inst_0001"], probabilities["inst_0001"])
