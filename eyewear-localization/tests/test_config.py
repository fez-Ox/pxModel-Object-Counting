from pathlib import Path
import tempfile
import unittest

from eyewear_localization.config import load_config


class ConfigTests(unittest.TestCase):
    def test_yaml_config_matches_spec_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "gazetteer: [Cartier]\n"
                "cue_reliability: {C1: 0.95}\n"
                "fusion: {temperature: 0.6, unknown_prior: 0.35, tau: 0.6, margin: 0.15}\n"
                "smoothing: {lambda: 0.25, gate_punknown: 0.5}\n"
                "cascade: {use_vlm_audit: false, uncertainty_band: [0.45, 0.7]}\n",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.gazetteer, ["cartier"])
        self.assertFalse(config.use_vlm_audit)
        self.assertEqual(config.uncertainty_band, (0.45, 0.7))
