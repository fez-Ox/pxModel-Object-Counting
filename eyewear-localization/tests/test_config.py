from pathlib import Path
import tempfile
import unittest

from eyewear_localization.config import LocalizationConfig, load_config


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
        self.assertEqual(config.cascade_t1, 0.70)
        self.assertEqual(config.cascade_t2, 0.75)
        self.assertEqual(config.c1_scales, (1.0, 2.0, 4.0))


class OverrideTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalizationConfig(
            gazetteer=["cartier"],
            tau=0.6,
            margin=0.15,
            temperature=0.6,
            smoothing_lambda=0.25,
        )

    def test_with_overrides_returns_copy(self):
        new = self.config.with_overrides({"fusion": {"tau": 0.5}})
        self.assertEqual(new.tau, 0.5)
        self.assertEqual(self.config.tau, 0.6, "original unchanged")

    def test_with_overrides_deep_merges_cue_reliability(self):
        new = self.config.with_overrides({"cue_reliability": {"C2": 0.9}})
        self.assertEqual(new.cue_reliability["C2"], 0.9)
        self.assertEqual(new.cue_reliability["C1"], 0.95, "other cues preserved")

    def test_with_overrides_none_returns_self(self):
        self.assertIs(self.config.with_overrides(None), self.config)
        self.assertIs(self.config.with_overrides({}), self.config)

    def test_with_overrides_smoothing(self):
        new = self.config.with_overrides({"smoothing": {"lambda": 0.4}})
        self.assertEqual(new.smoothing_lambda, 0.4)
        self.assertEqual(new.smoothing_gate_punknown, 0.5, "other smoothing preserved")

    def test_cli_parse_set_builds_nested_dict(self):
        from eyewear_localization.cli import set_overrides_from_args
        from argparse import Namespace
        args = Namespace(
            set=["fusion.tau=0.5", "cue_reliability.C2=0.8"],
            tau=None, margin=None, temperature=None, unknown_prior=None,
            smooth_lambda=None, smooth_gate=None,
            person_threshold=None, poster_threshold=None, shelf_threshold=None,
            cue_reliability=[],
        )
        overrides = set_overrides_from_args(args)
        self.assertEqual(overrides["fusion"]["tau"], 0.5)
        self.assertEqual(overrides["cue_reliability"]["C2"], 0.8)

    def test_cli_sugar_flags_take_precedence(self):
        from eyewear_localization.cli import set_overrides_from_args
        from argparse import Namespace
        args = Namespace(
            set=["fusion.tau=0.1"],
            tau=0.9, margin=0.2, temperature=None, unknown_prior=None,
            smooth_lambda=None, smooth_gate=None,
            person_threshold=None, poster_threshold=None, shelf_threshold=None,
            cue_reliability=["C1=0.7"],
        )
        overrides = set_overrides_from_args(args)
        self.assertEqual(overrides["fusion"]["tau"], 0.9, "sugar overrides --set")
        self.assertEqual(overrides["fusion"]["margin"], 0.2)
        self.assertEqual(overrides["cue_reliability"]["C1"], 0.7)

    def test_refined_c1_defaults(self):
        config = LocalizationConfig()
        self.assertEqual(config.cue_reliability["C1"], 0.95)
        self.assertEqual(config.cue_reliability["C2"], 0.30)
        self.assertEqual(config.c1_margin, 0.25)
        self.assertEqual(config.c1_wide_margin, 0.40)
        self.assertEqual(config.c1_scales, (1.0, 2.0, 4.0))
        self.assertTrue(config.c1_use_clahe)
        self.assertTrue(config.c1_dual_polarity)
        self.assertEqual(config.ocr_fallback_budget, 12)
        self.assertEqual(config.c1_batch_size, 4)
        self.assertEqual(config.sam3_prompt_batch_size, 4)

    def test_refined_c1_overrides_round_trip(self):
        new = self.config.with_overrides({
            "cascade": {"c1_threshold": 0.65, "c2_threshold": 0.80},
            "c1": {"wide_margin": 0.5, "scales": [1.0, 3.0]},
            "ocr": {"fallback_budget": 16},
        })
        self.assertEqual(new.cascade_t1, 0.65)
        self.assertEqual(new.cascade_t2, 0.80)
        self.assertEqual(new.c1_wide_margin, 0.5)
        self.assertEqual(new.c1_scales, (1.0, 3.0))
        self.assertEqual(new.ocr_fallback_budget, 16)

    def test_performance_overrides_round_trip(self):
        new = self.config.with_overrides({
            "performance": {
                "c1_batch_size": 4,
                "sam3_prompt_batch_size": 3,
                "sam3_compile": True,
            }
        })
        self.assertEqual(new.c1_batch_size, 4)
        self.assertEqual(new.sam3_prompt_batch_size, 3)
        self.assertTrue(new.sam3_compile)
        self.assertEqual(new.to_dict()["performance"]["c1_batch_size"], 4)

    def test_full_round_trip_overrides_to_config(self):
        new = self.config.with_overrides({
            "fusion": {"tau": 0.35, "margin": 0.1},
            "smoothing": {"lambda": 0.4},
            "cue_reliability": {"C2": 0.8},
        })
        self.assertEqual(new.tau, 0.35)
        self.assertEqual(new.margin, 0.1)
        self.assertEqual(new.smoothing_lambda, 0.4)
        self.assertEqual(new.cue_reliability["C1"], 0.95)
        self.assertEqual(new.cue_reliability["C2"], 0.8)
        raw = new.to_dict()
        self.assertEqual(raw["fusion"]["tau"], 0.35)
        self.assertEqual(raw["cue_reliability"]["C2"], 0.8)
