import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_speedup import compare_runs, summarize


class SpeedupReportTests(unittest.TestCase):
    def test_summarize_reports_mean_and_percentile(self):
        result = summarize([1.0, 2.0, 4.0])
        self.assertAlmostEqual(result["mean"], 7.0 / 3.0)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["p90"], 4.0)
        self.assertEqual(result["sum"], 7.0)

    def test_compare_runs_reports_positive_speedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            optimized = root / "optimized"
            reference.mkdir()
            optimized.mkdir()
            reference_payload = {
                "timings": {
                    "l0_perception_seconds": 10.0,
                    "c1_onproduct_seconds": 5.0,
                    "c2_c4_fusion_seconds": 1.0,
                    "brand_association_total_seconds": 6.0,
                    "total_pipeline_seconds": 16.0,
                }
            }
            optimized_payload = {
                "timings": {
                    "l0_perception_seconds": 8.0,
                    "c1_onproduct_seconds": 4.0,
                    "c2_c4_fusion_seconds": 1.0,
                    "brand_association_total_seconds": 5.0,
                    "total_pipeline_seconds": 13.0,
                }
            }
            (reference / "image.json").write_text(json.dumps(reference_payload), encoding="utf-8")
            (optimized / "image.json").write_text(json.dumps(optimized_payload), encoding="utf-8")

            report = compare_runs(reference, optimized)

        total = report["timings"]["total_pipeline_seconds"]
        self.assertAlmostEqual(total["mean_delta_seconds"], -3.0)
        self.assertAlmostEqual(total["mean_speedup_percent"], 18.75)
        self.assertAlmostEqual(total["mean_speedup_factor"], 16.0 / 13.0)
        self.assertAlmostEqual(report["per_image_total"]["image"]["speedup_percent"], 18.75)
