from pathlib import Path
import tempfile
import unittest

from scripts.check_lossless_parity import compare


class ParityCheckerTests(unittest.TestCase):
    def _write(self, directory: str, name: str, payload: dict) -> Path:
        path = Path(directory) / name
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_ignores_timing_and_performance_metadata(self):
        reference = {
            "outputs": [{"brand": "cartier"}],
            "timings": {"total_pipeline_seconds": 10.0},
            "effective_config": {"performance": {"c1_batch_size": 1}},
        }
        optimized = {
            "outputs": [{"brand": "cartier"}],
            "timings": {"total_pipeline_seconds": 2.0},
            "effective_config": {"performance": {"c1_batch_size": 4}},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = compare(
                self._write(directory, "reference.json", reference),
                self._write(directory, "optimized.json", optimized),
            )
        self.assertEqual(result, (True, None))

    def test_detects_evidence_difference_even_when_final_label_matches(self):
        reference = {
            "evidence": [{"instance_id": "inst_0001", "brand": "cartier"}],
            "outputs": [{"brand": "cartier"}],
        }
        optimized = {
            "evidence": [{"instance_id": "inst_0001", "brand": "gucci"}],
            "outputs": [{"brand": "cartier"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = compare(
                self._write(directory, "reference.json", reference),
                self._write(directory, "optimized.json", optimized),
            )
        self.assertFalse(result[0])
        self.assertIn("evidence", result[1] or "")


if __name__ == "__main__":
    unittest.main()
