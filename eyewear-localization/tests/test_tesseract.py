from pathlib import Path
import os
import tempfile
import unittest

from PIL import Image

from eyewear_localization.perception import TesseractOCRBackend


class TesseractParsingTests(unittest.TestCase):
    def test_invalid_confidence_and_punctuation_rows_are_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake-tesseract"
            script.write_text(
                "#!/usr/bin/env python3\n"
                "print('level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext')\n"
                "print('5\\t1\\t1\\t1\\t1\\t1\\t10\\t20\\t30\\t8\\t92\\tCartier')\n"
                "print('5\\t1\\t1\\t1\\t1\\t2\\t45\\t20\\t30\\t8\\t88\\tEyewear')\n"
                "print('5\\t1\\t1\\t1\\t2\\t1\\t10\\t40\\t30\\t8\\t0\\tGarbage')\n"
                "print('5\\t1\\t1\\t1\\t2\\t2\\t45\\t40\\t30\\t8\\t90\\t@@')\n",
                encoding="utf-8",
            )
            script.chmod(script.stat().st_mode | os.stat(script).st_mode | 0o111)
            backend = TesseractOCRBackend(
                executable=str(script),
                threshold_levels=(),
                max_dimension=512,
            )
            detections = backend._run(Image.new("RGB", (100, 100), "white"))

        texts = [item.text for item in detections]
        self.assertIn("Cartier", texts)
        self.assertIn("Eyewear", texts)
        self.assertIn("Cartier Eyewear", texts)
        self.assertNotIn("Garbage", texts)
        self.assertNotIn("@@", texts)


if __name__ == "__main__":
    unittest.main()
