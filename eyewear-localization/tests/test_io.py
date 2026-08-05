from pathlib import Path
import tempfile
import unittest

from eyewear_localization.io import expand_inputs, materialize_inputs


class IoTests(unittest.TestCase):
    def test_folder_expansion_and_recursive_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "top.jpg").write_bytes(b"image")
            (root / "nested").mkdir()
            (root / "nested" / "deep.png").write_bytes(b"image")
            self.assertEqual(len(expand_inputs([str(root)])), 1)
            self.assertEqual(len(expand_inputs([str(root)], recursive=True)), 2)

    def test_materialize_local_source_preserves_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"image")
            with materialize_inputs([str(path)]) as sources:
                self.assertEqual(sources[0].source, str(path))
                self.assertEqual(sources[0].path, path.resolve())
