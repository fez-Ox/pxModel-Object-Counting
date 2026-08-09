from pathlib import Path
import unittest

from eyewear_localization.perception import SAM3_CLASS_PROMPTS, SAM3Localizer


class Sam3PromptTests(unittest.TestCase):
    def test_adapter_releases_native_model_callback_once(self):
        calls = []
        localizer = SAM3Localizer(lambda path, prompt: {"boxes": [], "scores": []}, release_callback=lambda: calls.append("released"))
        localizer.release()
        localizer.release()
        self.assertEqual(calls, ["released"])

    def test_adapter_uses_only_class_agnostic_prompts(self):
        calls = []

        def predictor(path: Path, prompt: str):
            calls.append(prompt)
            return {"boxes": [[0, 0, 20, 20]], "scores": [0.9]}

        detections = SAM3Localizer(predictor).detect(Path("image.jpg"))
        self.assertEqual(calls, list(SAM3_CLASS_PROMPTS))
        self.assertEqual(len(detections), 1)
        self.assertFalse(any("cartier" in prompt.lower() for prompt in calls))
