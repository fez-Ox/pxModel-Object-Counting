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

    def test_discrete_prompt_batch_preserves_prompt_order_and_thresholds(self):
        single_calls = []
        batch_calls = []

        def predictor(path: Path, prompt: str):
            single_calls.append(prompt)
            return {"boxes": [[0, 0, 20, 20]], "scores": [0.4]}

        def batch_predictor(path: Path, prompts: list[str]):
            batch_calls.append(list(prompts))
            return [
                {"boxes": [[index, 0, index + 20, 20]], "scores": [score]}
                for index, score in enumerate((0.4, 0.8, 0.3))
            ]

        localizer = SAM3Localizer(
            predictor,
            score_threshold=0.5,
            batch_predictor=batch_predictor,
            prompt_batch_size=3,
        )
        result = localizer.detect_prompts(
            Path("image.jpg"),
            ["people", "person", "faces of people"],
            thresholds=[0.5, 0.75, 0.25],
        )

        self.assertEqual(batch_calls, [["people", "person", "faces of people"]])
        self.assertEqual(single_calls, [])
        self.assertEqual([len(item) for item in result], [0, 1, 1])
        self.assertEqual(result[1][0].prompt, "person")
        self.assertEqual(result[2][0].prompt, "faces of people")

    def test_malformed_batch_item_replays_single_prompt_calls(self):
        calls = []

        def predictor(path: Path, prompt: str):
            calls.append(prompt)
            return {"boxes": [[0, 0, 20, 20]], "scores": [0.9]}

        def malformed_batch_predictor(path: Path, prompts: list[str]):
            return [
                {"boxes": [[0, 0, 20, 20]], "scores": [0.9]},
                None,
            ]

        localizer = SAM3Localizer(
            predictor,
            batch_predictor=malformed_batch_predictor,
            prompt_batch_size=2,
        )
        result = localizer.detect_prompts(Path("image.jpg"), ["people", "person"])

        self.assertEqual(calls, ["people", "person"])
        self.assertEqual([len(item) for item in result], [1, 1])

    def test_invalid_batch_shape_replays_single_prompt_calls(self):
        calls = []

        def predictor(path: Path, prompt: str):
            calls.append(prompt)
            return {"boxes": [[0, 0, 20, 20]], "scores": [0.9]}

        def bad_batch_predictor(path: Path, prompts: list[str]):
            return {"boxes": [[0, 0, 20, 20]], "scores": [0.9]}

        localizer = SAM3Localizer(
            predictor,
            batch_predictor=bad_batch_predictor,
            prompt_batch_size=2,
        )
        result = localizer.detect_prompts(Path("image.jpg"), ["people", "person"])

        self.assertEqual(calls, ["people", "person"])
        self.assertEqual([len(item) for item in result], [1, 1])
