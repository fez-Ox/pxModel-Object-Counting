from pathlib import Path
import unittest

from eyewear_localization.perception import LocalizationDetection, SAM3Localizer
from eyewear_localization.scene_filter import SAM3SceneFilter
from eyewear_localization.schemas import Instance


def _predictor_factory(*, person_boxes=(), poster_boxes=(), shelf_boxes=()):
    def predictor(path: Path, prompt: str):
        lowered = prompt.lower()
        if any(word in lowered for word in ("people", "person", "faces")):
            boxes, scores = person_boxes, [0.9] * len(person_boxes)
        elif any(word in lowered for word in ("advertisement", "poster", "billboard")):
            boxes, scores = poster_boxes, [0.9] * len(poster_boxes)
        elif any(word in lowered for word in ("shelf",)):
            boxes, scores = shelf_boxes, [0.9] * len(shelf_boxes)
        else:
            boxes, scores = [], []
        return {"boxes": boxes, "scores": scores}
    return predictor


class SceneFilterTests(unittest.TestCase):
    def test_instance_inside_person_box_is_excluded(self):
        localizer = SAM3Localizer(
            _predictor_factory(person_boxes=[[0, 0, 200, 300]])
        )
        filter_ = SAM3SceneFilter(localizer, require_shelf=False)
        instances = [Instance("inst_0001", [50, 50, 40, 40])]
        kept, excluded = filter_.filter(Path("image.jpg"), instances)
        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 1)
        self.assertIn("worn_or_on_person", excluded[0].reasons)

    def test_instance_outside_detected_shelf_is_excluded(self):
        localizer = SAM3Localizer(
            _predictor_factory(shelf_boxes=[[0, 200, 300, 280]])
        )
        filter_ = SAM3SceneFilter(localizer, require_shelf=True)
        on_shelf = Instance("inst_0001", [50, 220, 40, 40])
        off_shelf = Instance("inst_0002", [50, 50, 40, 40])
        kept, excluded = filter_.filter(Path("image.jpg"), [on_shelf, off_shelf])
        self.assertEqual([item.id for item in kept], ["inst_0001"])
        self.assertEqual(len(excluded), 1)
        self.assertIn("not_on_detected_shelf", excluded[0].reasons)

    def test_shelf_requirement_can_be_disabled(self):
        localizer = SAM3Localizer(
            _predictor_factory(shelf_boxes=[[0, 200, 300, 280]])
        )
        filter_ = SAM3SceneFilter(localizer, require_shelf=False)
        instances = [Instance("inst_0001", [50, 50, 40, 40])]
        kept, excluded = filter_.filter(Path("image.jpg"), instances)
        self.assertEqual(kept, instances)
        self.assertEqual(excluded, [])

    def test_poster_regions_are_retained_for_sign_scope_filtering(self):
        localizer = SAM3Localizer(
            _predictor_factory(poster_boxes=[[0, 0, 200, 150]])
        )
        filter_ = SAM3SceneFilter(localizer, require_shelf=False)
        filter_.filter(Path("image.jpg"), [])
        self.assertEqual(filter_.last_poster_regions[0].bbox, [0.0, 0.0, 200.0, 150.0])
        self.assertIn("advertisements", filter_.last_poster_regions[0].source)

    def test_poster_instances_are_excluded(self):
        localizer = SAM3Localizer(
            _predictor_factory(poster_boxes=[[0, 0, 200, 150]])
        )
        filter_ = SAM3SceneFilter(localizer, require_shelf=False)
        instances = [Instance("inst_0001", [50, 50, 40, 40])]
        kept, excluded = filter_.filter(Path("image.jpg"), instances)
        self.assertEqual(kept, [])
        self.assertIn("inside_advertisement", excluded[0].reasons)

class SceneFilterFrontendTests(unittest.TestCase):
    def test_frontend_records_excluded_instances_in_perception_result(self):
        from eyewear_localization.gazetteer import Gazetteer
        from eyewear_localization.perception import PerceptionFrontend

        class FakeLocalizer:
            name = "fake"
            reliability = 1.0

            def detect(self, image_path):
                return [LocalizationDetection([10, 20, 50, 60], 0.9, "sunglasses")]

        localizer = SAM3Localizer(
            _predictor_factory(person_boxes=[[0, 0, 300, 300]])
        )
        frontend = PerceptionFrontend(
            localizer=FakeLocalizer(),
            ocr=None,
            gazetteer=Gazetteer([]),
            scene_filter=SAM3SceneFilter(localizer, require_shelf=False),
        )
        result = frontend.run(Path("image.jpg"))
        self.assertEqual(len(result.instances), 0)
        self.assertEqual(len(result.excluded_instances), 1)
        self.assertIn("worn_or_on_person", result.excluded_instances[0].reasons)
