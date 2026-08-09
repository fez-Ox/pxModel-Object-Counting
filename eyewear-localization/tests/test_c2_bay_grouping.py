import unittest

from eyewear_localization.cues import SignageScopeCue
from eyewear_localization.schemas import Instance, Sign, TextDetection


class BayGroupingTests(unittest.TestCase):
    def setUp(self):
        self.cue = SignageScopeCue()

    def test_all_items_in_nearest_horizontal_group_are_attributed(self):
        instances = [
            Instance("inst_0001", [40, 40, 20, 20]),
            Instance("inst_0002", [70, 40, 20, 20]),
            Instance("inst_0003", [100, 40, 20, 20]),
            # A separate distant bay must not be attributed to this sign.
            Instance("inst_0004", [420, 38, 20, 20]),
        ]
        signs = [Sign("s_01", "Cartier", "cartier", [60, 8, 30, 10], confidence=1.0)]
        scoped, evidence = self.cue.associate(instances, signs, [])
        self.assertEqual(scoped[0].scope.type, "bay_header")
        attributed = {item.instance_id for item in evidence}
        self.assertEqual(attributed, {"inst_0001", "inst_0002", "inst_0003"})
        self.assertNotIn("inst_0004", attributed)

    def test_sibling_signs_split_their_bays(self):
        instances = [
            Instance("inst_0001", [40, 40, 20, 20]),
            Instance("inst_0002", [80, 40, 20, 20]),
            Instance("inst_0003", [340, 40, 20, 20]),
            Instance("inst_0004", [380, 40, 20, 20]),
        ]
        signs = [
            Sign("s_01", "Cartier", "cartier", [45, 8, 30, 10], confidence=1.0),
            Sign("s_02", "Oakley", "oakley", [345, 8, 30, 10], confidence=1.0),
        ]
        scoped, evidence = self.cue.associate(instances, signs, [])
        cartier = {item.instance_id for item in evidence if item.brand == "cartier"}
        oakley = {item.instance_id for item in evidence if item.brand == "oakley"}
        self.assertEqual(cartier, {"inst_0001", "inst_0002"})
        self.assertEqual(oakley, {"inst_0003", "inst_0004"})

    def test_unknown_neighbor_label_is_only_a_column_separator(self):
        instances = [
            Instance("inst_0001", [40, 40, 20, 20]),
            Instance("inst_0002", [240, 40, 20, 20]),
        ]
        signs = [Sign("s_01", "Oakley", "oakley", [45, 8, 30, 10], confidence=1.0)]
        raw = [TextDetection("Meta", [245, 8, 40, 10], 0.9, source="ocr")]
        _scoped, evidence = self.cue.associate(
            instances, signs, [], image_width=300, text_detections=raw
        )
        self.assertEqual({item.instance_id for item in evidence}, {"inst_0001"})
        self.assertTrue(all(item.brand == "oakley" for item in evidence))

    def test_bottom_bay_labels_scope_items_above_them(self):
        instances = [
            Instance("inst_0001", [40, 40, 20, 20]),
            Instance("inst_0002", [80, 40, 20, 20]),
            Instance("inst_0003", [340, 40, 20, 20]),
            Instance("inst_0004", [380, 40, 20, 20]),
        ]
        signs = [
            Sign("s_01", "Cartier", "cartier", [45, 80, 30, 10], confidence=1.0),
            Sign("s_02", "Oakley", "oakley", [345, 80, 30, 10], confidence=1.0),
        ]
        scoped, evidence = self.cue.associate(instances, signs, [])
        self.assertEqual({item.instance_id for item in evidence if item.brand == "cartier"}, {"inst_0001", "inst_0002"})
        self.assertEqual({item.instance_id for item in evidence if item.brand == "oakley"}, {"inst_0003", "inst_0004"})
        self.assertTrue(all(sign.scope.type == "bay_header" for sign in scoped))

    def test_row_end_label_attributes_entire_row(self):
        instances = [
            Instance("inst_0001", [20, 35, 20, 20]),
            Instance("inst_0002", [50, 35, 20, 20]),
            Instance("inst_0003", [80, 35, 20, 20]),
        ]
        signs = [Sign("s_01", "Oakley", "oakley", [125, 39, 18, 12], confidence=1.0)]
        scoped, evidence = self.cue.associate(instances, signs, [])
        self.assertEqual(scoped[0].scope.type, "row_label")
        self.assertEqual(
            {item.instance_id for item in evidence},
            {"inst_0001", "inst_0002", "inst_0003"},
        )