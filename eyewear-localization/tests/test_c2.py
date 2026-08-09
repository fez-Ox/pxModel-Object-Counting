import unittest

from eyewear_localization.cues import SignageScopeCue
from eyewear_localization.schemas import Instance, PosterRegion, Scope, Sign


class ScopeAssociationTests(unittest.TestCase):
    def setUp(self):
        self.cue = SignageScopeCue()

    def test_header_scope_is_inferred_from_items_below(self):
        instances = [
            Instance("inst_0001", [40, 40, 20, 20]),
            Instance("inst_0002", [70, 40, 20, 20]),
        ]
        signs = [Sign("s_01", "Cartier", "cartier", [40, 10, 50, 12], confidence=1.0)]
        scoped, evidence = self.cue.associate(instances, signs, [])
        self.assertEqual(scoped[0].scope.type, "bay_header")
        self.assertEqual({item.instance_id for item in evidence}, {"inst_0001", "inst_0002"})

    def test_row_end_scope_is_inferred_without_column_assumption(self):
        instances = [
            Instance("inst_0001", [20, 35, 20, 20]),
            Instance("inst_0002", [50, 35, 20, 20]),
            Instance("inst_0003", [80, 35, 20, 20]),
        ]
        signs = [Sign("s_01", "Oakley", "oakley", [125, 39, 18, 12], confidence=1.0)]
        scoped, evidence = self.cue.associate(instances, signs, [])
        self.assertEqual(scoped[0].scope.type, "row_label")
        self.assertTrue(evidence)

    def test_top_campaign_poster_header_can_scope_display_below(self):
        instances = [
            Instance("inst_0001", [40, 40, 20, 20]),
            Instance("inst_0002", [80, 40, 20, 20]),
        ]
        signs = [Sign("s_01", "Oakley", "oakley", [40, 8, 50, 12], confidence=1.0)]
        posters = [PosterRegion([0, 0, 150, 25], 0.95)]
        scoped, evidence = self.cue.associate(instances, signs, posters)
        self.assertEqual(scoped[0].scope.type, "bay_header")
        self.assertEqual({item.instance_id for item in evidence}, {"inst_0001", "inst_0002"})

    def test_repeated_same_brand_anchors_cover_intervening_instances(self):
        instances = [
            Instance("inst_0001", [10, 40, 20, 20]),
            Instance("inst_0002", [50, 40, 20, 20]),
            Instance("inst_0003", [90, 40, 20, 20]),
        ]
        signs = [
            Sign("s_left", "DARLEY", "oakley", [0, 5, 20, 10], confidence=0.68),
            Sign("s_right", "DARKY", "oakley", [100, 5, 20, 10], confidence=0.68),
        ]
        _, evidence = self.cue.associate(instances, signs, [])
        self.assertEqual({item.instance_id for item in evidence}, {"inst_0001", "inst_0002", "inst_0003"})
        self.assertTrue(all(item.confidence >= 0.70 for item in evidence))

    def test_sign_inside_poster_is_not_associated(self):
        instances = [Instance("inst_0001", [40, 40, 20, 20])]
        signs = [Sign("s_01", "Cartier", "cartier", [40, 10, 50, 12], confidence=1.0)]
        posters = [PosterRegion([0, 0, 150, 100], 0.95)]
        scoped, evidence = self.cue.associate(instances, signs, posters)
        self.assertEqual(scoped[0].scope.type, "none")
        self.assertEqual(evidence, [])
