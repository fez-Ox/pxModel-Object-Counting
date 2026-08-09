import unittest

from eyewear_localization.gazetteer import Gazetteer, normalize_text


class GazetteerTests(unittest.TestCase):
    def setUp(self):
        self.gazetteer = Gazetteer(["Cartier", "Ray-Ban", "Dolce & Gabbana"])

    def test_normalization_strips_diacritics_and_punctuation(self):
        self.assertEqual(normalize_text("Dolcé & Gabbana!"), "dolce and gabbana")

    def test_exact_match(self):
        match = self.gazetteer.match("CARTIER")
        self.assertIsNotNone(match)
        self.assertEqual(match.brand, "cartier")
        self.assertEqual(match.method, "exact")

    def test_token_containment(self):
        match = self.gazetteer.match("Ray Ban eyewear")
        self.assertIsNotNone(match)
        self.assertEqual(match.brand, "ray ban")
        self.assertEqual(match.method, "token_containment")

    def test_one_edit_match(self):
        match = self.gazetteer.match("Cartie")
        self.assertIsNotNone(match)
        self.assertEqual(match.brand, "cartier")
        self.assertEqual(match.method, "edit_distance")

    def test_stylized_oakley_alias_is_closed_set(self):
        gazetteer = Gazetteer(["oakley"])
        match = gazetteer.match("DARLEY")
        self.assertIsNotNone(match)
        self.assertEqual(match.brand, "oakley")
        self.assertEqual(match.method, "closed_set_ocr_alias")
        self.assertIsNone(self.gazetteer.match("DARLEY"))

    def test_unknown_text_is_not_invented(self):
        self.assertIsNone(self.gazetteer.match("unlisted label"))
