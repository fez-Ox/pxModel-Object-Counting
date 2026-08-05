import unittest

from eyewear_localization.registry import ModelRegistry


class RegistryTests(unittest.TestCase):
    def test_missing_model_uses_fallback(self):
        registry = ModelRegistry()
        registry.register("optional", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing")), fallback=lambda **kwargs: "fallback")
        self.assertEqual(registry.build("optional"), "fallback")
