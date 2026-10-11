import json
import unittest

import numpy as np

from oracle import (
    FIXTURE_PATH,
    INPUT_PATH,
    MANIFEST_PATH,
    PINS_PATH,
    generate_twice,
    verify_fixture,
)

BUDGET_PATH = PINS_PATH.parent / "budgets.json"


class OracleFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays, cls.repeatability = generate_twice()

    def test_pinned_reference_repeats_bitwise(self):
        self.assertTrue(self.repeatability)
        for name, measurement in self.repeatability.items():
            with self.subTest(array=name):
                self.assertTrue(measurement["bitwise_equal"])
                self.assertEqual(measurement["max_abs_difference"], 0.0)

    def test_repeated_run_matches_checked_in_stage_fixtures(self):
        verify_fixture(self.arrays, self.repeatability)
        with np.load(FIXTURE_PATH, allow_pickle=False) as fixture:
            self.assertEqual(set(fixture.files), set(self.arrays))

    def test_fixed_ids_voice_row_and_provenance_are_recorded(self):
        inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(inputs["phoneme_ids"], [50, 83, 156, 54, 57, 135])
        self.assertEqual(inputs["input_ids"], [0, *inputs["phoneme_ids"], 0])
        self.assertEqual(inputs["voice_style_index"], 5)
        self.assertEqual(manifest["model"]["revision"], pins["model"]["revision"])
        self.assertEqual(manifest["inference_source"]["revision"], pins["inference_source"]["revision"])
        self.assertEqual(manifest["repeatability"]["runs"], 2)

    def test_every_stage_array_has_shape_dtype_and_hash_provenance(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["arrays"]), set(self.arrays))
        for name, array in self.arrays.items():
            with self.subTest(array=name):
                item = manifest["arrays"][name]
                self.assertEqual(item["shape"], list(array.shape))
                self.assertEqual(item["dtype"], str(array.dtype))
                self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["budgets"]["path"], "../budgets.json")
        self.assertRegex(manifest["budgets"]["sha256"], r"^[0-9a-f]{64}$")

    def test_error_budgets_match_reference_repeatability_and_float32_bound(self):
        budgets = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
        repeatability = budgets["reference_repeatability"]
        observed_max = max(item["max_abs_difference"] for item in self.repeatability.values())
        self.assertEqual(repeatability["observed_max_abs_difference"], observed_max)
        self.assertEqual(
            repeatability["observed_waveform_normalized_rmse"],
            self.repeatability["waveform"]["normalized_rmse"],
        )
        precision = budgets["precision_rationale"]
        unit_roundoff = precision["unit_roundoff"]
        terms = precision["largest_configured_reduction_terms"]
        gamma = terms * unit_roundoff / (1.0 - terms * unit_roundoff)
        self.assertAlmostEqual(precision["gamma_3270"], gamma, places=10)
        self.assertIn("noise", budgets["float32_stage_comparison"]["applies_to"])
        self.assertIn("excitation_harmonic", budgets["exact_comparison"])


if __name__ == "__main__":
    unittest.main()
