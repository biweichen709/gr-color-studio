import unittest

import numpy as np

from ricoh_color import codec, color, fit, lut, remap
from tests.looks import film


class Color(unittest.TestCase):
    def test_ciede2000_against_sharma_reference_pairs(self):
        pairs = [
            ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
            ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
            ((50, 0, 0), (50, -1, 2), 2.3669),
            ((50, 2.5, 0), (73, 25, -18), 27.1492),
            ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ]
        a = np.array([p[0] for p in pairs], dtype=float)
        b = np.array([p[1] for p in pairs], dtype=float)
        np.testing.assert_allclose(color.delta_e2000(a, b), [p[2] for p in pairs], atol=1e-4)

    def test_white_and_conversions(self):
        np.testing.assert_allclose(color.srgb_to_lab(np.ones(3)), [100, 0, 0], atol=1e-3)
        x = np.random.default_rng(0).random((100, 3))
        np.testing.assert_allclose(color.ycbcr_to_rgb(color.rgb_to_ycbcr(x)), x, atol=1e-9)
        np.testing.assert_allclose(color.linear_to_srgb(color.srgb_to_linear(x)), x, atol=1e-9)


class Remap(unittest.TestCase):
    def test_identity_look_or_zero_strength_keeps_bytes(self):
        spec = codec.TableSpec(grid=17, dtype="u16le", maxval=4095, offset=3)
        template = codec.encode(film(17), spec, bytes(spec.nbytes + 9))
        base = codec.decode(template, spec)
        for space in remap.SPACES:
            out = remap.remap(base, lut.identity(5), space=space)
            self.assertEqual(codec.encode(out, spec, template), template, space)
        out = remap.remap(base, film(9, sat=2.0), strength=0.0)
        self.assertEqual(codec.encode(out, spec, template), template)

    def test_node_values_are_look_of_base(self):
        base = film(17)
        look = film(33, warm=-0.05, sat=0.7)
        np.testing.assert_allclose(remap.remap(base, look), lut.apply(look, base), atol=1e-12)
        half = remap.remap(base, look, strength=0.5)
        np.testing.assert_allclose(half, (base + lut.apply(look, base)) / 2, atol=1e-12)

    def test_output_space_is_respected(self):
        display = film(9)
        look = film(9, sat=0.5)
        linear_base = color.srgb_to_linear(display)
        out = remap.remap(linear_base, look, space="linear")
        np.testing.assert_allclose(color.linear_to_srgb(out), lut.apply(look, display), atol=1e-9)


class Fit(unittest.TestCase):
    def test_fit_recovers_look_from_noisy_clustered_samples(self):
        rng = np.random.default_rng(0)
        target = film(65)
        centres = rng.random((30, 3))

        def sample(k):
            return np.clip(centres[rng.integers(0, 30, k)] + rng.normal(0, 0.08, (k, 3)), 0, 1)

        src = sample(120_000)
        dst = lut.apply(target, src) + rng.normal(0, 1 / 255, src.shape)
        fitted, info = fit.fit_lut(src, dst, grid=17)
        held_out = sample(10_000)
        stats = color.delta_e_stats(lut.apply(fitted, held_out), lut.apply(target, held_out))
        baseline = color.delta_e_stats(held_out, lut.apply(target, held_out))
        self.assertLess(stats["mean"], 0.5)
        self.assertGreater(baseline["mean"], 4)
        self.assertGreater(info["coverage"], 0.2)

    def test_unobserved_colours_stay_near_identity(self):
        src = np.random.default_rng(1).random((20_000, 3)) * 0.2
        fitted, _ = fit.fit_lut(src, src * 0.5, grid=9)
        self.assertLess(np.abs(lut.apply(fitted, np.array([0.95, 0.95, 0.95])) - 0.95).max(), 0.15)


if __name__ == "__main__":
    unittest.main()
