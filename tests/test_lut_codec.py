import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ricoh_color import codec, lut
from tests.looks import film


class Lut(unittest.TestCase):
    def test_identity_lookup_is_exact(self):
        x = np.random.default_rng(0).random((1000, 3))
        np.testing.assert_allclose(lut.apply(lut.identity(9), x), x, atol=1e-12)

    def test_cube_round_trip_and_red_fastest_order(self):
        table = lut.identity(5) ** np.array([1.0, 2.0, 0.5])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.cube"
            lut.write_cube(path, table, title="t")
            rows = [line for line in path.read_text().splitlines() if line[0].isdigit()]
            self.assertEqual(rows[1], "0.250000 0.000000 0.000000")
            np.testing.assert_allclose(lut.read_cube(path), table, atol=1e-6)

    def test_cube_rejects_other_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.cube"
            path.write_text("LUT_3D_SIZE 2\nDOMAIN_MAX 2 2 2\n" + "0 0 0\n" * 8)
            with self.assertRaisesRegex(ValueError, "DOMAIN_MAX"):
                lut.read_cube(path)

    def test_hald_identity_round_trip(self):
        image = lut.hald_identity(4).astype(np.float64) / 255
        np.testing.assert_allclose(lut.hald_to_lut(image), lut.identity(16), atol=0.5 / 255)

    def test_resample_preserves_smooth_look(self):
        x = np.random.default_rng(1).random((500, 3))
        fine = film(33)
        np.testing.assert_allclose(lut.apply(lut.resample(fine, 65), x), lut.apply(fine, x), atol=2e-3)


class Codec(unittest.TestCase):
    MAX = {"u8": 255, "u16le": 4095, "u16be": 4095, "u32le": 65535, "f32le": 1.0}

    def specs(self):
        for dtype, layout, fastest, channels, pad in itertools.product(
            codec.DTYPES, ("interleaved", "planar"), "rb", ("rgb", "bgr"), (0, 1)
        ):
            if layout == "planar" and pad:
                continue
            yield codec.TableSpec(
                grid=5, dtype=dtype, maxval=self.MAX[dtype], layout=layout,
                fastest=fastest, channels=channels, pad=pad, offset=7,
            )

    def test_decode_encode_is_byte_exact_and_keeps_length(self):
        rng = np.random.default_rng(0)
        for spec in self.specs():
            template = bytearray(rng.integers(0, 256, spec.nbytes + 20, dtype=np.uint8).tobytes())
            if spec.dtype == "f32le":
                template[7 : 7 + spec.nbytes] = rng.random(spec.count).astype("<f4").tobytes()
            template = bytes(template)
            self.assertEqual(codec.encode(codec.decode(template, spec), spec, template), template, spec)

    def test_encode_writes_only_the_table_region(self):
        rng = np.random.default_rng(1)
        table = film(5)
        for spec in self.specs():
            template = rng.integers(0, 256, spec.nbytes + 20, dtype=np.uint8).tobytes()
            out = codec.encode(table, spec, template)
            self.assertEqual(len(out), len(template))
            self.assertEqual(out[:7], template[:7])
            self.assertEqual(out[7 + spec.nbytes :], template[7 + spec.nbytes :])
            tolerance = 1e-6 if spec.dtype == "f32le" else 0.5 / spec.maxval + 1e-9
            np.testing.assert_allclose(codec.decode(out, spec), table, atol=tolerance, err_msg=str(spec))

    def test_memory_order(self):
        spec = codec.TableSpec(grid=2, dtype="u8", maxval=255, fastest="r", channels="rgb")
        data = codec.encode(lut.identity(2), spec, bytes(spec.nbytes))
        self.assertEqual(list(data[:6]), [0, 0, 0, 255, 0, 0])
        spec = codec.TableSpec(grid=2, dtype="u8", maxval=255, fastest="b", channels="bgr")
        data = codec.encode(lut.identity(2), spec, bytes(spec.nbytes))
        self.assertEqual(list(data[:6]), [0, 0, 0, 255, 0, 0])

    def test_spec_validation_and_bounds(self):
        with self.assertRaises(ValueError):
            codec.TableSpec(grid=5, layout="planar", pad=1)
        spec = codec.TableSpec(grid=5, offset=10)
        with self.assertRaisesRegex(ValueError, "file has"):
            codec.decode(bytes(spec.nbytes), spec)


if __name__ == "__main__":
    unittest.main()
