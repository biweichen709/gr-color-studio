import unittest

import numpy as np

from ricoh_color import codec, scan
from tests.looks import film


def embed(buf, spec, table):
    buf[spec.offset : spec.offset + spec.nbytes] = codec.encode(table, spec, bytes(buf))[
        spec.offset : spec.offset + spec.nbytes
    ]


class FindTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        buf = bytearray(rng.integers(0, 256, 1_400_000, dtype=np.uint8).tobytes())
        buf[100_000:140_000] = bytes(40_000)
        standard = codec.TableSpec(grid=17, dtype="u16le", maxval=4095, offset=0x12346)
        cls.specs = [
            standard,
            codec.TableSpec(grid=17, dtype="u16le", maxval=4095, offset=standard.offset + standard.nbytes),
            codec.TableSpec(grid=33, dtype="u16le", maxval=1023, fastest="b", offset=400_002),
            codec.TableSpec(grid=9, dtype="u8", maxval=255, layout="planar", offset=900_001),
            codec.TableSpec(grid=16, dtype="u16be", maxval=65535, pad=1, channels="bgr", offset=1_000_000),
        ]
        cls.tables = [
            film(17, warm=0.0, sat=1.0),
            film(17, warm=-0.04, sat=0.8, contrast=1.2),
            film(33),
            film(9, sat=1.5),
            film(16),
        ]
        for spec, table in zip(cls.specs, cls.tables):
            embed(buf, spec, table)
        curve = (4095 * np.linspace(0, 1, 1024) ** (1 / 2.2)).astype("<u2").tobytes()
        buf[1_300_000 : 1_300_000 + len(curve)] = curve
        cls.data = bytes(buf)
        cls.hits = {h["offset"]: h for h in scan.find_luts(cls.data)}

    def test_every_embedded_table_is_found_exactly(self):
        self.assertEqual(set(self.hits), {s.offset for s in self.specs})
        for spec, table in zip(self.specs, self.tables):
            hit = self.hits[spec.offset]
            found = codec.TableSpec(**hit["spec"])
            alternative = codec.TableSpec(**dict(hit["spec"], **hit["alternative"]))
            self.assertEqual((found.grid, found.dtype, found.layout, found.pad, found.maxval),
                             (spec.grid, spec.dtype, spec.layout, spec.pad, spec.maxval))
            errors = [np.abs(codec.decode(self.data, s) - table).max() for s in (found, alternative)]
            self.assertLess(min(errors), 1.0 / spec.maxval + 1e-6)

    def test_tone_curve_is_found(self):
        curves = scan.find_curves(self.data)
        self.assertIn((1_300_000, "u16le", 1024), [(c["offset"], c["dtype"], c["length"]) for c in curves])

    def test_no_tables_in_noise_or_smooth_non_tables(self):
        rng = np.random.default_rng(3)
        y, x = np.mgrid[0:240, 0:360]
        decoys = [
            rng.integers(0, 256, 500_000, dtype=np.uint8).tobytes(),
            np.stack([x / 360 * 255, y / 240 * 255, (x + y) / 600 * 255], -1).astype(np.uint8).tobytes(),
            (np.stack([x / 360, y / 240, (x + y) / 600], -1) * 4095).astype("<u2").tobytes(),
            (np.arange(100_000, dtype="<u4") * 4 + 0x53000000).tobytes(),
            (np.sin(np.linspace(0, 40 * np.pi, 150_000)) * 30000 + 32768).astype("<u2").tobytes(),
        ]
        data = b"".join(d + rng.integers(0, 256, 5000, dtype=np.uint8).tobytes() for d in decoys)
        self.assertEqual(scan.find_luts(data), [])


class Strings(unittest.TestCase):
    def test_paths_and_keywords_in_ascii_and_utf16(self):
        data = (
            b"\x00\x01junk A:\\Resource\\Param\\ic_positive.bin\x00"
            + "BleachBypass".encode("utf-16-le")
            + b"\xff\xfe nothing here \x00"
        )
        hits = list(scan.interesting_strings(data))
        self.assertEqual(len(hits), 2)
        self.assertTrue(hits[0]["path"])
        self.assertIn("positive", hits[0]["keywords"])
        self.assertEqual(hits[1]["encoding"], "utf-16-le")
        self.assertIn("bleach", hits[1]["keywords"])


if __name__ == "__main__":
    unittest.main()
