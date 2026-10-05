import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ricoh_color import lut, profile, safety
from tests.looks import film


def plant_firmware(seed=0):
    rng = np.random.default_rng(seed)
    buf = bytearray(rng.integers(0, 256, 2_000_000, dtype=np.uint8).tobytes())
    names = b"\x00".join([b"Standard", b"Vivid", b"Monotone", b"Positive Film", b"Bleach Bypass", b"Retro"])
    anchor = 900_000
    buf[anchor:anchor + len(names)] = names
    mats = []
    for sat in (1.0, 1.3, 0.0001, 1.1, 0.4, 0.9):
        base = np.eye(3) * 0.9 + 0.05
        base /= base.sum(1, keepdims=True)
        m = (base - np.eye(3) / 3) * sat + np.eye(3) / 3
        mats.append(m / m.sum(1, keepdims=True))
    ccm_at = anchor + 2000
    buf[ccm_at:ccm_at + 9 * 6 * 2] = (np.concatenate([m.ravel() for m in mats]) * 4096).round().astype("<i2").tobytes()
    curve_at = anchor + 3000
    cv = b"".join((65535 * np.linspace(0, 1, 256) ** g).astype("<u2").tobytes() for g in (0.45, 0.5, 0.55))
    buf[curve_at:curve_at + len(cv)] = cv
    return bytes(buf), {"anchor": anchor, "ccm": ccm_at, "curve": curve_at}


class Detect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.at = plant_firmware()
        cls.found = profile.profile_scan(cls.data)

    def test_mode_anchors_and_cluster(self):
        self.assertGreaterEqual(len(self.found["anchors"]), 6)
        self.assertEqual(len(self.found["clusters"]), 1)
        self.assertIn("Positive Film", self.found["clusters"][0]["modes"])

    def test_ccm_bank_found_at_planted_offset(self):
        offsets = {h["offset"]: h for h in self.found["ccm"]}
        self.assertIn(self.at["ccm"], offsets)
        hit = offsets[self.at["ccm"]]
        self.assertEqual((hit["dtype"], hit["scale"], hit["count"]), ("int16", 4096, 6))

    def test_curve_bank_found(self):
        offsets = {h["offset"]: h for h in self.found["curves"]}
        self.assertIn(self.at["curve"], offsets)
        self.assertEqual(offsets[self.at["curve"]]["count"], 3)

    def test_mono_triple_role(self):
        rng = np.random.default_rng(2)
        buf = bytearray(rng.integers(0, 256, 400_000, dtype=np.uint8).tobytes())
        mono = (65535 * np.linspace(0, 1, 256) ** 0.7).astype("<u2").tobytes()
        buf[100_000:100_000 + 3 * len(mono)] = mono * 3
        found = profile.profile_scan(bytes(buf), whole_file=True)
        self.assertTrue(any(h["role"] == "mono_triple" for h in found["curves"]))

    def test_no_false_positives_on_noise(self):
        noise = np.random.default_rng(9).integers(0, 256, 2_000_000, dtype=np.uint8).tobytes()
        found = profile.profile_scan(noise, whole_file=True)
        self.assertEqual(found["ccm"], [])
        self.assertEqual(found["curves"], [])


class BakeDown(unittest.TestCase):
    def test_identity_look_fits_identity_matrix(self):
        m, stats = profile.fit_ccm(lut.identity(17))
        np.testing.assert_allclose(m, np.eye(3), atol=1e-6)
        self.assertLess(stats["mean"], 0.01)

    def test_fitted_matrix_is_white_preserving_and_lossy_is_reported(self):
        m, stats = profile.fit_ccm(film(17, sat=1.3, warm=0.02))
        np.testing.assert_allclose(m.sum(axis=1), [1, 1, 1], atol=1e-6)
        self.assertGreater(stats["mean"], 0.5)  # a 3x3 cannot carry the full look

    def test_encoders_are_byte_exact_and_length_preserving(self):
        m, _ = profile.fit_ccm(film(9, sat=1.2))
        spec = {"offset": 10, "dtype": "int16", "scale": 4096, "index": 1}
        enc = profile.encode_matrix(bytes(200), spec, m)
        self.assertEqual(len(enc), 200)
        back = np.frombuffer(enc, "<i2", count=9, offset=10 + 9 * 2).astype(float) / 4096
        np.testing.assert_allclose(back.reshape(3, 3), m, atol=1.0 / 4096 + 1e-9)

        cspec = {"offset": 4, "dtype": "u16le", "length": 256, "index": 2}
        cv = profile.fit_curve(film(9), 256)
        enc = profile.encode_curve(bytes(2000), cspec, cv)
        self.assertEqual(len(enc), 2000)
        back = np.frombuffer(enc, "<u2", count=256, offset=4 + 2 * 256 * 2).astype(float) / 65535
        np.testing.assert_allclose(back, cv, atol=1.0 / 65535 + 1e-9)


class DiffBackups(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def archive(self, name, files):
        root = self.tmp / name
        (root / "backup").mkdir(parents=True)
        entries = []
        for i, (target, data) in enumerate(files.items(), 1):
            fn = f"RCB{i:03d}.BIN"
            (root / "backup" / fn).write_bytes(data)
            entries.append({"tag": f"{i:03d}", "target": target, "file": fn, "status": "OK"})
        (root / "manifest.json").write_text(json.dumps({"kind": "backup-archive", "entries": entries}))
        return root

    def test_detects_changed_and_unchanged_files(self):
        a = self.archive("std", {"A:\\IC.bin": b"standard", "E:\\Cal.bin": b"calib"})
        b = self.archive("vivid", {"A:\\IC.bin": b"vividddd", "E:\\Cal.bin": b"calib"})
        result = profile.diff_archives(a, b)
        self.assertEqual(result["changed"], ["A:\\IC.bin"])
        self.assertEqual(result["unchanged_count"], 1)


if __name__ == "__main__":
    unittest.main()
