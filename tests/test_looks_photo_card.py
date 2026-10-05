import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ricoh_color import card, looks, lut, photo, safety


class Looks(unittest.TestCase):
    def test_presets_are_valid_smooth_tables(self):
        for look in looks.PRESETS:
            table = looks.bake(look, grid=17)
            self.assertTrue(np.isfinite(table).all(), look.key)
            self.assertGreaterEqual(table.min(), 0.0, look.key)
            self.assertLessEqual(table.max(), 1.0, look.key)
            grey = table[np.arange(17), np.arange(17), np.arange(17)].mean(axis=-1)
            self.assertTrue(np.all(np.diff(grey) > 0), f"{look.key}: grey axis must stay monotonic")

    def test_mono_presets_are_neutral_without_toning(self):
        for key in ("acros", "acros_r", "mono"):
            table = looks.bake(looks.BY_KEY[key], grid=9)
            self.assertLess(np.ptp(table, axis=-1).max(), 1e-6, key)

    def test_zero_strength_and_neutral_look_are_identity(self):
        ident = lut.identity(9)
        zero = looks.bake(looks.BY_KEY["velvia"], looks.Adjust(strength=0.0), grid=9)
        np.testing.assert_allclose(zero, ident, atol=1e-12)
        np.testing.assert_allclose(looks.bake(looks.NEUTRAL, grid=9), ident, atol=1e-4)
        np.testing.assert_allclose(looks.bake_lut(lut.identity(5), grid=9), ident, atol=1e-4)

    def test_adjustments_move_in_the_expected_direction(self):
        mid = np.array([0.5, 0.5, 0.5])
        warm = looks.render(looks.NEUTRAL, mid, looks.Adjust(temp=0.5))
        self.assertGreater(warm[0], warm[2])
        sat = np.array([0.7, 0.4, 0.3])
        more = looks.render(looks.NEUTRAL, sat, looks.Adjust(saturation=1.4))
        self.assertGreater(np.ptp(more), np.ptp(sat))
        dark = np.array([0.2, 0.2, 0.2])
        self.assertLess(looks.render(looks.NEUTRAL, dark, looks.Adjust(contrast=1.0))[0], 0.2)
        self.assertGreater(looks.render(looks.NEUTRAL, np.zeros(3), looks.Adjust(fade=1.0))[0], 0.05)

    def test_monotone_curve(self):
        f = looks.monotone_curve(((0, 0.05), (0.3, 0.2), (0.5, 0.5), (1, 0.95)))
        x = np.linspace(0, 1, 1001)
        self.assertTrue(np.all(np.diff(f(x)) >= -1e-12))
        np.testing.assert_allclose(f(np.array([0.0, 0.5, 1.0])), [0.05, 0.5, 0.95])


class Photos(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_apply_matches_reference_interpolation(self):
        table = looks.bake(looks.BY_KEY["classic_neg"])
        pixels = (np.random.default_rng(0).random((64, 80, 3)) * 255).astype(np.uint8)
        reference = np.clip(np.rint(lut.apply(table, pixels / 255.0) * 255), 0, 255)
        self.assertLessEqual(np.abs(photo.apply_lut(table, pixels).astype(int) - reference).max(), 1)

    def test_jpeg_keeps_exif_icc_and_orientation(self):
        exif = Image.Exif()
        exif[0x0112] = 6
        exif[0x0110] = "GR IV"
        icc = b"\0" * 128
        Image.fromarray(np.zeros((30, 40, 3), np.uint8)).save(self.tmp / "in.jpg", exif=exif.tobytes(), icc_profile=icc)
        p = photo.load(self.tmp / "in.jpg")
        self.assertEqual(p.preview().shape, (40, 30, 3))
        photo.save_jpeg(self.tmp / "out.jpg", p.pixels, p)
        with Image.open(self.tmp / "out.jpg") as out:
            self.assertEqual(out.size, (40, 30))
            self.assertEqual(out.getexif()[0x0112], 6)
            self.assertEqual(out.getexif()[0x0110], "GR IV")
            self.assertEqual(out.info.get("icc_profile"), icc)


class Card(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.card = self.tmp / "card"
        (self.card / "script").mkdir(parents=True)

    def test_existing_files_are_moved_aside_never_overwritten(self):
        (self.card / "script" / "startup.ttl").write_text("; my own script\n")
        (self.card / safety.RUN_MARK).write_text("RUN")
        (self.card / "RCB001.BIN").write_bytes(b"older original")
        safety.backup_plan(["A:\\Resource\\x.bin"], self.tmp / "stage")
        moved = card.stage_to_card(self.tmp / "stage", self.card, stamp="T")
        self.assertEqual(sorted(moved), sorted([
            str(Path("RC_OLD/T/RCBRUN.TXT")), str(Path("RC_OLD/T/RCB001.BIN")),
            str(Path("RC_OLD/T/script/startup.ttl")),
        ]))
        self.assertEqual((self.card / "RC_OLD/T/RCB001.BIN").read_bytes(), b"older original")
        self.assertIn("ricoh_color backup", (self.card / "script" / "startup.ttl").read_text())

    def test_entry_sweep_written_and_removed(self):
        from ricoh_color import firmware

        files = firmware.entry_sweep(78350)
        self.assertEqual(len(files), 1001)
        card.write_files(self.card, files)
        (self.card / "00078350.ABC").write_bytes(firmware.ENTRY_MARKER)
        (self.card / "00078350.007").write_bytes(b"user file")
        self.assertEqual(card.remove_sweep(self.card, 78350, firmware.ENTRY_MARKER), 999)
        self.assertTrue((self.card / "00078350.ABC").exists())
        self.assertTrue((self.card / "00078350.007").exists())
        self.assertTrue((self.card / "DEVELOP.MOD").exists())

    def test_filesystem_name_is_reported_on_windows_only(self):
        fs = card.filesystem(self.tmp)
        if os.name == "nt":
            self.assertTrue(fs)
        else:
            self.assertIsNone(fs)

    def test_identical_files_are_left_in_place_and_finish_moves_script(self):
        safety.backup_plan(["A:\\Resource\\x.bin"], self.tmp / "stage")
        card.stage_to_card(self.tmp / "stage", self.card, stamp="A")
        self.assertEqual(card.stage_to_card(self.tmp / "stage", self.card, stamp="B"), [])
        self.assertEqual(card.finish(self.card, stamp="C"), str(Path("RC_OLD/C/script/startup.ttl")))
        self.assertIsNone(card.finish(self.card))
        self.assertEqual(json.loads((self.tmp / "stage" / "plan.json").read_text())["kind"], "backup")


if __name__ == "__main__":
    unittest.main()
