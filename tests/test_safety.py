import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ricoh_color import safety
from tests.ttl_sim import Camera

FOO = "A:\\Resource\\Param\\foo.bin"
BAR = "A:\\Resource\\Param\\bar.bin"
BLK = "E:\\BlkCtl15.bin"
GONE = "A:\\Resource\\Param\\missing.bin"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Workflow(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.drives = {d: self.tmp / d for d in "ACE"}
        for d in self.drives.values():
            d.mkdir()
        (self.drives["A"] / "Resource" / "Param").mkdir(parents=True)
        self.foo = bytes(range(256)) * 4
        self.bar = b"bar-original" * 10
        self.write("A", FOO, self.foo)
        self.write("A", BAR, self.bar)
        self.write("E", BLK, b"\xa5\x5a\x5a\xa5\xe0\x32\x01\x00calibration")
        self.camera = Camera(self.drives)

    def write(self, drive, name, data):
        self.drives[drive].joinpath(*name[3:].split("\\")).write_bytes(data)

    def read(self, drive, name):
        return self.drives[drive].joinpath(*name[3:].split("\\")).read_bytes()

    def insert_card(self, stage):
        shutil.copytree(stage / "card", self.drives["C"], dirs_exist_ok=True)

    def boot(self):
        script = self.drives["C"] / "script" / "startup.ttl"
        return self.camera.run(script.read_text())

    def backup(self):
        stage = self.tmp / "stage-backup"
        safety.backup_plan([FOO, BAR, BLK, GONE, FOO], stage)
        self.insert_card(stage)
        self.boot()
        return safety.backup_verify(stage / "plan.json", self.drives["C"], self.tmp / "archive")

    def test_scripts_use_only_camera_proven_commands(self):
        stage = self.tmp / "s"
        safety.backup_plan([FOO, BLK], stage)
        texts = [(stage / "card/script/startup.ttl").read_text()]
        self.backup()
        new = self.tmp / "new.bin"
        new.write_bytes(bytes(len(self.foo)))
        safety.write_plan("install", self.tmp / "archive", self.tmp / "i", {FOO: new})
        texts.append((self.tmp / "i/card/script/startup.ttl").read_text())
        for text in texts:
            labels = {line[1:] for line in text.splitlines() if line.startswith(":")}
            for line in text.splitlines():
                line = line.strip()
                if not line or line[0] in ";:":
                    continue
                words = line.split()
                if len(words) == 3 and words[1] == "=":
                    continue
                self.assertIn(words[0], safety.PROVEN_COMMANDS, line)
                if words[0] == "goto":
                    self.assertIn(words[1], labels)
                if words[0] == "if":
                    self.assertIn(words[2], {"=", "<>", "<", ">"}, line)

    def test_backup_reads_only_and_archives_verified_copies(self):
        manifest = self.backup()
        status = {e["target"]: e["status"] for e in manifest["entries"]}
        self.assertEqual(status, {FOO: "OK", BAR: "OK", BLK: "OK", GONE: "MISSING"})
        self.assertTrue(manifest["complete_run"])
        self.assertTrue(all(dst.startswith("C:") for _, dst in self.camera.copies))
        foo = next(e for e in manifest["entries"] if e["target"] == FOO)
        self.assertEqual(foo["sha256"], sha(self.foo))
        self.assertEqual((self.tmp / "archive/backup" / foo["file"]).read_bytes(), self.foo)

    def test_backup_is_one_shot_and_keeps_first_copy(self):
        self.backup()
        self.write("A", FOO, b"x" * len(self.foo))
        self.boot()
        self.assertEqual((self.drives["C"] / "RCB001.BIN").read_bytes(), self.foo)

    def test_install_readback_and_restore_cycle(self):
        self.backup()
        new = self.tmp / "film.bin"
        new.write_bytes(bytes(reversed(self.foo)))
        stage = self.tmp / "stage-install"
        safety.write_plan("install", self.tmp / "archive", stage, {FOO: new})
        self.insert_card(stage)
        self.boot()
        report = safety.readback_verify(stage / "plan.json", self.drives["C"])
        self.assertTrue(report["permit_consumed"])
        self.assertEqual([e["verdict"] for e in report["entries"]], ["PASS"])
        self.assertEqual(self.read("A", FOO), new.read_bytes())
        self.assertEqual(self.read("A", BAR), self.bar)

        # A second boot must not write again: the permit is spent.
        copies = len(self.camera.copies)
        self.boot()
        self.assertEqual(len(self.camera.copies), copies)

        stage = self.tmp / "stage-restore"
        plan = safety.write_plan("restore", self.tmp / "archive", stage)
        self.assertEqual({e["target"] for e in plan["entries"]}, {FOO, BAR})
        self.insert_card(stage)
        self.boot()
        report = safety.readback_verify(stage / "plan.json", self.drives["C"])
        self.assertEqual({e["verdict"] for e in report["entries"]}, {"PASS"})
        self.assertEqual(self.read("A", FOO), self.foo)

    def test_camera_skips_target_whose_length_changed(self):
        self.backup()
        new = self.tmp / "film.bin"
        new.write_bytes(bytes(len(self.foo)))
        stage = self.tmp / "stage-install"
        safety.write_plan("install", self.tmp / "archive", stage, {FOO: new})
        self.write("A", FOO, self.foo + b"grown")
        self.insert_card(stage)
        self.boot()
        report = safety.readback_verify(stage / "plan.json", self.drives["C"])
        self.assertEqual(report["entries"][0]["verdict"], "NOT-WRITTEN")
        self.assertEqual(self.read("A", FOO), self.foo + b"grown")

    def test_failed_sd_copy_is_never_treated_as_a_backup(self):
        class EmptyCopies(Camera):  # Issue #1: some cards produced empty copies
            def copy(self, src, dst):
                dst.write_bytes(b"")

        self.camera = EmptyCopies(self.drives)
        manifest = self.backup()
        self.assertEqual({e["status"] for e in manifest["entries"]}, {"SIZE", "MISSING"})
        new = self.tmp / "new.bin"
        new.write_bytes(bytes(len(self.foo)))
        with self.assertRaisesRegex(ValueError, "no verified backup"):
            safety.write_plan("install", self.tmp / "archive", self.tmp / "i", {FOO: new})

    def test_write_plan_refusals(self):
        self.backup()
        archive = self.tmp / "archive"
        short = self.tmp / "short.bin"
        short.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "lengths must match"):
            safety.write_plan("install", archive, self.tmp / "a", {FOO: short})
        with self.assertRaisesRegex(ValueError, "only A:"):
            safety.write_plan("install", archive, self.tmp / "b", {BLK: short})
        with self.assertRaisesRegex(ValueError, "no verified backup"):
            safety.write_plan("install", archive, self.tmp / "c", {GONE: short})
        with self.assertRaisesRegex(ValueError, "SD card"):
            safety.backup_plan(["C:\\x.bin"], self.tmp / "d")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            safety.backup_plan(["A:\\it's.bin"], self.tmp / "e")

    def test_tampered_archive_is_rejected(self):
        self.backup()
        manifest = json.loads((self.tmp / "archive/manifest.json").read_text())
        foo = next(e for e in manifest["entries"] if e["target"] == FOO)
        (self.tmp / "archive/backup" / foo["file"]).write_bytes(b"\0" * len(self.foo))
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            safety.write_plan("restore", self.tmp / "archive", self.tmp / "r")


if __name__ == "__main__":
    unittest.main()
