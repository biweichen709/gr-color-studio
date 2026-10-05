import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ricoh_color import codec
from ricoh_color.cli import main
from tests.looks import film
from tests.ttl_sim import Camera

TABLE = "A:\\Resource\\Param\\ic.bin"


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([str(a) for a in argv])
    return out.getvalue()


class CameraWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.drives = {d: self.tmp / d for d in "AC"}
        (self.drives["A"] / "Resource" / "Param").mkdir(parents=True)
        self.drives["C"].mkdir()
        rng = np.random.default_rng(0)
        self.std = codec.TableSpec(grid=17, dtype="u16le", maxval=4095, offset=64)
        self.pos = codec.TableSpec(grid=17, dtype="u16le", maxval=4095, offset=64 + self.std.nbytes)
        data = rng.integers(0, 256, 64 + 2 * self.std.nbytes + 64, dtype=np.uint8).tobytes()
        data = codec.encode(film(17, warm=0.0, sat=1.0), self.std, data)
        self.original = codec.encode(film(17, sat=1.4, contrast=1.15), self.pos, data)
        self.camera_file().write_bytes(self.original)
        self.camera = Camera(self.drives)

    def camera_file(self):
        return self.drives["A"] / "Resource" / "Param" / "ic.bin"

    def boot_with(self, stage):
        shutil.copytree(stage / "card", self.drives["C"], dirs_exist_ok=True)
        self.camera.run((self.drives["C"] / "script" / "startup.ttl").read_text())

    def test_full_cycle(self):
        paths = self.tmp / "paths.txt"
        paths.write_text(f"# colour tables\n{TABLE}\n")
        run("backup-plan", paths, self.tmp / "s1")
        self.boot_with(self.tmp / "s1")
        run("backup-verify", self.tmp / "s1", self.drives["C"], self.tmp / "archive")

        backup = self.tmp / "archive" / "backup" / "RCB001.BIN"
        run("scan", backup, "--grid", "17", "--out", self.tmp / "cands")
        look = self.tmp / "look.cube"
        from ricoh_color.lut import write_cube

        write_cube(look, film(9, warm=0.05, sat=0.8))
        new = self.tmp / "ic-film.bin"
        run("remap", backup, self.tmp / "cands/cand002.json", look,
            "--base", backup, self.tmp / "cands/cand001.json", "-o", new)
        self.assertEqual(len(new.read_bytes()), len(self.original))

        run("install-plan", self.tmp / "archive", self.tmp / "s2", "--replace", f"{TABLE}={new}")
        self.boot_with(self.tmp / "s2")
        report = run("verify-readback", self.tmp / "s2", self.drives["C"])
        self.assertIn('"PASS"', report)
        self.assertEqual(self.camera_file().read_bytes(), new.read_bytes())

        run("restore-plan", self.tmp / "archive", self.tmp / "s3")
        self.boot_with(self.tmp / "s3")
        run("verify-readback", self.tmp / "s3", self.drives["C"])
        self.assertEqual(self.camera_file().read_bytes(), self.original)

    def test_verify_readback_fails_loudly_when_script_did_not_run(self):
        paths = self.tmp / "paths.txt"
        paths.write_text(TABLE + "\n")
        run("backup-plan", paths, self.tmp / "s1")
        self.boot_with(self.tmp / "s1")
        run("backup-verify", self.tmp / "s1", self.drives["C"], self.tmp / "archive")
        new = self.tmp / "new.bin"
        new.write_bytes(bytes(len(self.original)))
        run("install-plan", self.tmp / "archive", self.tmp / "s2", "--replace", f"{TABLE}={new}")
        shutil.copytree(self.tmp / "s2" / "card", self.drives["C"], dirs_exist_ok=True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            run("verify-readback", self.tmp / "s2", self.drives["C"])
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(self.camera_file().read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
