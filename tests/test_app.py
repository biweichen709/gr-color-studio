import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ricoh_color import codec
from tests.looks import film
from tests.ttl_sim import Camera

try:
    from tkinter import messagebox

    HAVE_TK = True
except ImportError:
    HAVE_TK = False

TABLE = "A:\\Resource\\Param\\ic.bin"
HAVE_DISPLAY = os.name == "nt" or bool(os.environ.get("DISPLAY"))


@unittest.skipUnless(HAVE_TK and HAVE_DISPLAY, "needs tkinter and a display")
class CameraWizard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.drives = {d: self.tmp / d for d in "AC"}
        (self.drives["A"] / "Resource" / "Param").mkdir(parents=True)
        self.drives["C"].mkdir()
        std = codec.TableSpec(grid=17, dtype="u16le", maxval=4095, offset=64)
        pos = codec.TableSpec(grid=17, dtype="u16le", maxval=4095, offset=64 + std.nbytes)
        data = np.random.default_rng(0).integers(0, 256, 128 + 2 * std.nbytes, dtype=np.uint8).tobytes()
        data = codec.encode(film(17, warm=0.0, sat=1.0), std, data)
        self.original = codec.encode(film(17, sat=1.4, contrast=1.15), pos, data)
        self.camera_file = self.drives["A"] / "Resource" / "Param" / "ic.bin"
        self.camera_file.write_bytes(self.original)

        self.dialogs = []
        for name in ("showinfo", "showwarning", "showerror"):
            patcher = mock.patch.object(messagebox, name, side_effect=lambda *a, n=name, **k: self.dialogs.append((n, a)))
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("askyesno", "askokcancel"):
            patcher = mock.patch.object(messagebox, name, return_value=True)
            patcher.start()
            self.addCleanup(patcher.stop)

        from ricoh_color import app as appmod

        self.app = appmod.run_gui()()
        self.app.withdraw()
        self.addCleanup(self.app.destroy)
        self.page = self.app.camera_page
        self.page.workspace.set(str(self.tmp / "workspace"))
        self.page.load_state()
        self.page.card_path.set(str(self.drives["C"]))

    def boot_camera(self):
        Camera(self.drives).run((self.drives["C"] / "script" / "startup.ttl").read_text())

    def wait(self):
        for _ in range(1000):
            self.app.update()
            if not self.app.busy:
                return
            time.sleep(0.01)
        self.fail("background task did not finish")

    def test_backup_install_restore_round_trip(self):
        page = self.page
        page.paths.insert("1.0", TABLE)
        page.write_backup()
        self.boot_camera()
        page.verify_backup()

        page.scan_archive()
        self.wait()
        self.app.update()
        self.assertEqual(len(page.data["candidates"]), 2)
        page.cands.selection_set("0")
        page.mark("base")
        page.cands.selection_set("1")
        page.mark("slot")

        self.app.looks_page.tree.selection_set("classic_chrome")
        self.app.update()
        page.write_install()
        self.boot_camera()
        page.verify("install")
        installed = self.camera_file.read_bytes()
        self.assertEqual(len(installed), len(self.original))
        self.assertNotEqual(installed, self.original)
        self.assertEqual(installed[: 64 + 17**3 * 6], self.original[: 64 + 17**3 * 6])

        page.write_restore()
        self.boot_camera()
        page.verify("restore")
        self.assertEqual(self.camera_file.read_bytes(), self.original)

        page.finish()
        self.assertFalse((self.drives["C"] / "script" / "startup.ttl").exists())
        self.assertTrue(list((self.drives["C"] / "RC_OLD").rglob("startup.ttl")))
        self.assertEqual([d for d in self.dialogs if d[0] != "showinfo"], [])

    def test_firmware_button_fills_ranked_backup_paths(self):
        from tests.test_firmware import compressed_frame, container

        strings = b"\0".join([b"A:\\Resource\\Jpeg\\GoodBye.jpg", b"E:\\BlkCtl15.bin", b"A:\\Resource\\Param\\ImgCtrl_Lut.bin"])
        fw = self.tmp / "fwdc248b.bin"
        fw.write_bytes(container(compressed_frame(list(strings))))
        with mock.patch("tkinter.filedialog.askopenfilename", return_value=str(fw)):
            self.page.paths_from_firmware()
            self.wait()
        self.app.update()
        lines = self.page.paths.get("1.0", "end").split()
        self.assertEqual(lines, ["A:\\Resource\\Param\\ImgCtrl_Lut.bin", "E:\\BlkCtl15.bin"])

    def test_entry_found_in_firmware_is_written_to_card(self):
        from ricoh_color import firmware
        from tests.test_firmware import compressed_frame, container

        (self.drives["C"] / "DEVELOP.MOD").write_bytes(b"typed in notepad")
        fw = self.tmp / "gr3.bin"
        fw.write_bytes(container(compressed_frame(list(b"DEVELOP.MOD\x0000077777.123\0"))))
        with mock.patch("tkinter.filedialog.askopenfilename", return_value=str(fw)):
            self.page.find_entry()
            self.wait()
        self.app.update()
        self.assertEqual(self.page.entry_name(), "00077777.123")
        self.page.write_entry()
        self.assertEqual((self.drives["C"] / "00077777.123").read_bytes(), firmware.ENTRY_MARKER)
        self.assertEqual((self.drives["C"] / "DEVELOP.MOD").read_bytes(), firmware.ENTRY_KEY)
        self.assertTrue(list((self.drives["C"] / "RC_OLD").rglob("DEVELOP.MOD")))

    def test_style_page_exports_photo_and_cube(self):
        from PIL import Image

        from ricoh_color import lut, photo

        page = self.app.looks_page
        source = self.tmp / "in.jpg"
        Image.fromarray(np.full((40, 60, 3), 128, np.uint8)).save(source)
        page.photo = photo.load(source)
        page.before = page.photo.preview()
        page.tree.selection_set("velvia")
        page.vars["contrast"].set(20)
        self.app.update()
        page.refresh()
        with mock.patch("tkinter.filedialog.asksaveasfilename", return_value=str(self.tmp / "out.jpg")):
            page.export_photo()
            self.wait()
        with Image.open(self.tmp / "out.jpg") as out:
            self.assertEqual(out.size, (60, 40))
        with mock.patch("tkinter.filedialog.asksaveasfilename", return_value=str(self.tmp / "v.cube")):
            page.export_cube()
        self.assertEqual(lut.read_cube(self.tmp / "v.cube").shape, (65, 65, 65, 3))


if __name__ == "__main__":
    unittest.main()
