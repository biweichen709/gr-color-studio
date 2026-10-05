import unittest

from ricoh_color import firmware


def compressed_frame(items):
    body = bytearray()
    for start in range(0, len(items), 16):
        group = items[start : start + 16]
        flags = 0
        chunk = bytearray()
        for bit, item in zip(range(15, -1, -1), group):
            if isinstance(item, int):
                chunk.append(item)
                continue
            flags |= 1 << bit
            distance, length = item
            count = length - 3
            chunk += bytes([((distance >> 8) << 3) | min(count, 7), distance & 0xFF])
            if count >= 7:
                rest = count - 7
                while rest >= 255:
                    chunk.append(255)
                    rest -= 255
                chunk.append(rest)
        body += flags.to_bytes(2, "big") + chunk
    return len(body).to_bytes(2, "big") + bytes(body)


def container(*frames):
    header = bytearray(firmware.PAYLOAD_START)
    header[8:13] = b"GR IV"
    return bytes(header) + b"".join(frames) + b"\0\0"


class Unpack(unittest.TestCase):
    def test_literals_references_and_raw_frames(self):
        items = list(b"abcdef") + [(6, 12)] + list(b"XYZ") + [(3, 300)] + [(1, 4)]
        raw = b"raw-frame-bytes"
        data = container(compressed_frame(items), (0x8000 | len(raw)).to_bytes(2, "big") + raw)
        expected = b"abcdef" + b"abcdefabcdef" + b"XYZ" + b"XYZ" * 100 + b"ZZZZ" + raw
        self.assertTrue(firmware.is_container(data))
        self.assertEqual(firmware.unpack(data), expected)

    def test_bad_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "back reference"):
            firmware.unpack(container(compressed_frame([(5, 3)])))


class Paths(unittest.TestCase):
    def test_paths_are_filtered_and_ranked(self):
        payload = b"\0".join([
            b"A:\\Resource\\Jpeg\\GoodBye.jpg",
            b"A:\\Resource\\Param\\ImgCtrl_Lut.bin",
            b"A:\\Resource\\Font\\main.ttf",
            b"open failed: E:\\BlkCtl15.bin",
            b"C:\\script\\startup.ttl",
            "A:\\Resource\\Param\\Tone.dat".encode("utf-16-le"),
            b"A:\\Resource\\Param\\IC%02d.bin",
            b"Resource\\Table\\gamma.tbl",
        ])
        found = firmware.candidate_paths(payload)
        paths = [f["path"] for f in found["files"]]
        self.assertEqual(paths[0], "A:\\Resource\\Param\\ImgCtrl_Lut.bin")
        self.assertIn("A:\\Resource\\Param\\Tone.dat", paths)
        self.assertIn("E:\\BlkCtl15.bin", paths)
        self.assertNotIn("A:\\Resource\\Jpeg\\GoodBye.jpg", paths)
        self.assertFalse(any(p.startswith("C:") or p.endswith(".ttf") for p in paths))
        self.assertEqual(found["patterns"], ["A:\\Resource\\Param\\IC%02d.bin"])
        self.assertEqual(found["partial"], ["Resource\\Table\\gamma.tbl"])


class Entry(unittest.TestCase):
    def test_entry_name_near_factory_strings_ranks_first(self):
        payload = (b"IMG_0001.JPG\0version 00012345.678\0" + bytes(5000)
                   + b"DEVELOP.MOD\0" + firmware.ENTRY_KEY + b"\x0000099999.111\0[OPEN_FACTORY_DEBUG_MENU]")
        found = firmware.find_entry(payload)
        self.assertEqual([n["name"] for n in found["names"]], ["00099999.111", "00012345.678"])
        self.assertTrue(found["key_present"] and found["marker_present"] and found["develop_mod_present"])

    def test_entry_files_are_exact(self):
        files = firmware.entry_files("00078560.636")
        self.assertEqual(files["DEVELOP.MOD"], bytes.fromhex("07 01 2c 1f 10 03 1e 16 05 2d"))
        self.assertEqual(files["00078560.636"], b"[OPEN_FACTORY_DEBUG_MENU]\r\n")
        with self.assertRaises(ValueError):
            firmware.entry_files("DEVELOP.MOD")


    def test_utf16_names_formats_and_context(self):
        payload = (bytes(100) + "DEVELOP.MOD".encode("utf-16-le") + b"\0\0"
                   + "00055555.222".encode("utf-16-le") + b"\0\0"
                   + b"name %08d.%03d\0" + firmware.ENTRY_KEY + b"\0[OPEN_FACTORY_DEBUG_MENU]\0" + bytes(100))
        found = firmware.find_entry(payload)
        self.assertEqual(found["names"][0]["name"], "00055555.222")
        self.assertEqual(found["names"][0]["encoding"], "utf-16")
        self.assertTrue(found["develop_mod_present"])
        self.assertEqual(found["formats"], ["name %08d.%03d"])
        self.assertIn("DEVELOP.MOD", [c["text"] for c in found["context"]])


if __name__ == "__main__":
    unittest.main()
