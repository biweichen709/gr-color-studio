import re

from .scan import find_strings

PAYLOAD_START = 0x80
FULL_PATH = re.compile(r"([A-Z]):\\([\x21-\x7e][\x20-\x7e]*)")
PARTIAL_PATH = re.compile(r"(?:^|[^A-Za-z0-9])((?:Resource|Param|Table|Data|IQ|Isp)\\[\x21-\x7e][\x20-\x7e]*)", re.I)
FORMAT_SPEC = re.compile(r"%[-0-9.]*[sdiuxX]")
SKIP_EXT = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".ttf", ".otf", ".fnt", ".wav", ".mp3", ".aac",
    ".mov", ".mp4", ".htm", ".html", ".css", ".js", ".ttl", ".log", ".mod", ".dpof",
}
HINTS = {
    "param": 3, "prm": 3, "table": 3, "tbl": 3, "lut": 5, "3dlut": 5, "color": 4, "colour": 4,
    "imgctrl": 5, "imagectrl": 5, "imagecontrol": 5, "effect": 4, "iq": 3, "isp": 3, "tone": 3,
    "gamma": 4, "matrix": 3, "ccm": 4, "wb": 2, "adj": 2, "calib": 2, "dsp": 2, "pict": 2,
    "style": 3, "film": 4, "positive": 5, "negative": 5, "bleach": 5, "retro": 4, "cross": 3,
    "mono": 2, "hdr": 2, ".bin": 2, ".dat": 2, ".tbl": 4, ".prm": 4, ".lut": 5,
}


def is_container(data):
    return len(data) > PAYLOAD_START and data[8:10] == b"GR"


def unpack(data):
    out = bytearray()
    pos = PAYLOAD_START
    end_of_data = len(data)
    while pos + 2 <= end_of_data:
        prefix = int.from_bytes(data[pos : pos + 2], "big")
        pos += 2
        if prefix == 0:
            break
        end = pos + (prefix & 0x7FFF)
        if end > end_of_data:
            raise ValueError(f"frame at 0x{pos - 2:x} runs past the end of the file")
        if prefix & 0x8000:
            out += data[pos:end]
            pos = end
            continue
        while pos < end:
            flags = int.from_bytes(data[pos : pos + 2], "big")
            pos += 2
            for bit in range(15, -1, -1):
                if pos >= end:
                    break
                if not flags >> bit & 1:
                    out.append(data[pos])
                    pos += 1
                    continue
                hi, lo = data[pos], data[pos + 1]
                pos += 2
                distance = ((hi & 0xF8) << 5) | lo
                count = hi & 7
                if count == 7:
                    while True:
                        extra = data[pos]
                        pos += 1
                        count += extra
                        if extra != 255:
                            break
                if distance == 0:
                    break
                if distance > len(out):
                    raise ValueError(f"invalid back reference at 0x{pos:x}")
                for _ in range(count + 3):
                    out.append(out[-distance])
        pos = end
    return bytes(out)


def _score(path):
    low = path.lower()
    reasons = [word for word in HINTS if word in low]
    return sum(HINTS[w] for w in reasons), reasons


def _extension(path):
    name = path.rsplit("\\", 1)[-1]
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def candidate_paths(payload):
    files, patterns, partial = {}, set(), set()
    for _, _, text in find_strings(payload):
        for m in FULL_PATH.finditer(text):
            drive, rest = m.group(1), m.group(2).rstrip(" .,;:)'\"")
            if drive == "C" or not rest or rest.endswith("\\"):
                continue
            path = f"{drive}:\\{rest}"
            if FORMAT_SPEC.search(path):
                patterns.add(path)
            elif _extension(path) not in SKIP_EXT and "'" not in path and '"' not in path:
                files[path] = _score(path)
        if not FULL_PATH.search(text):
            for m in PARTIAL_PATH.finditer(text):
                partial.add(m.group(1))
    ranked = sorted(files.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return {
        "files": [{"path": p, "score": s, "hints": r} for p, (s, r) in ranked],
        "patterns": sorted(patterns),
        "partial": sorted(partial),
    }


ENTRY_MARKER = b"[OPEN_FACTORY_DEBUG_MENU]\r\n"
ENTRY_KEY = bytes.fromhex("07012c1f10031e16052d")
KNOWN_ENTRIES = {
    "GR IV / HDF / Monochrome 1.11": "00078560.636",
    "GR IIIx Urban / HDF 1.60": "00078490.609",
}
ENTRY_NAME = re.compile(rb"(?<![0-9A-Za-z])(\d{8}\.\d{3})(?![0-9A-Za-z])")
ANCHORS = (b"DEVELOP.MOD", b"OPEN_FACTORY", b"FACTORY", b"Factory")


def find_entry(payload):
    anchors = sorted(m.start() for a in ANCHORS for m in re.finditer(re.escape(a), payload))
    names = {}
    for m in ENTRY_NAME.finditer(payload):
        name = m.group(1).decode()
        near = min((abs(m.start() - a) for a in anchors), default=None)
        if name not in names or (near is not None and (names[name] is None or near < names[name])):
            names[name] = near
    ranked = sorted(names.items(), key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 0, kv[0]))
    return {
        "names": [{"name": n, "distance": d} for n, d in ranked],
        "key_present": ENTRY_KEY in payload,
        "marker_present": b"OPEN_FACTORY_DEBUG_MENU" in payload,
        "develop_mod_present": b"DEVELOP.MOD" in payload,
    }


def entry_files(name):
    if not re.fullmatch(r"\d{8}\.\d{3}", name):
        raise ValueError(f"{name} is not an 8.3 numeric entry name")
    return {name: ENTRY_MARKER, "DEVELOP.MOD": ENTRY_KEY}
