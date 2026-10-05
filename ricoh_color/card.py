import datetime
import json
import os
import shutil
from pathlib import Path

from . import safety


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def filesystem(root):
    if os.name != "nt":
        return None
    import ctypes

    drive = os.path.splitdrive(os.path.abspath(root))[0] + "\\"
    name = ctypes.create_unicode_buffer(64)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive), None, 0, None, None, None, name, len(name)
    )
    return name.value if ok else None


def has_backup_run(card):
    return (Path(card) / safety.RUN_MARK).exists()


def _stale_names(plan):
    if plan["kind"] == "backup":
        names = [safety.RUN_MARK, safety.END_MARK]
        for e in plan["entries"]:
            names += [e["copy"], e["status"]]
        return names
    return [e["readback"] for e in plan["entries"]]


def _move_aside(path, card, stamp, moved):
    target = card / "RC_OLD" / stamp / path.relative_to(card)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    moved.append(str(target.relative_to(card)))


def stage_to_card(stage, card, stamp=None):
    stage, card = Path(stage), Path(card)
    plan = json.loads((stage / "plan.json").read_text())
    stamp = stamp or _timestamp()
    moved = []
    for name in _stale_names(plan):
        if (card / name).exists():
            _move_aside(card / name, card, stamp, moved)
    source_root = stage / "card"
    for src in sorted(p for p in source_root.rglob("*") if p.is_file()):
        dst = card / src.relative_to(source_root)
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            _move_aside(dst, card, stamp, moved)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return moved


def finish(card, stamp=None):
    card = Path(card)
    script = card / "script" / "startup.ttl"
    if not script.exists():
        return None
    moved = []
    _move_aside(script, card, stamp or _timestamp(), moved)
    return moved[0]


def write_files(card, files, stamp=None):
    card = Path(card)
    stamp = stamp or _timestamp()
    moved = []
    for name, data in files.items():
        dst = card / name
        if dst.exists() and dst.read_bytes() != data:
            _move_aside(dst, card, stamp, moved)
        dst.write_bytes(data)
    return moved


def remove_sweep(card, model_id, marker):
    removed = 0
    for path in Path(card).glob(f"{model_id:08d}.*"):
        if len(path.suffix) == 4 and path.suffix[1:].isdigit() and path.read_bytes() == marker:
            path.unlink()
            removed += 1
    return removed
