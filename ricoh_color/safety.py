import hashlib
import json
import re
import shutil
from pathlib import Path

PROVEN_COMMANDS = {
    "filesearch", "filestat", "filecopy", "filecreate", "filewrite", "fileclose",
    "fileopen", "fileread", "strcompare", "if", "endif", "goto", "exit",
}
PATH_RE = re.compile(r"[A-Z]:\\[\x20-\x7e]+")
MAX_ENTRIES = 999
ARM = "RCARM.TXT"
RUN_MARK = "RCBRUN.TXT"
END_MARK = "RCBEND.TXT"
WRITE_KINDS = {"install": ("RCN", "RCW"), "restore": ("RCB", "RCR")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_path(path, writable=False):
    if not PATH_RE.fullmatch(path) or "'" in path or '"' in path:
        raise ValueError(f"unsupported camera path {path!r}")
    if path.startswith("C:"):
        raise ValueError(f"{path} is on the SD card, not camera storage")
    if writable and not path.startswith("A:\\"):
        raise ValueError(f"refusing to write {path}: only A: resource files may be replaced")
    return path


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{path} must be new or empty; existing output is never overwritten")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _q(name):
    return f"'{name}'"


def _sd(name):
    return f"C:\\{name}"


def _leave_if(cond, target="exit"):
    action = "exit" if target == "exit" else f"goto {target}"
    return [f"if {cond} then", f"    {action}", "endif"]


def backup_script(entries):
    lines = [
        "; ricoh_color backup: reads camera storage, writes only to the SD card.",
        f"filesearch {_q(_sd(RUN_MARK))}",
        *_leave_if("result = 1"),
        f"filecreate fh {_q(_sd(RUN_MARK))}",
        *_leave_if("fh < 0"),
        "filewrite fh 'RUN'",
        "fileclose fh",
    ]
    for e in entries:
        tag, target, copy = e["tag"], _q(e["target"]), _q(_sd(e["copy"]))
        lines += [
            f"; {tag}: {e['target']}",
            "st = 'MISSING'",
            "s1 = -1",
            f"filestat {target} s1",
            *_leave_if("s1 < 0", f"w{tag}"),
            f"filesearch {copy}",
            *_leave_if("result = 1", f"k{tag}"),
            f"filecopy {target} {copy}",
            f":k{tag}",
            "s2 = -1",
            f"filestat {copy} s2",
            "st = 'SIZE'",
            "if s1 = s2 then",
            "    st = 'OK'",
            "endif",
            f":w{tag}",
            f"filecreate fh {_q(_sd(e['status']))}",
            *_leave_if("fh < 0", f"n{tag}"),
            "filewrite fh st",
            "fileclose fh",
            f":n{tag}",
        ]
    lines += [
        f"filecreate fh {_q(_sd(END_MARK))}",
        *_leave_if("fh < 0"),
        "filewrite fh 'END'",
        "fileclose fh",
        "exit",
    ]
    return "\n".join(lines) + "\n"


def backup_plan(targets, stage):
    targets = list(dict.fromkeys(check_path(t.strip()) for t in targets if t.strip()))
    if not 0 < len(targets) <= MAX_ENTRIES:
        raise ValueError(f"need 1..{MAX_ENTRIES} camera paths")
    entries = [
        {"tag": f"{i:03d}", "target": t, "copy": f"RCB{i:03d}.BIN", "status": f"RCB{i:03d}.TXT"}
        for i, t in enumerate(targets, 1)
    ]
    stage = fresh_dir(stage)
    script = stage / "card" / "script" / "startup.ttl"
    script.parent.mkdir(parents=True)
    script.write_text(backup_script(entries))
    plan = {"kind": "backup", "entries": entries}
    (stage / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def backup_verify(plan_path, card, archive):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("kind") != "backup":
        raise ValueError("not a backup plan")
    card = Path(card)
    if not (card / RUN_MARK).exists():
        raise ValueError("RCBRUN.TXT missing: the backup script did not run (Script enabled?)")
    complete = (card / END_MARK).exists()
    archive = fresh_dir(archive)
    (archive / "backup").mkdir()
    results = []
    for e in plan["entries"]:
        status_file, copy = card / e["status"], card / e["copy"]
        status = status_file.read_text().strip() if status_file.exists() else "NO-STATUS"
        row = {"tag": e["tag"], "target": e["target"], "file": e["copy"], "status": status}
        if status == "OK" and not copy.exists():
            row["status"] = "NO-COPY"
        elif status == "OK" and copy.stat().st_size == 0:
            row["status"] = "EMPTY"
        elif status == "OK":
            shutil.copy2(copy, archive / "backup" / e["copy"])
            row.update(size=copy.stat().st_size, sha256=sha256(copy))
        results.append(row)
    manifest = {"kind": "backup-archive", "complete_run": complete, "entries": results}
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_script(entries, kind):
    lines = [
        f"; ricoh_color {kind}: one-shot, length-guarded writes with readback to the SD card.",
        f"fileopen fh {_q(_sd(ARM))} 0",
        *_leave_if("fh < 0"),
        "fileread fh 1 arm",
        "fileclose fh",
        "strcompare arm '1'",
        *_leave_if("result <> 0"),
        f"filecreate fh {_q(_sd(ARM))}",
        *_leave_if("fh < 0"),
        "filewrite fh '0'",
        "fileclose fh",
        f"fileopen fh {_q(_sd(ARM))} 0",
        *_leave_if("fh < 0"),
        "fileread fh 1 arm",
        "fileclose fh",
        "strcompare arm '0'",
        *_leave_if("result <> 0"),
    ]
    for e in entries:
        tag, size = e["tag"], e["size"]
        target, source, readback = _q(e["target"]), _q(_sd(e["source"])), _q(_sd(e["readback"]))
        lines += [
            f"; {tag}: {e['target']} ({size} bytes)",
            f"filesearch {readback}",
            *_leave_if("result = 1", f"n{tag}"),
            "sz = -1",
            f"filestat {source} sz",
            *_leave_if(f"sz <> {size}", f"n{tag}"),
            "sz = -1",
            f"filestat {target} sz",
            *_leave_if(f"sz <> {size}", f"n{tag}"),
            f"filecreate fh {readback}",
            *_leave_if("fh < 0", f"n{tag}"),
            "fileclose fh",
            f"filecopy {source} {target}",
            f"filecopy {target} {readback}",
            f":n{tag}",
        ]
    lines.append("exit")
    return "\n".join(lines) + "\n"


def _load_archive(archive):
    archive = Path(archive)
    manifest = json.loads((archive / "manifest.json").read_text())
    if manifest.get("kind") != "backup-archive":
        raise ValueError("not a backup archive")
    by_target = {}
    for e in manifest["entries"]:
        if e["status"] != "OK":
            continue
        stored = archive / "backup" / e["file"]
        if sha256(stored) != e["sha256"]:
            raise ValueError(f"archived backup {stored} no longer matches its recorded SHA-256")
        by_target[e["target"]] = dict(e, path=stored)
    return by_target


def write_plan(kind, archive, stage, replacements=None, only=None):
    source_prefix, readback_prefix = WRITE_KINDS[kind]
    backups = _load_archive(archive)
    if kind == "install":
        wanted = {check_path(t, writable=True): Path(p) for t, p in replacements.items()}
    else:
        targets = only or [t for t in backups if t.startswith("A:\\")]
        wanted = {check_path(t, writable=True): None for t in targets}
    if not wanted:
        raise ValueError("nothing to write")

    entries, files = [], {}
    for target, new_file in wanted.items():
        backup = backups.get(target)
        if backup is None:
            raise ValueError(f"{target} has no verified backup in {archive}; back it up first")
        tag = backup["tag"]
        source = new_file or backup["path"]
        size = source.stat().st_size
        if size != backup["size"]:
            raise ValueError(
                f"{source} is {size} bytes but {target} is {backup['size']}; lengths must match"
            )
        source_name = f"{source_prefix}{tag}.BIN"
        files[source_name] = source
        entries.append({
            "tag": tag, "target": target, "source": source_name,
            "readback": f"{readback_prefix}{tag}.BIN", "size": size, "sha256": sha256(source),
        })

    stage = fresh_dir(stage)
    card = stage / "card"
    (card / "script").mkdir(parents=True)
    (card / "script" / "startup.ttl").write_text(write_script(entries, kind))
    (card / ARM).write_text("1")
    for name, src in files.items():
        shutil.copyfile(src, card / name)
    plan = {"kind": kind, "entries": entries}
    (stage / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def readback_verify(plan_path, card):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("kind") not in WRITE_KINDS:
        raise ValueError("not an install/restore plan")
    card = Path(card)
    arm = (card / ARM).read_text().strip() if (card / ARM).exists() else ""
    results = []
    for e in plan["entries"]:
        readback = card / e["readback"]
        if not readback.exists():
            verdict = "NOT-WRITTEN"
        elif readback.stat().st_size == 0:
            verdict = "EMPTY"
        elif readback.stat().st_size != e["size"] or sha256(readback) != e["sha256"]:
            verdict = "MISMATCH"
        else:
            verdict = "PASS"
        results.append({"tag": e["tag"], "target": e["target"], "verdict": verdict})
    return {"kind": plan["kind"], "permit_consumed": arm == "0", "entries": results}
