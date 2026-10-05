import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import codec, color, firmware, fit, lut, profile, remap, safety, scan


def _print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_strings(args):
    data = Path(args.file).read_bytes()
    for hit in scan.interesting_strings(data, extra=args.keyword):
        if args.paths_only and not hit["path"]:
            continue
        tags = ",".join(hit["keywords"]) or "path"
        print(f"0x{hit['offset']:08x}  {hit['encoding']:9}  [{tags}]  {hit['text']}")


def cmd_scan(args):
    data = Path(args.file).read_bytes()
    luts = scan.find_luts(data, grids=tuple(args.grid), dtypes=tuple(args.dtype))
    curves = scan.find_curves(data) if args.curves else []
    for i, hit in enumerate(luts, 1):
        s = hit["spec"]
        print(
            f"#{i:03d} 0x{hit['offset']:08x}  {s['grid']}^3 {s['dtype']} {s['layout']}"
            f"{'+pad' if s['pad'] else ''} fastest={s['fastest']} channels={s['channels']}"
            f" maxval={s['maxval']}  rough={hit['rough']} structure={hit['structure']}"
            f" {hit['note']}"
        )
    for hit in curves:
        print(
            f"curve 0x{hit['offset']:08x}  {hit['dtype']} x{hit['length']}  "
            f"{hit['first']:.0f}->{hit['last']:.0f}{'  (linear)' if hit['linear'] else ''}"
        )
    if not luts and not curves:
        print("no candidates found")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for i, hit in enumerate(luts, 1):
            spec = codec.TableSpec(**hit["spec"])
            spec.save(out / f"cand{i:03d}.json")
            lut.write_cube(out / f"cand{i:03d}.cube", codec.decode(data, spec), title=f"cand{i:03d}")
        (out / "candidates.json").write_text(json.dumps({"luts": luts, "curves": curves}, indent=2))
        print(f"wrote {len(luts)} spec/.cube pairs and candidates.json to {out}")


def cmd_extract(args):
    data = Path(args.file).read_bytes()
    lut.write_cube(args.out, codec.decode(data, codec.TableSpec.load(args.spec)))
    print(f"wrote {args.out}")


def cmd_remap(args):
    into = Path(args.into).read_bytes()
    into_spec = codec.TableSpec.load(args.into_spec)
    if args.base:
        base = codec.decode(Path(args.base[0]).read_bytes(), codec.TableSpec.load(args.base[1]))
        if base.shape[0] != into_spec.grid:
            base = lut.resample(base, into_spec.grid)
    else:
        base = codec.decode(into, into_spec)
    new = remap.remap(base, lut.read_cube(args.look), space=args.space, strength=args.strength)
    out = codec.encode(new, into_spec, into)
    if len(out) != len(into):
        raise SystemExit("internal error: output length changed")
    Path(args.output).write_bytes(out)
    old = codec.decode(into, into_spec)
    changed = sum(a != b for a, b in zip(out, into))
    print(f"wrote {args.output}: {len(out)} bytes (same as input), {changed} bytes differ")
    print(f"node change vs. slot's original values: mean {np.abs(new - old).mean():.4f}, max {np.abs(new - old).max():.4f}")


def cmd_fit(args):
    srcs, dsts = [], []
    for src, dst in args.pair:
        a = lut.load_image(src, args.downscale)
        b = lut.load_image(dst, args.downscale)
        if a.shape != b.shape:
            raise SystemExit(f"{src} and {dst} differ in size; pairs must be pixel-aligned")
        srcs.append(a.reshape(-1, 3))
        dsts.append(b.reshape(-1, 3))
    src, dst = np.concatenate(srcs), np.concatenate(dsts)
    result, info = fit.fit_lut(src, dst, grid=args.grid, smooth=args.smooth)
    lut.write_cube(args.output, result, title=Path(args.output).stem)
    sample = np.random.default_rng(1).choice(len(src), min(len(src), 200_000), replace=False)
    info["delta_e2000"] = color.delta_e_stats(lut.apply(result, src[sample]), dst[sample])
    _print_json(info)
    print(f"wrote {args.output}")


def cmd_apply(args):
    lut.save_image(args.output, lut.apply(lut.read_cube(args.lut), lut.load_image(args.image)))
    print(f"wrote {args.output}")


def cmd_compare(args):
    a = lut.load_image(args.a, args.downscale)
    b = lut.load_image(args.b, args.downscale)
    if a.shape != b.shape:
        raise SystemExit("images differ in size")
    _print_json(color.delta_e_stats(a, b))


def cmd_hald(args):
    from PIL import Image

    Image.fromarray(lut.hald_identity(args.level), "RGB").save(args.output)
    print(f"wrote {args.output}: grade it with your preset (no grain/vignette/sharpening), then hald-to-cube")


def cmd_hald_to_cube(args):
    lut.write_cube(args.output, lut.hald_to_lut(lut.load_image(args.image)), title=Path(args.output).stem)
    print(f"wrote {args.output}")


def _report(result, ok_values):
    _print_json(result)
    bad = [e for e in result["entries"] if e.get("status", e.get("verdict")) not in ok_values]
    if bad:
        print(f"{len(bad)} entr{'y' if len(bad) == 1 else 'ies'} need attention", file=sys.stderr)
        raise SystemExit(1)


def cmd_backup_plan(args):
    targets = [
        line.strip() for line in Path(args.paths).read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    plan = safety.backup_plan(targets, args.stage)
    print(f"staged {len(plan['entries'])} paths. Copy {Path(args.stage) / 'card'}/ onto the SD card root.")


def cmd_backup_verify(args):
    result = safety.backup_verify(Path(args.stage) / "plan.json", args.card, args.archive)
    if not result["complete_run"]:
        print("warning: RCBEND.TXT missing, the run did not finish", file=sys.stderr)
    _report(result, {"OK", "MISSING"})


def cmd_install_plan(args):
    replacements = {}
    for item in args.replace:
        target, _, path = item.partition("=")
        replacements[target] = path
    plan = safety.write_plan("install", args.archive, args.stage, replacements=replacements)
    print(f"staged {len(plan['entries'])} replacement(s). Copy {Path(args.stage) / 'card'}/ onto the SD card root.")


def cmd_restore_plan(args):
    plan = safety.write_plan("restore", args.archive, args.stage, only=args.only)
    print(f"staged restore of {len(plan['entries'])} file(s). Copy {Path(args.stage) / 'card'}/ onto the SD card root.")


def cmd_verify_readback(args):
    result = safety.readback_verify(Path(args.stage) / "plan.json", args.card)
    if not result["permit_consumed"]:
        print("warning: RCARM.TXT was not consumed; the script probably did not run", file=sys.stderr)
    _report(result, {"PASS"})


def cmd_firmware_paths(args):
    data = Path(args.file).read_bytes()
    payload = firmware.unpack(data) if firmware.is_container(data) else data
    found = firmware.candidate_paths(payload)
    for f in found["files"]:
        print(f"{f['score']:3d}  {f['path']}  {','.join(f['hints'])}")
    for p in found["patterns"]:
        print(f"pattern  {p}")
    if args.output:
        Path(args.output).write_text("\n".join(f["path"] for f in found["files"]) + "\n", encoding="utf-8")
        print(f"wrote {len(found['files'])} paths to {args.output}")


def cmd_factory_entry(args):
    data = Path(args.file).read_bytes()
    found = firmware.find_entry(firmware.unpack(data) if firmware.is_container(data) else data)
    print(f"DEVELOP.MOD string: {found['develop_mod_present']}  marker: {found['marker_present']}  known key bytes: {found['key_present']}")
    for n in found["names"]:
        print(f"{n['name']}  [{n['encoding']}]  distance={n['distance']}")
    for f in found["formats"]:
        print(f"format: {f}")
    print("key offsets:", ", ".join(f"0x{o:x}" for o in found["key_offsets"]))
    for ctx in found["context"]:
        print(f"{ctx['relative']:+6d} [{ctx['encoding']}] {ctx['text'][:100]}")
    if args.write:
        name = args.name or (found["names"][0]["name"] if found["names"] else None)
        if not name:
            raise SystemExit("no entry name found; pass --name")
        Path(args.write).mkdir(parents=True, exist_ok=True)
        for f, data in firmware.entry_files(name).items():
            (Path(args.write) / f).write_bytes(data)
        print(f"wrote {name} and DEVELOP.MOD to {args.write}")


def cmd_profile(args):
    data = Path(args.file).read_bytes()
    payload = firmware.unpack(data) if firmware.is_container(data) else data
    found = profile.profile_scan(payload, whole_file=args.whole)
    print(f"mode-name anchors: {len(found['anchors'])}; clusters: {len(found['clusters'])}")
    for c in found["clusters"]:
        print(f"  cluster @0x{c['lo']:x}: {', '.join(c['modes'])}")
    for h in found["ccm"]:
        near = "firmware" if h["anchor_distance"] is None else f"{h['anchor_distance']}B from a mode name"
        print(f"CCM  @0x{h['offset']:x}  {h['dtype']}/scale {h['scale']}  x{h['count']}  ({near})  row0 {h['matrix0'][0]}")
    for h in found["curves"]:
        print(f"CURVE@0x{h['offset']:x}  {h['dtype']} len {h['length']} x{h['count']} {h['role']}"
              + ("  [near calibration]" if h["near_calibration"] else ""))
    for h in found["mode_blocks"]:
        print(f"BLOCK@0x{h['offset']:x}  stride {h['stride']} x{h['count']}  const {h['constant_frac']} vary {h['varying_frac']}")
    if not (found["ccm"] or found["curves"] or found["mode_blocks"]):
        print("no matrix/curve/parameter-block candidates found")
    if args.out:
        Path(args.out).write_text(json.dumps(found, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")


def cmd_diff_backups(args):
    result = profile.diff_archives(args.archive_a, args.archive_b)
    if result["changed"]:
        print("files that CHANGED between the two backups (where that look is stored):")
        for t in result["changed"]:
            print(f"  {t}")
    else:
        print("no backed-up file changed between the two modes.")
        print("=> the look is not in any captured file; it is compiled into firmware (not SD-replaceable).")
    print(f"unchanged: {result['unchanged_count']}; only in one: {len(result['only_in_one'])}")


def build_parser():
    p = argparse.ArgumentParser(prog="ricoh_color", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("strings", help="list resource paths and colour-related strings in a binary")
    s.add_argument("file")
    s.add_argument("--keyword", action="append", default=[], help="extra keyword (repeatable)")
    s.add_argument("--paths-only", action="store_true")
    s.set_defaults(func=cmd_strings)

    s = sub.add_parser("scan", help="find candidate 3D colour tables and tone curves")
    s.add_argument("file")
    s.add_argument("--grid", type=int, action="append", help="grid size (repeatable)")
    s.add_argument("--dtype", action="append", choices=list(codec.DTYPES), help="sample type (repeatable)")
    s.add_argument("--curves", action="store_true", help="also list 1D curves")
    s.add_argument("--out", help="write a spec .json and preview .cube per candidate here")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("extract", help="decode a table to .cube using a spec")
    s.add_argument("file")
    s.add_argument("spec")
    s.add_argument("out")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("remap", help="apply a look to every node of a camera table (same file length)")
    s.add_argument("into", help="file whose table slot is rewritten")
    s.add_argument("into_spec")
    s.add_argument("look", help=".cube look defined on the camera's display output")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--base", nargs=2, metavar=("FILE", "SPEC"), help="take node values from another table (e.g. Standard)")
    s.add_argument("--space", choices=list(remap.SPACES), default="rgb", help="what the table's output values are")
    s.add_argument("--strength", type=float, default=1.0)
    s.set_defaults(func=cmd_remap)

    s = sub.add_parser("fit", help="fit a .cube from pixel-aligned image pairs")
    s.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SRC", "DST"))
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--grid", type=int, default=33)
    s.add_argument("--smooth", type=float, default=1.0)
    s.add_argument("--downscale", type=int, default=2, help="box-average N x N pixels first")
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("apply", help="preview a .cube on an image")
    s.add_argument("lut")
    s.add_argument("image")
    s.add_argument("output")
    s.set_defaults(func=cmd_apply)

    s = sub.add_parser("compare", help="CIEDE2000 statistics between two aligned images")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--downscale", type=int, default=1)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("hald", help="write an identity Hald CLUT image to grade with a preset")
    s.add_argument("--level", type=int, default=8, help="8 -> 512x512 image, 64^3 cube")
    s.add_argument("output")
    s.set_defaults(func=cmd_hald)

    s = sub.add_parser("hald-to-cube", help="convert a graded Hald image to .cube")
    s.add_argument("image")
    s.add_argument("output")
    s.set_defaults(func=cmd_hald_to_cube)

    s = sub.add_parser("factory-entry", help="find factory-menu entry file names in a firmware file")
    s.add_argument("file")
    s.add_argument("--write", metavar="DIR", help="write the entry files into DIR (e.g. the SD card root)")
    s.add_argument("--name", help="entry name to write instead of the best candidate")
    s.set_defaults(func=cmd_factory_entry)

    s = sub.add_parser("profile", help="find colour matrices / tone-curve banks / mode blocks (not just 3D LUTs)")
    s.add_argument("file", help="firmware, decoded payload, or a backed-up .bin")
    s.add_argument("--whole", action="store_true", help="scan the whole file, not only mode-name windows")
    s.add_argument("-o", "--output", dest="out", help="write full JSON results here")
    s.set_defaults(func=cmd_profile)

    s = sub.add_parser("diff-backups", help="byte-compare two backup archives (e.g. Standard vs Vivid)")
    s.add_argument("archive_a")
    s.add_argument("archive_b")
    s.set_defaults(func=cmd_diff_backups)

    s = sub.add_parser("firmware-paths", help="list camera file paths referenced by a firmware file, ranked")
    s.add_argument("file", help="official firmware (e.g. fwdc248b.bin) or a decoded payload")
    s.add_argument("-o", "--output", help="write the paths, one per line, for backup-plan")
    s.set_defaults(func=cmd_firmware_paths)

    s = sub.add_parser("backup-plan", help="stage a read-only backup script for camera paths")
    s.add_argument("paths", help="text file, one camera path per line (A:\\..., E:\\...)")
    s.add_argument("stage")
    s.set_defaults(func=cmd_backup_plan)

    s = sub.add_parser("backup-verify", help="check the card after a backup and archive the originals")
    s.add_argument("stage")
    s.add_argument("card")
    s.add_argument("archive", help="new directory; keep it safe")
    s.set_defaults(func=cmd_backup_verify)

    s = sub.add_parser("install-plan", help="stage a one-shot, length-guarded replacement")
    s.add_argument("archive")
    s.add_argument("stage")
    s.add_argument("--replace", action="append", required=True, metavar="CAMERA_PATH=FILE")
    s.set_defaults(func=cmd_install_plan)

    s = sub.add_parser("restore-plan", help="stage a one-shot restore of archived originals")
    s.add_argument("archive")
    s.add_argument("stage")
    s.add_argument("--only", action="append", metavar="CAMERA_PATH")
    s.set_defaults(func=cmd_restore_plan)

    s = sub.add_parser("verify-readback", help="compare readbacks after install/restore")
    s.add_argument("stage")
    s.add_argument("card")
    s.set_defaults(func=cmd_verify_readback)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        args.grid = args.grid or list(scan.GRIDS)
        args.dtype = args.dtype or list(scan.SCAN_DTYPES)
    try:
        args.func(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
