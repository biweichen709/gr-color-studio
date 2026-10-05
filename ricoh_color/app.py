import datetime
import json
import os
import queue
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np

from . import card, codec, looks, lut, photo, remap, safety, scan

APP_NAME = "GR 色彩工坊"
CHECKER = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
    (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
]
SLIDERS = [
    ("strength", "强度", 0, 100, 100),
    ("temp", "色温", -50, 50, 0),
    ("tint", "色调", -50, 50, 0),
    ("contrast", "对比", -50, 50, 0),
    ("saturation", "饱和", -50, 50, 0),
    ("fade", "褪色", 0, 50, 0),
]
HELP = """\
【风格调色】
1. 左侧选择风格：哈苏 HNCS 风格、富士胶片模拟风格、黑白。也可以「导入 .cube」使用你自己的 LUT。
2. 「打开照片」支持 JPG / PNG / TIFF，以及 GR 的 DNG 等 RAW 文件。没打开照片时显示内置色卡。
3. 用「强度 / 色温 / 色调 / 对比 / 饱和 / 褪色」微调，上方可切换「效果 / 原图 / 左右对比」。
4. 「导出照片」输出高质量 JPEG 并保留 EXIF；「批量处理」处理整个文件夹；
   「导出 .cube」得到 LUT，可用于 Photoshop、DaVinci Resolve、Final Cut、手机修图 App 等。

预设是依据公开描述手工调校的近似风格，不含哈苏、富士或理光的任何数据，与这些公司无关。
JPEG 按 sRGB 处理；颗粒、暗角等空间效果不在 LUT 中。

【写入相机（实验）】
把当前风格写进 GR IV 机内的某个影像风格。流程基于 radium-wang/ricoh-gr4-firmware-analysis-and-
feature-expansion 的实机研究：工厂菜单开启 Script，SD 卡启动脚本可以读写 A: 资源盘。
本工具不刷固件，只做文件级「备份 → 替换 → 读回核对 → 随时恢复」。

重要：
· 生成的相机脚本只在电脑端模拟验证过，尚未在真机上跑过。第一次只做「1. 备份」。
· 理光色彩表在哪个文件、是什么格式目前未知，「2. 扫描」只给候选，需要实拍确认。
· 只允许写 A: 盘、只允许写已备份过的文件、新文件必须与原文件等长，并且每次写入后读回核对。
· 使用 FAT32 格式的卡，电量充足；中途断电后先「校验」，不要反复重试。
"""


def hsv_to_rgb(h, s, v):
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    choices = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    out = np.zeros(h.shape + (3,))
    for k, (r, g, b) in enumerate(choices):
        mask = i == k
        out[mask] = np.stack([np.broadcast_to(c, h.shape)[mask] for c in (r, g, b)], axis=-1)
    return out


def sample_image(width=960, height=640):
    img = np.zeros((height, width, 3))
    pw, ph, band = width // 6, height * 10 // 64, height * 12 // 64
    for k, c in enumerate(CHECKER):
        row, col = divmod(k, 6)
        img[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw] = np.array(c) / 255
    y0 = 4 * ph
    hue = np.linspace(0, 1, width, endpoint=False)[None, :].repeat(band, 0)
    sat = np.linspace(0.15, 0.85, band)[:, None].repeat(width, 1)
    img[y0 : y0 + band] = hsv_to_rgb(hue, sat, np.full_like(hue, 0.9))
    img[y0 + band :] = np.linspace(0, 1, width)[None, :, None]
    return np.rint(img * 255).astype(np.uint8)


def table_for(source, adjust, grid=33):
    if isinstance(source, looks.Look):
        return looks.bake(source, adjust, grid)
    return looks.bake_lut(source, adjust, grid)


def default_workspace():
    return Path.home() / "Documents" / "GR色彩工坊"


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    class StylePage(ttk.Frame):
        def __init__(self, app, master):
            super().__init__(master, padding=8)
            self.app = app
            self.sources = {}
            self.source_names = {}
            self.photo = None
            self.before = sample_image()
            self.graded = self.before
            self.vars = {}
            self.pending = None
            self.cancel = threading.Event()
            self._image = None
            self.mode = tk.StringVar(value="split")
            self.grid_size = tk.StringVar(value="65")
            self.columnconfigure(1, weight=1)
            self.rowconfigure(0, weight=1)
            self._build_left()
            self._build_right()
            first = looks.PRESETS[0].key
            self.tree.selection_set(first)
            self.tree.see(first)

        def _build_left(self):
            left = ttk.Frame(self)
            left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
            left.rowconfigure(0, weight=1)
            self.tree = ttk.Treeview(left, show="tree", selectmode="browse", height=18)
            self.tree.column("#0", width=250)
            self.tree.grid(row=0, column=0, columnspan=3, sticky="nsew")
            groups = {}
            for look in looks.PRESETS:
                if look.group not in groups:
                    groups[look.group] = self.tree.insert("", "end", text=look.group, open=True)
                self.tree.insert(groups[look.group], "end", iid=look.key, text=look.name)
                self.sources[look.key] = look
                self.source_names[look.key] = f"{look.group} · {look.name}"
            self.cube_group = None
            self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
            self.note = ttk.Label(left, wraplength=250, foreground="#555")
            self.note.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 6))

            for row, (key, label, lo, hi, default) in enumerate(SLIDERS, start=2):
                ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
                var = tk.IntVar(value=default)
                self.vars[key] = var
                scale = ttk.Scale(left, from_=lo, to=hi, variable=var, length=170,
                                  command=lambda v, var=var: (var.set(round(float(v))), self.schedule()))
                scale.grid(row=row, column=1, sticky="ew")
                ttk.Label(left, textvariable=var, width=4, anchor="e").grid(row=row, column=2, sticky="e")
            buttons = ttk.Frame(left)
            buttons.grid(row=20, column=0, columnspan=3, sticky="ew", pady=(8, 0))
            buttons.columnconfigure((0, 1), weight=1)
            items = [
                ("重置微调", self.reset), ("打开照片…", self.open_photo),
                ("导出照片…", self.export_photo), ("批量处理文件夹…", self.batch),
                ("导出 .cube…", self.export_cube), ("导入 .cube…", self.import_cube),
            ]
            for k, (text, cmd) in enumerate(items):
                ttk.Button(buttons, text=text, command=cmd).grid(row=k // 2, column=k % 2, sticky="ew", padx=2, pady=2)
            size = ttk.Frame(left)
            size.grid(row=21, column=0, columnspan=3, sticky="w", pady=(4, 0))
            ttk.Label(size, text="导出 .cube 精度").pack(side="left")
            ttk.Combobox(size, textvariable=self.grid_size, values=("33", "65"), width=5, state="readonly").pack(side="left", padx=4)

        def _build_right(self):
            right = ttk.Frame(self)
            right.grid(row=0, column=1, sticky="nsew")
            right.rowconfigure(1, weight=1)
            right.columnconfigure(0, weight=1)
            bar = ttk.Frame(right)
            bar.grid(row=0, column=0, sticky="ew")
            for text, value in (("效果", "after"), ("原图", "before"), ("左右对比", "split")):
                ttk.Radiobutton(bar, text=text, value=value, variable=self.mode, command=self.show).pack(side="left", padx=4)
            self.caption = ttk.Label(bar, text="内置色卡（打开照片后显示照片）", foreground="#555")
            self.caption.pack(side="right")
            self.canvas = tk.Canvas(right, background="#2b2b2b", highlightthickness=0)
            self.canvas.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
            self.canvas.bind("<Configure>", lambda e: self.show())


        def source(self):
            selection = self.tree.selection()
            key = selection[0] if selection else looks.PRESETS[0].key
            return self.sources.get(key, looks.PRESETS[0]), self.source_names.get(key, "")

        def adjust(self):
            v = {k: var.get() for k, var in self.vars.items()}
            return looks.Adjust(
                strength=v["strength"] / 100, temp=v["temp"] / 100, tint=v["tint"] / 100,
                contrast=v["contrast"] / 50, saturation=1 + v["saturation"] / 100, fade=v["fade"] / 50,
            )

        def table(self, grid=33):
            return table_for(self.source()[0], self.adjust(), grid)

        def look_label(self):
            name = self.source()[1]
            return name if self.adjust().is_neutral() else f"{name}（已微调）"


        def _on_select(self):
            selection = self.tree.selection()
            if not selection or selection[0] not in self.sources:
                return
            src = self.sources[selection[0]]
            self.note.configure(text=src.note if isinstance(src, looks.Look) else "导入的 LUT；微调滑块作用在它之后。")
            self.schedule()

        def schedule(self):
            if self.pending:
                self.after_cancel(self.pending)
            self.pending = self.after(90, self.refresh)

        def refresh(self):
            self.pending = None
            self.graded = photo.apply_lut(self.table(33), self.before)
            self.show()

        def show(self):
            w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
            mode = self.mode.get()
            if mode == "before":
                img = self.before
            elif mode == "after":
                img = self.graded
            else:
                img = self.graded.copy()
                half = img.shape[1] // 2
                img[:, :half] = self.before[:, :half]
                img[:, half : half + 2] = 255
            im = Image.fromarray(img)
            im.thumbnail((w, h), Image.LANCZOS)
            self._image = ImageTk.PhotoImage(im)
            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, image=self._image)
            if mode == "split":
                self.canvas.create_text(w // 2 - 8, 14, text="原图", fill="white", anchor="e")
                self.canvas.create_text(w // 2 + 8, 14, text="效果", fill="white", anchor="w")

        def reset(self):
            for key, _, _, _, default in SLIDERS:
                self.vars[key].set(default)
            self.schedule()


        def open_photo(self):
            exts = " ".join(f"*{e} *{e.upper()}" for e in sorted(photo.IMAGE_EXTENSIONS | photo.RAW_EXTENSIONS))
            path = filedialog.askopenfilename(title="打开照片", filetypes=[("照片", exts), ("所有文件", "*.*")])
            if not path:
                return

            def work():
                return photo.load(path)

            def done(p):
                self.photo = p
                self.before = p.preview(1400)
                self.caption.configure(text=f"{Path(path).name}  {p.pixels.shape[1]}×{p.pixels.shape[0]}")
                self.refresh()

            self.app.run_task(f"正在读取 {Path(path).name}…", work, done)

        def export_photo(self):
            if self.photo is None:
                messagebox.showinfo(APP_NAME, "请先打开一张照片。")
                return
            src_key = self.source()[0]
            stem = Path(self.photo.source).stem
            suffix = src_key.key if isinstance(src_key, looks.Look) else "lut"
            path = filedialog.asksaveasfilename(
                title="导出照片", defaultextension=".jpg", initialfile=f"{stem}_{suffix}.jpg",
                filetypes=[("JPEG", "*.jpg")],
            )
            if not path:
                return
            table = self.table(65)
            current = self.photo

            def work():
                photo.save_jpeg(path, photo.apply_lut(table, current.pixels), current)
                return path

            self.app.run_task("正在导出照片…", work, lambda p: self.app.status(f"已导出 {p}"))

        def batch(self):
            folder = filedialog.askdirectory(title="选择要处理的照片文件夹")
            if not folder:
                return
            files = sorted(p for p in Path(folder).iterdir() if p.is_file() and photo.is_supported(p))
            if not files:
                messagebox.showinfo(APP_NAME, "该文件夹中没有支持的照片。")
                return
            src = self.source()[0]
            name = src.key if isinstance(src, looks.Look) else "lut"
            out_dir = Path(folder) / f"GR色彩工坊_{name}"
            if not messagebox.askokcancel(APP_NAME, f"将处理 {len(files)} 个文件，输出到：\n{out_dir}"):
                return
            out_dir.mkdir(exist_ok=True)
            table = self.table(65)
            self.cancel.clear()

            def work():
                failed = []
                for k, f in enumerate(files, 1):
                    if self.cancel.is_set():
                        break
                    self.app.status(f"批量处理 {k}/{len(files)}：{f.name}（按 Esc 取消）")
                    try:
                        p = photo.load(f)
                        target = out_dir / f"{f.stem}_{name}.jpg"
                        photo.save_jpeg(target, photo.apply_lut(table, p.pixels), p)
                    except Exception as exc:
                        failed.append(f"{f.name}: {exc}")
                return failed

            def done(failed):
                msg = f"完成，输出在 {out_dir}"
                if failed:
                    msg += f"\n{len(failed)} 个失败：\n" + "\n".join(failed[:10])
                messagebox.showinfo(APP_NAME, msg)

            self.app.run_task("批量处理中…", work, done)

        def export_cube(self):
            src, label = self.source()
            name = src.key if isinstance(src, looks.Look) else "custom"
            path = filedialog.asksaveasfilename(
                title="导出 .cube", defaultextension=".cube", initialfile=f"{name}.cube",
                filetypes=[("Cube LUT", "*.cube")],
            )
            if path:
                title = name if self.adjust().is_neutral() else f"{name}-adjusted"
                lut.write_cube(path, self.table(int(self.grid_size.get())), title=title)
                self.app.status(f"已导出 {path}")

        def import_cube(self):
            path = filedialog.askopenfilename(title="导入 .cube", filetypes=[("Cube LUT", "*.cube")])
            if not path:
                return
            try:
                table = lut.read_cube(path)
            except ValueError as exc:
                messagebox.showerror(APP_NAME, f"无法读取：{exc}")
                return
            if self.cube_group is None:
                self.cube_group = self.tree.insert("", "end", text="导入的 .cube", open=True)
            key = f"cube:{path}"
            if key not in self.sources:
                self.tree.insert(self.cube_group, "end", iid=key, text=Path(path).stem)
            self.sources[key] = table
            self.source_names[key] = f"导入 · {Path(path).stem}"
            self.tree.selection_set(key)
            self.tree.see(key)

    class CameraPage(ttk.Frame):
        def __init__(self, app, master):
            super().__init__(master)
            self.app = app
            self.workspace = tk.StringVar(value=str(default_workspace()))
            self.card_path = tk.StringVar()
            self.space = tk.StringVar(value="rgb")
            self.data = {}
            canvas = tk.Canvas(self, highlightthickness=0)
            bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
            self.body = ttk.Frame(canvas, padding=10)
            self.body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            window = canvas.create_window((0, 0), window=self.body, anchor="nw")
            canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
            canvas.configure(yscrollcommand=bar.set)
            canvas.pack(side="left", fill="both", expand=True)
            bar.pack(side="right", fill="y")
            wheel = lambda e: canvas.yview_scroll(-1 if e.delta > 0 or e.num == 4 else 1, "units")
            canvas.bind("<Enter>", lambda e: [canvas.bind_all(k, wheel) for k in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
            canvas.bind("<Leave>", lambda e: [canvas.unbind_all(k) for k in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
            self.body.columnconfigure(0, weight=1)
            self._build()
            self.load_state()


        def _step(self, row, title, text):
            frame = ttk.LabelFrame(self.body, text=title, padding=8)
            frame.grid(row=row, column=0, sticky="ew", pady=4)
            frame.columnconfigure(1, weight=1)
            ttk.Label(frame, text=text, wraplength=820, justify="left").grid(row=0, column=0, columnspan=4, sticky="w")
            return frame

        def _buttons(self, frame, row, items):
            bar = ttk.Frame(frame)
            bar.grid(row=row, column=0, columnspan=4, sticky="w", pady=(4, 0))
            for text, command in items:
                ttk.Button(bar, text=text, command=command).pack(side="left", padx=(0, 6))
            return bar

        def _build(self):
            warn = ttk.Label(
                self.body, foreground="#b00020", wraplength=840, justify="left",
                text="实验功能：相机脚本只在电脑端模拟验证，尚未在真机上运行。第一次请只做「1. 备份」，"
                     "确认结果后再继续。不刷固件，只做文件级备份/替换/恢复；风险自负。",
            )
            warn.grid(row=0, column=0, sticky="w", pady=(0, 6))

            f = self._step(1, "0. 准备", "工作文件夹保存备份原件和每一步的记录，请妥善保管。SD 卡请用 FAT32 格式，"
                                       "并按参考项目说明放好工厂菜单入口文件、在工厂菜单里把 Script 设为 Enable。")
            ttk.Label(f, text="工作文件夹").grid(row=1, column=0, sticky="w")
            ttk.Entry(f, textvariable=self.workspace).grid(row=1, column=1, sticky="ew", padx=4)
            ttk.Button(f, text="选择…", command=self.pick_workspace).grid(row=1, column=2)
            ttk.Label(f, text="SD 卡根目录").grid(row=2, column=0, sticky="w")
            ttk.Entry(f, textvariable=self.card_path).grid(row=2, column=1, sticky="ew", padx=4)
            ttk.Button(f, text="选择…", command=self.pick_card).grid(row=2, column=2)

            f = self._step(2, "1. 备份（只读相机，只写 SD 卡）",
                           "每行一个要备份的相机路径（A:\\... 或 E:\\...）。可用命令行 strings 工具从解包固件中找候选路径。"
                           "写入后：取出 SD 卡 → 插入相机 → 正常开机一次并等待约 10 秒 → 关机 → 卡插回电脑 → 点「校验备份」。")
            self.paths = tk.Text(f, height=4, width=80)
            self.paths.grid(row=1, column=0, columnspan=4, sticky="ew", pady=4)
            self._buttons(f, 2, [("生成备份脚本并写入 SD 卡", self.write_backup), ("校验备份并归档", self.verify_backup)])

            f = self._step(3, "2. 定位色彩表",
                           "扫描已归档的原件，列出疑似 3D 色彩表。选中一行后分别设为「基准表」（例如 Standard）"
                           "和「替换槽位」（要被你的风格替换的那个影像风格）。")
            self._buttons(f, 1, [
                ("扫描备份文件", self.scan_archive), ("设为基准表", lambda: self.mark("base")),
                ("设为替换槽位", lambda: self.mark("slot")), ("近似预览", self.preview_candidate),
            ])
            cols = ("file", "offset", "format", "score", "role")
            self.cands = ttk.Treeview(f, columns=cols, show="headings", height=6)
            for col, text, width in zip(cols, ("相机文件", "偏移", "格式", "平滑度/结构", "角色"), (300, 90, 260, 120, 80)):
                self.cands.heading(col, text=text)
                self.cands.column(col, width=width, anchor="w")
            self.cands.grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)
            self.paths.configure(width=60)

            f = self._step(4, "3. 写入当前风格",
                           "用「风格调色」页当前选中的风格（含微调），按「基准表」逐节点映射后写入「替换槽位」，文件长度不变。"
                           "写入后同样插卡开机一次，再点「校验写入」。")
            bar = self._buttons(f, 1, [("生成替换文件并写入 SD 卡", self.write_install), ("校验写入", lambda: self.verify("install"))])
            ttk.Label(bar, text="色彩表输出空间").pack(side="left", padx=(12, 4))
            ttk.Combobox(bar, textvariable=self.space, values=list(remap.SPACES), width=8, state="readonly").pack(side="left")

            f = self._step(5, "4. 恢复原厂", "把已归档的原件写回相机，然后插卡开机一次，再点「校验恢复」。")
            self._buttons(f, 1, [("生成恢复脚本并写入 SD 卡", self.write_restore), ("校验恢复", lambda: self.verify("restore"))])

            f = self._step(6, "5. 收尾", "移走 SD 卡上的启动脚本（不会删除，移到 RC_OLD 文件夹），然后在工厂菜单把 Script 设回 Disable。")
            self._buttons(f, 1, [("移走启动脚本", self.finish)])

            self.log = tk.Text(self.body, height=12, width=100, state="disabled")
            self.log.grid(row=7, column=0, sticky="ew", pady=(6, 0))


        def say(self, text):
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log.configure(state="normal")
            self.log.insert("end", f"[{stamp}] {text}\n")
            self.log.see("end")
            self.log.configure(state="disabled")
            try:
                ws = Path(self.workspace.get())
                ws.mkdir(parents=True, exist_ok=True)
                with open(ws / "log.txt", "a", encoding="utf-8") as fh:
                    fh.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {text}\n")
            except OSError:
                pass

        def ws(self):
            path = Path(self.workspace.get())
            path.mkdir(parents=True, exist_ok=True)
            return path

        def stamp(self):
            return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        def save_state(self):
            self.data["card"] = self.card_path.get()
            (self.ws() / "state.json").write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

        def load_state(self):
            try:
                self.data = json.loads((Path(self.workspace.get()) / "state.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.data = {}
            self.card_path.set(self.data.get("card", ""))
            self.paths.delete("1.0", "end")
            self.paths.insert("1.0", "\n".join(self.data.get("paths", [])))
            self.show_candidates()

        def need_card(self):
            path = self.card_path.get().strip()
            if not path or not Path(path).is_dir():
                messagebox.showwarning(APP_NAME, "请先在「0. 准备」里选择 SD 卡根目录。")
                return None
            return Path(path)

        def pick_workspace(self):
            path = filedialog.askdirectory(title="选择工作文件夹")
            if path:
                self.workspace.set(path)
                self.load_state()

        def pick_card(self):
            path = filedialog.askdirectory(title="选择 SD 卡根目录")
            if not path:
                return
            self.card_path.set(path)
            fs = card.filesystem(path)
            if fs and fs.upper() != "FAT32":
                messagebox.showwarning(APP_NAME, f"这张卡是 {fs} 格式。参考项目在 exFAT 卡上遇到过复制得到空文件，建议改用 FAT32。")
            self.say(f"SD 卡：{path}" + (f"（{fs}）" if fs else ""))
            self.save_state()

        def put_on_card(self, stage, what):
            target = self.need_card()
            if target is None:
                return False
            moved = card.stage_to_card(stage, target)
            for m in moved:
                self.say(f"卡上原有文件已移到 {m}")
            self.say(f"{what}已写入 SD 卡。请插入相机正常开机一次，等待约 10 秒后关机，再把卡插回电脑。")
            return True

        def report(self, result, ok):
            bad = 0
            for e in result["entries"]:
                verdict = e.get("status", e.get("verdict"))
                bad += verdict not in ok
                self.say(f"  {e['target']}：{verdict}")
            return bad


        def write_backup(self):
            paths = [p.strip() for p in self.paths.get("1.0", "end").splitlines() if p.strip()]
            if not paths:
                messagebox.showwarning(APP_NAME, "请至少填写一个相机路径。")
                return
            target = self.need_card()
            if target is None:
                return
            if card.has_backup_run(target) or self.data.get("archive"):
                if not messagebox.askyesno(
                    APP_NAME,
                    "已经备份过一次。重新备份会把相机「当前」的文件当作原件；"
                    "如果你已经写入过替换文件，请不要重新备份，否则会丢失真正的原件。\n\n确定要重新备份吗？",
                ):
                    return
            try:
                stage = self.ws() / f"stage-backup-{self.stamp()}"
                safety.backup_plan(paths, stage)
            except ValueError as exc:
                messagebox.showerror(APP_NAME, str(exc))
                return
            self.data.update(paths=paths, backup_stage=str(stage))
            self.save_state()
            self.put_on_card(stage, "备份脚本")

        def verify_backup(self):
            target = self.need_card()
            stage = self.data.get("backup_stage")
            if target is None or not stage:
                messagebox.showwarning(APP_NAME, "请先完成「生成备份脚本并写入 SD 卡」。")
                return
            try:
                archive = self.ws() / f"archive-{self.stamp()}"
                result = safety.backup_verify(Path(stage) / "plan.json", target, archive)
            except ValueError as exc:
                messagebox.showerror(APP_NAME, str(exc))
                return
            self.say(f"备份归档到 {archive}" + ("" if result["complete_run"] else "（警告：脚本没有跑完）"))
            bad = self.report(result, {"OK", "MISSING"})
            if any(e["status"] == "OK" for e in result["entries"]):
                self.data["archive"] = str(archive)
                self.data.pop("candidates", None)
                self.save_state()
            if bad:
                messagebox.showwarning(APP_NAME, f"{bad} 个文件没有通过校验，详见日志。不要对这些文件做写入。")
            else:
                messagebox.showinfo(APP_NAME, f"备份完成并已校验。请把整个文件夹另存一份：\n{archive}")

        def scan_archive(self):
            archive = self.data.get("archive")
            if not archive:
                messagebox.showwarning(APP_NAME, "请先完成备份并校验。")
                return
            manifest = json.loads((Path(archive) / "manifest.json").read_text())
            entries = [e for e in manifest["entries"] if e["status"] == "OK"]

            def work():
                found = []
                for e in entries:
                    data = (Path(archive) / "backup" / e["file"]).read_bytes()
                    for hit in scan.find_luts(data):
                        found.append({"target": e["target"], "file": e["file"], **hit})
                return found

            def done(found):
                self.data["candidates"] = found
                self.data.pop("base", None)
                self.data.pop("slot", None)
                self.save_state()
                self.show_candidates()
                self.say(f"扫描完成：{len(found)} 个候选色彩表。")

            self.app.run_task("正在扫描备份文件…", work, done)

        def show_candidates(self):
            self.cands.delete(*self.cands.get_children())
            for k, c in enumerate(self.data.get("candidates", [])):
                s = c["spec"]
                role = "基准" if self.data.get("base") == k else "替换槽位" if self.data.get("slot") == k else ""
                fmt = f"{s['grid']}³ {s['dtype']} {s['layout']}{'+pad' if s['pad'] else ''} {s['fastest']}/{s['channels']}"
                self.cands.insert("", "end", iid=str(k), values=(
                    c["target"], f"0x{c['offset']:x}", fmt, f"{c['rough']} / {c['structure']}", role))

        def selected_candidate(self):
            sel = self.cands.selection()
            if not sel:
                messagebox.showinfo(APP_NAME, "请先在列表中选中一个候选。")
                return None
            return int(sel[0])

        def mark(self, role):
            k = self.selected_candidate()
            if k is None:
                return
            self.data[role] = k
            self.save_state()
            self.show_candidates()

        def candidate_table(self, k):
            c = self.data["candidates"][k]
            data = (Path(self.data["archive"]) / "backup" / c["file"]).read_bytes()
            return c, data, codec.TableSpec(**c["spec"])

        def preview_candidate(self):
            k = self.selected_candidate()
            if k is None:
                return
            _, data, spec = self.candidate_table(k)
            table = np.clip(codec.decode(data, spec), 0, 1)
            img = photo.apply_lut(table, sample_image(480, 320))
            top = tk.Toplevel(self)
            top.title("候选色彩表近似预览（输入空间未知，仅供辨认）")
            pic = ImageTk.PhotoImage(Image.fromarray(img))
            label = ttk.Label(top, image=pic)
            label.image = pic
            label.pack(padx=8, pady=8)

        def write_install(self):
            if "base" not in self.data or "slot" not in self.data:
                messagebox.showwarning(APP_NAME, "请先在「2. 定位色彩表」里设置基准表和替换槽位。")
                return
            look_table = self.app.looks_page.table(65)
            label = self.app.looks_page.look_label()
            base_c, base_data, base_spec = self.candidate_table(self.data["base"])
            slot_c, slot_data, slot_spec = self.candidate_table(self.data["slot"])
            base = codec.decode(base_data, base_spec)
            if base.shape[0] != slot_spec.grid:
                base = lut.resample(base, slot_spec.grid)
            new = codec.encode(remap.remap(base, look_table, space=self.space.get()), slot_spec, slot_data)
            out = self.ws() / "replacements" / f"{slot_c['file'][:-4]}-{self.stamp()}.bin"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(new)
            try:
                stage = self.ws() / f"stage-install-{self.stamp()}"
                safety.write_plan("install", self.data["archive"], stage, {slot_c["target"]: out})
            except ValueError as exc:
                messagebox.showerror(APP_NAME, str(exc))
                return
            self.data["install_stage"] = str(stage)
            self.save_state()
            self.say(f"风格「{label}」→ {slot_c['target']} @0x{slot_c['offset']:x}，替换文件 {out.name}（与原文件等长）")
            self.put_on_card(stage, "安装脚本")

        def write_restore(self):
            if not self.data.get("archive"):
                messagebox.showwarning(APP_NAME, "没有已归档的备份，无法恢复。")
                return
            try:
                stage = self.ws() / f"stage-restore-{self.stamp()}"
                safety.write_plan("restore", self.data["archive"], stage)
            except ValueError as exc:
                messagebox.showerror(APP_NAME, str(exc))
                return
            self.data["restore_stage"] = str(stage)
            self.save_state()
            self.put_on_card(stage, "恢复脚本")

        def verify(self, kind):
            target = self.need_card()
            stage = self.data.get(f"{kind}_stage")
            if target is None or not stage:
                messagebox.showwarning(APP_NAME, "还没有生成对应的脚本。")
                return
            result = safety.readback_verify(Path(stage) / "plan.json", target)
            if not result["permit_consumed"]:
                self.say("警告：许可文件没有被消耗，脚本可能没有运行（Script 是否已开启？）")
            bad = self.report(result, {"PASS"})
            name = "写入" if kind == "install" else "恢复"
            if bad or not result["permit_consumed"]:
                messagebox.showwarning(APP_NAME, f"{name}校验未通过，详见日志。不要重复执行，先排查原因。")
            else:
                messagebox.showinfo(APP_NAME, f"{name}校验通过：读回内容与预期逐字节一致。请实拍确认效果。")

        def finish(self):
            target = self.need_card()
            if target is None:
                return
            moved = card.finish(target)
            self.say(f"启动脚本已移到 {moved}。请在工厂菜单把 Script 设回 Disable。" if moved else "卡上没有启动脚本。")

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(f"{APP_NAME}  ·  哈苏 / 富士风格 · GR 色彩表实验")
            self.geometry("1280x820")
            self.minsize(980, 640)
            if os.name == "nt":
                self.option_add("*Font", ("Microsoft YaHei UI", 9))
            self.tasks = queue.Queue()
            self.busy = False
            notebook = ttk.Notebook(self)
            notebook.pack(fill="both", expand=True)
            self.looks_page = StylePage(self, notebook)
            self.camera_page = CameraPage(self, notebook)
            help_page = ttk.Frame(notebook, padding=12)
            text = tk.Text(help_page, wrap="word", height=30)
            text.insert("1.0", HELP)
            text.configure(state="disabled")
            text.pack(fill="both", expand=True)
            notebook.add(self.looks_page, text="  风格调色  ")
            notebook.add(self.camera_page, text="  写入相机（实验）  ")
            notebook.add(help_page, text="  说明  ")
            self.statusbar = ttk.Label(self, anchor="w", padding=(8, 2))
            self.statusbar.pack(fill="x")
            self.status("就绪。选择左侧风格，打开照片预览效果。")
            self.bind("<Escape>", lambda e: self.looks_page.cancel.set())
            self._poll_job = self.after(100, self._poll)

        def destroy(self):
            self.after_cancel(self._poll_job)
            if self.looks_page.pending:
                self.looks_page.after_cancel(self.looks_page.pending)
            super().destroy()

        def status(self, text):
            self.tasks.put(("status", text))

        def run_task(self, message, work, done):
            if self.busy:
                messagebox.showinfo(APP_NAME, "上一个任务还在进行中，请稍候。")
                return
            self.busy = True
            self.status(message)

            def runner():
                try:
                    self.tasks.put(("done", done, work()))
                except Exception as exc:
                    self.tasks.put(("error", exc, traceback.format_exc()))

            threading.Thread(target=runner, daemon=True).start()

        def _poll(self):
            try:
                while True:
                    item = self.tasks.get_nowait()
                    if item[0] == "status":
                        self.statusbar.configure(text=item[1])
                    elif item[0] == "done":
                        self.busy = False
                        self.statusbar.configure(text="完成")
                        item[1](item[2])
                    else:
                        self.busy = False
                        self.statusbar.configure(text="出错")
                        log_error(item[2])
                        messagebox.showerror(APP_NAME, f"{item[1]}")
            except queue.Empty:
                pass
            self._poll_job = self.after(100, self._poll)

        def report_callback_exception(self, exc, value, tb):
            log_error("".join(traceback.format_exception(exc, value, tb)))
            messagebox.showerror(APP_NAME, f"出错了：{value}")

    return App


def log_error(text):
    try:
        path = Path(tempfile.gettempdir()) / "GRColorStudio-error.log"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"--- {datetime.datetime.now().isoformat()}\n{text}\n")
    except OSError:
        pass


def selftest(gui=True):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sample = sample_image()
        for look in looks.PRESETS:
            table = looks.bake(look, grid=17)
            assert np.isfinite(table).all() and table.min() >= 0 and table.max() <= 1, look.key
        table = looks.bake(looks.BY_KEY["classic_chrome"], looks.Adjust(contrast=0.3, fade=0.2))
        out = photo.apply_lut(table, sample)
        photo.save_jpeg(tmp / "out.jpg", out, photo.Photo(sample))
        assert photo.load(tmp / "out.jpg").pixels.shape == sample.shape
        lut.write_cube(tmp / "x.cube", table)
        assert np.abs(lut.read_cube(tmp / "x.cube") - table).max() < 1e-5
        safety.backup_plan(["A:\\Resource\\x.bin"], tmp / "stage")
        assert (tmp / "stage" / "card" / "script" / "startup.ttl").exists()
        assert photo.raw_supported(), "rawpy missing from build"
        if gui:
            App = run_gui()
            app = App()
            app.withdraw()
            for key in ("hncs", "velvia", "acros"):
                app.looks_page.tree.selection_set(key)
                app.update()
                app.looks_page.refresh()
            app.update()
            app.destroy()
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    if "--selftest" in argv:
        try:
            selftest(gui="--no-gui" not in argv)
        except Exception:
            log_error(traceback.format_exc())
            print(traceback.format_exc(), file=sys.stderr)
            sys.exit(1)
        print("selftest ok")
        sys.exit(0)
    App = run_gui()
    App().mainloop()


if __name__ == "__main__":
    main()
