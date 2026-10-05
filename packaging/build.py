import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    icon = build / "icon.ico"
    subprocess.run([sys.executable, str(ROOT / "packaging" / "make_icon.py"), str(icon)], check=True)
    subprocess.run(
        [
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
            "--name", "GRColorStudio", "--icon", str(icon),
            "--collect-all", "rawpy",
            "--hidden-import", "PIL._tkinter_finder",
            "--distpath", str(ROOT / "dist"),
            "--workpath", str(build / "pyinstaller"),
            "--specpath", str(build),
            str(ROOT / "gr_color_studio.py"),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
