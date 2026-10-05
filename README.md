# GR Color Studio

理光 GR 色彩风格工具。提供 HNCS 与胶片风格预设，支持 JPG/DNG 处理与 `.cube` 导出。

## 下载

[**下载 GRColorStudio.exe**](https://github.com/biweichen709/gr-color-studio/releases/latest/download/GRColorStudio.exe)（Windows 10/11，64 位）

- 免安装，双击运行，完全离线。
- 完整包（含使用说明）：[GRColorStudio-windows-x64.zip](https://github.com/biweichen709/gr-color-studio/releases/latest/download/GRColorStudio-windows-x64.zip)
- 程序未做代码签名。首次运行若出现 SmartScreen 提示，选择「更多信息 → 仍要运行」。

## 功能

| 功能 | 说明 |
| --- | --- |
| 风格预设 | HNCS 风格 3 款、胶片风格 10 款、黑白 4 款 |
| 微调 | 强度、色温、色调、对比、饱和、褪色；原图与效果对比预览 |
| 照片处理 | JPG / PNG / TIFF / DNG 等 RAW；导出 JPEG 保留 EXIF；批量处理文件夹 |
| LUT | 导出 33 / 65 精度 `.cube`，可导入自定义 `.cube` |
| 写入相机（实验） | GR IV 机内色彩表的备份、替换与恢复向导，见 [docs/camera-workflow.md](docs/camera-workflow.md) |
| 固件分析 | 解包官方固件，按可能性列出候选资源文件，并检测固件内嵌的色彩表 |
| 工厂菜单入口 | 从固件推定并写入入口文件；未知机型用 1000 候选法（GR III 实测见下） |

## 预设

- **HNCS 风格：** 自然、人像、风光
- **胶片风格：** PROVIA、Velvia、ASTIA、Classic Chrome、Classic Neg、Nostalgic Neg、ETERNA、ETERNA 漂白、PRO Neg. Std、REALA ACE
- **黑白：** ACROS、ACROS + 红滤镜、单色、棕褐色

预设依据公开资料调校，为近似风格，不含原厂数据。本项目与 Hasselblad、Fujifilm、Ricoh 无关，相关名称仅用于说明风格方向。

## 从源码运行与构建

```sh
python -m pip install -r requirements-build.txt
python -m ricoh_color.app           # 运行
python packaging/build.py           # 构建 dist/GRColorStudio.exe（需在 Windows 上）
python -m unittest discover -s tests -t .
```

推送到 `main` 后，GitHub Actions 会在 Windows 上运行测试、构建并自检 exe，然后发布到 Releases。

**GR III 调查结论**：GR III v2.10 的工厂菜单可进入（机型编号 78350 + 候选法，已实测），但其色彩风格参数化写死在固件中、没有可替换的色彩表文件，机内替换色彩不可行；详见 [docs/gr3-factory-and-color.md](docs/gr3-factory-and-color.md)。GR III 请在电脑端用本工具处理照片。

固件容器格式参考 [yeahnope/gr_unpack](https://github.com/yeahnope/gr_unpack) 的公开说明；相机脚本流程参考 [radium-wang/ricoh-gr4-firmware-analysis-and-feature-expansion](https://github.com/radium-wang/ricoh-gr4-firmware-analysis-and-feature-expansion) 的实机研究。
