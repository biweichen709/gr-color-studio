# 写入相机：机内色彩表替换流程（实验）

> 另见：[GR III v2.10 工厂菜单与色彩调查记录](gr3-factory-and-color.md)——GR III 可进工厂菜单，但无可替换的色彩表文件。


把自己的胶片预设"倒映射"到理光机内色彩表的同一组网格坐标上，替换机内的某个影像风格，并且**随时能恢复原样**。

建立在 [radium-wang/ricoh-gr4-firmware-analysis-and-feature-expansion](https://github.com/radium-wang/ricoh-gr4-firmware-analysis-and-feature-expansion) 的 GR IV 研究之上：工厂菜单入口、`Script` 启动脚本、`A:` 资源盘可写、`filecopy` 不截断等结论都来自该项目的实机记录。本仓库代码为独立实现，未复制其文件。

## 先说结论：为什么不做"刷不坏的固件"

| 方案 | 结论 |
| --- | --- |
| 改固件包再刷机 | **不做。** 官方升级包有型号/版本字段和整包校验，理光明确不支持降级，是否另有签名未知；改坏后没有已知恢复通道（串口未打通、无 JTAG 记录）。刷坏可能直接变砖，官方包也不保证能覆盖回去。 |
| SD 卡启动脚本改 `A:` 资源文件 | **采用。** 参考项目已在 GR IV 实机验证：工厂菜单 `Script=Enable` 后，相机开机执行卡上的 `script/startup.ttl`，可用内置 `filecopy` 读写 `A:` 资源盘，改动持久保存。 |

所以这里的"刷回去"是**文件级**的：先把要动的文件原样备份到卡和电脑（带 SHA-256），任何时候都能用恢复脚本写回并逐字节核对。

前提是色彩表确实是 `A:` 上的独立文件。如果它编译在 RTOS 程序里，目前没有安全的写入方法，流程应在第 3 步停下，而不是冒险改固件。

## 对原始思路的修正

- **相机并不是"先生成 DNG 再套表"。** 传感器数据直接进 ISP 流水线（Socionext Milbeaut），DNG 是并行输出。一个影像风格通常由白平衡、色彩矩阵、色调曲线、饱和度/色相、3D LUT、颗粒/暗角等组合而成，不一定只有一张 3D 表。
- **"坐标 1.2.3 一一对应"的思路是对的。** 保留理光表的网格坐标不变，只改每个节点存的值：

  ```
  新表[i,j,k] = 胶片外观( Standard表[i,j,k] )
  ```

  输入轴完全不动，所以**不需要知道表的输入色彩空间**，只需要知道输出值是什么空间（`--space rgb|linear|ycbcr`）。
- **文件长度必须与原文件完全一致**（GR IV 的 `filecopy` 覆盖时不截断）。工具只改表所在的字节，长度不变。
- 颗粒、暗角、锐化、清晰度是空间效果，放不进色彩表。

## 工作流程

需要 Python 3.10+、FAT32 格式的 SD 卡（exFAT 卡上曾出现复制得到空文件，见参考项目 Issue #1），以及充足电量。

```sh
python3 -m pip install -r requirements.txt
```

### 0. 进入工厂菜单并打开 Script

按参考项目 [extensions 文档](https://github.com/radium-wang/ricoh-gr4-firmware-analysis-and-feature-expansion/tree/main/docs/extensions) 制作入口文件，开机时按住 MENU 进入工厂菜单，把 `Script` 设为 Enable。**不要改其他菜单项。**

### 1. 找候选文件（离线）

用参考项目的 `tools/inspect_firmware.py --unpack` 解包官方固件，再列出资源路径和色彩相关字符串：

```sh
python3 -m ricoh_color strings gr4-decoded.bin --paths-only
python3 -m ricoh_color strings gr4-decoded.bin            # 含 positive / bleach / lut / gamma 等关键字
```

把可能与色彩有关的 `A:\...` 路径写进 `paths.txt`，一行一个。`#` 开头的行是注释。

### 2. 只读备份：安全网

```sh
python3 -m ricoh_color backup-plan paths.txt stage-backup
# 把 stage-backup/card/ 里的内容拷到 SD 卡根目录，正常开机一次，等存储灯结束后关机
python3 -m ricoh_color backup-verify stage-backup /Volumes/SD archive-body1
```

备份脚本只读相机存储、只写 SD 卡。它是一次性的，并且不会覆盖已有备份。每个文件在机内比对长度，结果写入 `RCBnnn.TXT`。`backup-verify` 把通过的原件和 SHA-256 存进 `archive-body1/`。**再拷一份到别处，每台机身单独一套。**

### 3. 定位色彩表

```sh
python3 -m ricoh_color scan archive-body1/backup/RCB001.BIN --curves --out cands
```

每个候选会生成 `candNNN.json`（格式描述）和 `candNNN.cube`（预览）。用 `apply` 把预览表套到照片上，看它像哪个风格。

结构上无法区分红和蓝：如果颜色看起来红蓝反了，就改用候选里 `alternative` 给出的 `fastest`/`channels`。`scan` 是启发式的：命中只是假设，最终要靠第 7 步的实拍闭环确认。

### 4. 准备胶片外观（`.cube`）

三种来源任选：

- 现成的、针对 sRGB/Rec.709 显示输入的 `.cube`；
- Lightroom/Photoshop 预设：`hald --level 8 hald.png`，对它套预设导出（关闭颗粒、暗角、锐化、镜头校正），再 `hald-to-cube`。这是最准确的方法；
- 只有样片时：用机内 Standard JPEG 和你调好的同一张 JPEG 拟合，`fit --pair std.jpg film.jpg -o film.cube`。

外观要定义在"Standard 出片之上"，也就是输入是理光 Standard 的 JPEG 颜色。

### 5. 映射：把外观写进理光表的节点

```sh
python3 -m ricoh_color remap archive-body1/backup/RCB001.BIN cands/cand002.json film.cube \
    --base archive-body1/backup/RCB001.BIN cands/cand001.json \
    --strength 1.0 -o ic-film.bin
```

`--base` 指定取值的源表（例如 Standard）。结果写进第一个参数对应的槽位（例如要被替换的 Positive Film）。输出与原文件等长，表以外的字节保持不变。

### 6. 安装（一次性、等长校验、带读回）

```sh
python3 -m ricoh_color install-plan archive-body1 stage-install --replace "A:\...\原路径=ic-film.bin"
# 把 stage-install/card/ 拷到卡根目录，开机一次
python3 -m ricoh_color verify-readback stage-install /Volumes/SD     # 必须全部 PASS
```

### 7. 闭环验证

用机内 RAW 显像，把同一张 DNG 分别用 Standard 和被替换的风格出 JPEG。再把 Standard 那张套上 `film.cube` 作为目标，与相机实际输出对比：

```sh
python3 -m ricoh_color compare camera-film.jpg target.jpg --downscale 4
```

### 8. 恢复原样

```sh
python3 -m ricoh_color restore-plan archive-body1 stage-restore
# 拷到卡、开机一次
python3 -m ricoh_color verify-readback stage-restore /Volumes/SD
```

### 9. 收尾

删除卡上的 `script/startup.ttl`，在工厂菜单把 `Script` 设回 Disable，正常开关机确认。

## 工具强制的安全规则

- 只为**已有通过校验的备份**的文件生成写入脚本；归档原件的 SHA-256 不符时拒绝。
- 只允许写 `A:`；`E:` 等区域（可能含机身校准数据）只读备份。
- 新文件必须与备份**完全等长**；相机端写入前再次核对源文件和目标文件的长度，长度变了就跳过。
- 许可文件 `RCARM.TXT` 先被消耗并读回确认，之后才写入，所以重复开机不会重复写。每个目标的读回文件名在写入前预留。
- 电脑端对每个读回文件做完整 SHA-256 比对。空文件、长度不符或内容不符都算失败。
- 生成的脚本只使用参考项目已在 GR IV 上执行过的命令（`filesearch filestat filecopy filecreate filewrite fileclose fileopen fileread strcompare if/endif goto exit`）。

## 未验证与局限（请认真看）

- **新的组合脚本只在电脑端模拟器（`tests/ttl_sim.py`）里跑过，未上机。** 第一次上机请只做第 2 步备份，确认结果后再继续。
- 理光的色彩表在哪个文件、是什么格式，目前**未知**。`scan` 只能给出候选。
- 参考项目只证实了 `A:` 根目录和关机图路径可写，其他路径能否写入需要逐个验证。
- 写入不是断电原子操作。请保证电量充足；中断后先用 `verify-readback` 诊断，不要反复重试。
- 只针对 GR IV（1.11）的研究结论，其他机型、固件版本不保证适用。
- 纯黑白表不会被 `scan` 报出（设计上要求三个输入轴各驱动一个不同的输出通道）。
- 不要分发理光固件、解包文件或相机读回数据。

## 命令一览

| 命令 | 作用 |
| --- | --- |
| `strings FILE` | 列出 `A:\` 等路径和色彩关键字（ASCII / UTF-16） |
| `scan FILE [--out DIR] [--curves]` | 找 3D 色彩表和 1D 曲线候选 |
| `extract FILE SPEC OUT.cube` | 按格式描述把表导出为 `.cube` |
| `remap INTO SPEC LOOK.cube -o OUT [--base FILE SPEC]` | 节点级映射，输出与原文件等长 |
| `hald` / `hald-to-cube` | 用 Hald 图把任意预设变成 `.cube` |
| `fit --pair SRC DST -o OUT.cube` | 从对齐的图片对拟合 `.cube` |
| `apply LUT IMG OUT` / `compare A B` | 预览 / CIEDE2000 色差统计 |
| `backup-plan` / `backup-verify` | 只读备份与归档 |
| `install-plan` / `restore-plan` / `verify-readback` | 一次性写入、恢复与读回核对 |

## 开发

```sh
python3 -m unittest discover -s tests -t .
```

测试覆盖编解码逐字节往返、扫描器召回率与误报（噪声、渐变图、指针表、正弦表）、拟合精度、CIEDE2000 参考值，以及在 TTL 模拟器里跑通"备份 → 安装 → 读回 → 恢复"全流程（包括复制得到空文件、目标长度变化、归档被篡改等故障）。
