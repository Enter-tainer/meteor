# 🌠 流星检测脚本

从大量拍摄的星空照片中自动检测流星，并将包含流星的图片（JPG + RAW）复制到指定文件夹。

## 动机

拍摄流星雨时，相机会连续拍摄成百上千张照片，但实际捕捉到流星的可能只有几十张。手动筛选既耗时又容易遗漏。

这个脚本可以：
1. **自动检测**：使用 OpenCV 霍夫变换检测图像中的流星轨迹
2. **智能过滤**：通过长宽比、亮度、角度等特征排除误检（地面物体、飞机等）
3. **批量复制**：将检测到流星的 JPG 和对应的 RAW 文件一起复制出来
4. **并行处理**：多进程加速，充分利用 CPU

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd meteor

# 安装依赖（使用 uv）
uv sync
```

## 快速开始

### Debug 模式（只复制 JPG，用于调试参数）

```bash
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug
```

### 正式使用（复制 JPG + RAW）

```bash
uv run python detect_meteor.py /mnt/sdcard/DCIM output
```

## 常用命令

### 基本检测

```bash
# Debug 模式 + 保存标注图像
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --save-debug-images
```

### 从指定文件开始

```bash
# 从 MGT04412 开始处理，跳过之前的文件
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --start-from MGT04412
```

### 指定处理范围

```bash
# 从 MGT04412 开始，到 MGT05000 结束
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --start-from MGT04412 --end-at MGT05000
```

```bash
# 只处理到 MGT05000
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --end-at MGT05000
```

### 调整检测参数

```bash
# 提高长宽比要求（减少误检）
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --min-aspect-ratio 12 --max-width 10
```

```bash
# 降低亮度阈值（检测更暗的流星）
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --threshold 180 --min-brightness 120
```

```bash
# 排除更多底部区域（地面占比大）
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --exclude-bottom 0.3
```

### 并行处理

```bash
# 使用 8 个进程并行处理
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug -j 8
```

### 递归搜索子文件夹

```bash
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug --recursive
```

### 模拟运行（不实际复制）

```bash
uv run python detect_meteor.py /mnt/sdcard/DCIM output --dry-run
```

### 详细输出

```bash
uv run python detect_meteor.py /mnt/sdcard/DCIM output --debug -v --save-debug-images
```

## 参数说明

### 模式选项

| 参数 | 说明 |
|------|------|
| `--debug, -d` | Debug 模式：只复制 JPG，不复制 RAW |
| `--dry-run, -n` | 模拟运行，不实际复制文件 |
| `--verbose, -v` | 详细输出 |
| `--recursive, -r` | 递归搜索子文件夹 |
| `--save-debug-images` | 保存带检测标注的图像 |
| `--start-from` | 从指定文件名开始处理 |
| `--end-at` | 处理到指定文件名为止（包含） |
| `--workers, -j` | 并行进程数，默认为 CPU 核心数 |

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threshold, -t` | 200 | 亮度阈值 (0-255) |
| `--min-length` | 50 | 最小流星长度 (px) |
| `--min-brightness` | 150 | 流星最小平均亮度 |
| `--hough-threshold` | 30 | 霍夫变换阈值 |
| `--max-gap` | 10 | 最大线段间隙 (px) |
| `--min-aspect-ratio` | 8.0 | 最小长宽比（越大越严格） |
| `--max-width` | 15 | 最大线条宽度 (px) |
| `--exclude-bottom` | 0.1 | 排除底部区域比例 (0-1) |
| `--min-angle` | 3.0 | 最小角度（度），排除水平线 |

## 调参指南

### 误检太多（把地面物体当流星）

```bash
# 提高长宽比，降低最大宽度，排除更多底部
uv run python detect_meteor.py input output --debug \
    --min-aspect-ratio 12 \
    --max-width 10 \
    --exclude-bottom 0.3 \
    --min-angle 15
```

### 漏检（真正的流星没检测到）

```bash
# 降低阈值和最小长度
uv run python detect_meteor.py input output --debug \
    --threshold 180 \
    --min-length 30 \
    --min-brightness 120
```

### 流星轨迹断成多段

```bash
# 增加线段间隙容忍度
uv run python detect_meteor.py input output --debug --max-gap 20
```

## 调试图像说明

使用 `--save-debug-images` 后，会在 `output/debug/` 目录生成带标注的图像：

- **绿色线条 + 标注**：被识别为流星
  - `L`: 长度 (px)
  - `B`: 亮度
  - `R`: 长宽比
  - `W`: 宽度 (px)
- **红色线条**（verbose 模式）：被过滤的线条

## 检测原理

1. **灰度化 + 高斯模糊**：减少噪声
2. **二值化**：提取高亮区域
3. **霍夫变换**：检测直线
4. **特征过滤**：
   - 长宽比（流星很细长）
   - 宽度（流星很细）
   - 角度（排除水平线）
   - 位置（排除底部地面）
   - 亮度（流星较亮）

## 支持的 RAW 格式

- Sony: `.ARW`
- Canon: `.CR2`
- Nikon: `.NEF`
- 通用: `.RAW`

## License

MIT
