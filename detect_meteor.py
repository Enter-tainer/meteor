#!/usr/bin/env python3
"""
流星检测脚本
从输入文件夹检测流星，并将包含流星的图片复制到输出文件夹
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def detect_meteor(image_path: Path, config: dict) -> tuple[bool, np.ndarray | None]:
    """
    检测图片中是否有流星
    
    流星特征：
    1. 亮度较高的线性物体
    2. 通常比较细长
    
    Args:
        image_path: 图片路径
        config: 检测配置参数
        
    Returns:
        (是否检测到流星, 调试用的标注图像)
    """
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"警告: 无法读取图片 {image_path}")
        return False, None
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用阈值提取亮区域（流星通常很亮）
    threshold = config.get("brightness_threshold", 200)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # 形态学操作：先膨胀后腐蚀，连接断开的线段
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # 使用霍夫变换检测直线
    min_line_length = config.get("min_line_length", 50)
    max_line_gap = config.get("max_line_gap", 10)
    hough_threshold = config.get("hough_threshold", 30)
    
    lines = cv2.HoughLinesP(
        eroded,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    # 准备调试图像
    debug_img = img.copy() if config.get("debug", False) else None
    
    if lines is None:
        return False, debug_img
    
    # 分析检测到的线条
    meteor_lines = []
    min_aspect_ratio = config.get("min_aspect_ratio", 3.0)  # 流星应该是细长的
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # 计算线条周围的亮度
        # 创建一个mask来提取线条区域
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 5)
        mean_brightness = cv2.mean(gray, mask=mask)[0]
        
        # 流星判断条件：
        # 1. 长度足够长
        # 2. 线条区域的平均亮度足够高
        min_brightness = config.get("min_line_brightness", 150)
        
        if length >= min_line_length and mean_brightness >= min_brightness:
            meteor_lines.append((x1, y1, x2, y2, length, mean_brightness))
            
            if debug_img is not None:
                # 在调试图像上标注检测到的流星
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_img,
                    f"L:{length:.0f} B:{mean_brightness:.0f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
    
    has_meteor = len(meteor_lines) > 0
    
    if has_meteor and config.get("verbose", False):
        print(f"  检测到 {len(meteor_lines)} 条潜在流星轨迹")
        for i, (x1, y1, x2, y2, length, brightness) in enumerate(meteor_lines):
            print(f"    #{i+1}: 长度={length:.1f}px, 亮度={brightness:.1f}")
    
    return has_meteor, debug_img


def find_raw_file(jpg_path: Path, raw_extensions: list[str] = None) -> Path | None:
    """
    查找与JPG对应的RAW文件
    
    Args:
        jpg_path: JPG文件路径
        raw_extensions: RAW文件扩展名列表
        
    Returns:
        RAW文件路径，如果不存在则返回None
    """
    if raw_extensions is None:
        raw_extensions = [".arw", ".ARW", ".raw", ".RAW", ".cr2", ".CR2", ".nef", ".NEF"]
    
    stem = jpg_path.stem
    parent = jpg_path.parent
    
    for ext in raw_extensions:
        raw_path = parent / (stem + ext)
        if raw_path.exists():
            return raw_path
    
    return None


def process_folder(
    input_folder: Path,
    output_folder: Path,
    config: dict
) -> dict:
    """
    处理输入文件夹中的所有图片
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        config: 配置参数
        
    Returns:
        处理统计信息
    """
    stats = {
        "total": 0,
        "meteor_detected": 0,
        "copied_jpg": 0,
        "copied_raw": 0,
        "raw_not_found": 0
    }
    
    # 确保输出文件夹存在
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 如果开启调试模式，创建调试输出文件夹
    debug_folder = None
    if config.get("debug", False) and config.get("save_debug_images", False):
        debug_folder = output_folder / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JPG文件
    jpg_patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    jpg_files = []
    for pattern in jpg_patterns:
        jpg_files.extend(input_folder.glob(pattern))
    
    # 递归搜索
    if config.get("recursive", False):
        for pattern in jpg_patterns:
            jpg_files.extend(input_folder.rglob(pattern))
    
    jpg_files = sorted(set(jpg_files))  # 去重并排序
    
    # 如果指定了起始文件，跳过之前的文件
    start_from = config.get("start_from")
    if start_from:
        start_index = None
        for i, f in enumerate(jpg_files):
            if start_from in f.stem:
                start_index = i
                break
        if start_index is not None:
            skipped = len(jpg_files[:start_index])
            jpg_files = jpg_files[start_index:]
            print(f"从 {start_from} 开始，跳过前 {skipped} 个文件")
        else:
            print(f"警告: 未找到包含 '{start_from}' 的文件，从头开始处理")
    
    stats["total"] = len(jpg_files)
    
    print(f"找到 {len(jpg_files)} 个JPG文件")
    print("-" * 50)
    
    for i, jpg_path in enumerate(jpg_files, 1):
        relative_path = jpg_path.relative_to(input_folder) if jpg_path.is_relative_to(input_folder) else jpg_path.name
        print(f"[{i}/{len(jpg_files)}] 处理: {relative_path}")
        
        # 检测流星
        has_meteor, debug_img = detect_meteor(jpg_path, config)
        
        if has_meteor:
            stats["meteor_detected"] += 1
            print(f"  ✓ 检测到流星!")
            
            # 复制JPG
            dst_jpg = output_folder / jpg_path.name
            if not config.get("dry_run", False):
                shutil.copy2(jpg_path, dst_jpg)
                stats["copied_jpg"] += 1
                print(f"  → 复制JPG: {dst_jpg.name}")
            else:
                print(f"  → [模拟] 复制JPG: {dst_jpg.name}")
            
            # 如果不是debug模式，也复制RAW文件
            if not config.get("debug", False):
                raw_path = find_raw_file(jpg_path)
                if raw_path:
                    dst_raw = output_folder / raw_path.name
                    if not config.get("dry_run", False):
                        shutil.copy2(raw_path, dst_raw)
                        stats["copied_raw"] += 1
                        print(f"  → 复制RAW: {dst_raw.name}")
                    else:
                        print(f"  → [模拟] 复制RAW: {dst_raw.name}")
                else:
                    stats["raw_not_found"] += 1
                    print(f"  ! 未找到对应的RAW文件")
            
            # 保存调试图像
            if debug_folder and debug_img is not None:
                debug_path = debug_folder / f"debug_{jpg_path.stem}.jpg"
                cv2.imwrite(str(debug_path), debug_img)
                print(f"  → 保存调试图像: {debug_path.name}")
        else:
            if config.get("verbose", False):
                print(f"  - 未检测到流星")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="流星检测脚本 - 从图片中检测流星并复制相关文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python detect_meteor.py /path/to/sd_card/DCIM /path/to/output
  
  # Debug模式（只复制JPG，不复制RAW）
  python detect_meteor.py /path/to/input /path/to/output --debug
  
  # 调整检测参数
  python detect_meteor.py /path/to/input /path/to/output --threshold 180 --min-length 100
  
  # 模拟运行（不实际复制文件）
  python detect_meteor.py /path/to/input /path/to/output --dry-run
        """
    )
    
    parser.add_argument("input", type=Path, help="输入文件夹路径（SD卡目录）")
    parser.add_argument("output", type=Path, help="输出文件夹路径")
    
    # 模式选项
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Debug模式：只复制JPG，不复制RAW文件"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="模拟运行：只显示会执行的操作，不实际复制文件"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="递归搜索子文件夹"
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help="保存带有检测标注的调试图像"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="从指定文件名开始处理（跳过之前的文件），例如: MGT04412"
    )
    
    # 检测参数
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=200,
        help="亮度阈值 (0-255)，默认200"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="最小流星长度（像素），默认50"
    )
    parser.add_argument(
        "--min-brightness",
        type=int,
        default=150,
        help="流星最小平均亮度，默认150"
    )
    parser.add_argument(
        "--hough-threshold",
        type=int,
        default=30,
        help="霍夫变换阈值，默认30"
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=10,
        help="最大线段间隙（像素），默认10"
    )
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not args.input.exists():
        print(f"错误: 输入路径不存在: {args.input}")
        return 1
    
    if not args.input.is_dir():
        print(f"错误: 输入路径不是文件夹: {args.input}")
        return 1
    
    # 构建配置
    config = {
        "debug": args.debug,
        "dry_run": args.dry_run,
        "verbose": args.verbose,
        "recursive": args.recursive,
        "save_debug_images": args.save_debug_images or args.debug,
        "brightness_threshold": args.threshold,
        "min_line_length": args.min_length,
        "min_line_brightness": args.min_brightness,
        "hough_threshold": args.hough_threshold,
        "max_line_gap": args.max_gap,
        "start_from": args.start_from,
    }
    
    # 打印配置信息
    print("=" * 50)
    print("流星检测脚本")
    print("=" * 50)
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print(f"模式: {'Debug (只复制JPG)' if args.debug else '正常 (复制JPG+RAW)'}")
    if args.dry_run:
        print("** 模拟运行模式 - 不会实际复制文件 **")
    print("-" * 50)
    print(f"检测参数:")
    print(f"  亮度阈值: {config['brightness_threshold']}")
    print(f"  最小流星长度: {config['min_line_length']}px")
    print(f"  最小平均亮度: {config['min_line_brightness']}")
    print(f"  霍夫阈值: {config['hough_threshold']}")
    print("=" * 50)
    print()
    
    # 处理文件夹
    stats = process_folder(args.input, args.output, config)
    
    # 打印统计信息
    print()
    print("=" * 50)
    print("处理完成!")
    print("=" * 50)
    print(f"总共扫描: {stats['total']} 张图片")
    print(f"检测到流星: {stats['meteor_detected']} 张")
    if not args.dry_run:
        print(f"复制JPG: {stats['copied_jpg']} 个")
        if not args.debug:
            print(f"复制RAW: {stats['copied_raw']} 个")
            if stats['raw_not_found'] > 0:
                print(f"未找到RAW: {stats['raw_not_found']} 个")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())
