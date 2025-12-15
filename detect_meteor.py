#!/usr/bin/env python3
"""
流星检测脚本
从输入文件夹检测流星，并将包含流星的图片复制到输出文件夹
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
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
    min_aspect_ratio = config.get("min_aspect_ratio", 8.0)  # 流星应该是非常细长的
    max_line_width = config.get("max_line_width", 15)  # 流星宽度不应该太宽
    exclude_bottom = config.get("exclude_bottom", 0.2)  # 排除底部区域
    min_angle = config.get("min_angle", 15.0)  # 最小角度，排除水平线
    
    img_height = gray.shape[0]
    max_y = int(img_height * (1 - exclude_bottom))  # 底部排除线
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 检查是否在底部排除区域
        if y1 > max_y and y2 > max_y:
            continue  # 整条线都在底部区域，跳过
        
        # 计算线条角度（相对于水平线）
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        angle = np.degrees(np.arctan2(dy, dx))  # 0度=水平，90度=垂直
        
        # 排除接近水平的线条
        if angle < min_angle:
            continue
        
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # 创建一个mask来分析线条区域
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
        
        # 计算线条区域的平均亮度
        mean_brightness = cv2.mean(gray, mask=mask)[0]
        
        # 分析线条的宽度：在二值图上沿线条方向采样
        # 创建一个更宽的mask来提取线条周围区域
        wide_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(wide_mask, (x1, y1), (x2, y2), 255, 30)  # 较宽的区域
        
        # 在这个区域内找到实际的亮区域轮廓
        masked_binary = cv2.bitwise_and(binary, binary, mask=wide_mask)
        contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算最大轮廓的长宽比
        line_width = 0
        aspect_ratio = 0
        if contours:
            # 找到最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            # 使用最小外接矩形计算长宽比
            rect = cv2.minAreaRect(max_contour)
            w, h = rect[1]
            if w > 0 and h > 0:
                # 确保长边/短边
                if w < h:
                    w, h = h, w
                aspect_ratio = w / h
                line_width = h  # 短边就是宽度
        
        # 流星判断条件：
        # 1. 长度足够长
        # 2. 线条区域的平均亮度足够高
        # 3. 长宽比足够大（细长）
        # 4. 宽度不能太宽
        min_brightness = config.get("min_line_brightness", 150)
        
        is_meteor = (
            length >= min_line_length and 
            mean_brightness >= min_brightness and
            aspect_ratio >= min_aspect_ratio and
            line_width <= max_line_width
        )
        
        if is_meteor:
            meteor_lines.append((x1, y1, x2, y2, length, mean_brightness, aspect_ratio, line_width))
            
            if debug_img is not None:
                # 在调试图像上标注检测到的流星（绿色）
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_img,
                    f"L:{length:.0f} B:{mean_brightness:.0f} R:{aspect_ratio:.1f} W:{line_width:.1f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
        elif debug_img is not None and config.get("verbose", False):
            # 在调试图像上标注被排除的线条（红色）
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(
                debug_img,
                f"L:{length:.0f} R:{aspect_ratio:.1f} W:{line_width:.1f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
    
    has_meteor = len(meteor_lines) > 0
    
    if has_meteor and config.get("verbose", False):
        print(f"  检测到 {len(meteor_lines)} 条潜在流星轨迹")
        for i, (x1, y1, x2, y2, length, brightness, aspect_ratio, width) in enumerate(meteor_lines):
            print(f"    #{i+1}: 长度={length:.1f}px, 亮度={brightness:.1f}, 长宽比={aspect_ratio:.1f}, 宽度={width:.1f}px")
    
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


def process_single_image(args: tuple) -> dict:
    """
    处理单张图片（用于并行处理）
    
    Args:
        args: (jpg_path, config, output_folder, debug_folder)
        
    Returns:
        处理结果字典
    """
    jpg_path, config, output_folder, debug_folder = args
    
    result = {
        "path": jpg_path,
        "has_meteor": False,
        "copied_jpg": False,
        "copied_raw": False,
        "raw_not_found": False,
        "debug_img": None,
    }
    
    # 检测流星
    has_meteor, debug_img = detect_meteor(jpg_path, config)
    result["has_meteor"] = has_meteor
    
    if has_meteor:
        # 复制JPG
        dst_jpg = output_folder / jpg_path.name
        if not config.get("dry_run", False):
            shutil.copy2(jpg_path, dst_jpg)
            result["copied_jpg"] = True
        
        # 如果不是debug模式，也复制RAW文件
        if not config.get("debug", False):
            raw_path = find_raw_file(jpg_path)
            if raw_path:
                dst_raw = output_folder / raw_path.name
                if not config.get("dry_run", False):
                    shutil.copy2(raw_path, dst_raw)
                    result["copied_raw"] = True
            else:
                result["raw_not_found"] = True
        
        # 保存调试图像
        if debug_folder and debug_img is not None:
            debug_path = debug_folder / f"debug_{jpg_path.stem}.jpg"
            cv2.imwrite(str(debug_path), debug_img)
    
    return result


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
    
    # 如果指定了结束文件，截断列表
    end_at = config.get("end_at")
    if end_at:
        end_index = None
        for i, f in enumerate(jpg_files):
            if end_at in f.stem:
                end_index = i
                break
        if end_index is not None:
            jpg_files = jpg_files[:end_index + 1]  # 包含结束文件
            print(f"处理到 {end_at} 为止（共 {len(jpg_files)} 个文件）")
        else:
            print(f"警告: 未找到包含 '{end_at}' 的文件，处理到最后")
    
    stats["total"] = len(jpg_files)
    
    num_workers = config.get("workers", multiprocessing.cpu_count())
    print(f"找到 {len(jpg_files)} 个JPG文件")
    print(f"使用 {num_workers} 个并行进程")
    print("-" * 50)
    
    # 准备并行任务参数
    tasks = [(jpg_path, config, output_folder, debug_folder) for jpg_path in jpg_files]
    
    # 使用进程池并行处理
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_path = {executor.submit(process_single_image, task): task[0] for task in tasks}
        
        # 收集结果
        for future in as_completed(future_to_path):
            jpg_path = future_to_path[future]
            completed += 1
            
            try:
                result = future.result()
                relative_path = jpg_path.relative_to(input_folder) if jpg_path.is_relative_to(input_folder) else jpg_path.name
                
                if result["has_meteor"]:
                    stats["meteor_detected"] += 1
                    print(f"[{completed}/{len(jpg_files)}] ✓ 流星: {relative_path}")
                    
                    if result["copied_jpg"]:
                        stats["copied_jpg"] += 1
                    if result["copied_raw"]:
                        stats["copied_raw"] += 1
                    if result["raw_not_found"]:
                        stats["raw_not_found"] += 1
                elif config.get("verbose", False):
                    print(f"[{completed}/{len(jpg_files)}] - 无流星: {relative_path}")
                else:
                    # 简单进度显示
                    if completed % 50 == 0 or completed == len(jpg_files):
                        print(f"进度: {completed}/{len(jpg_files)} ({completed*100//len(jpg_files)}%)")
                        
            except Exception as e:
                print(f"[{completed}/{len(jpg_files)}] 错误处理 {jpg_path.name}: {e}")
    
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
    parser.add_argument(
        "--end-at",
        type=str,
        default=None,
        help="处理到指定文件名为止（包含该文件），例如: MGT05000"
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="并行进程数，默认为CPU核心数"
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
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=8.0,
        help="最小长宽比（流星应该很细长），默认8.0"
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=15.0,
        help="最大线条宽度（像素），默认15"
    )
    parser.add_argument(
        "--exclude-bottom",
        type=float,
        default=0.1,
        help="排除图像底部的比例 (0-1)，默认0.1表示排除底部10%%区域"
    )
    parser.add_argument(
        "--min-angle",
        type=float,
        default=3.0,
        help="最小角度（度），排除接近水平的线条，默认3度"
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
        "min_aspect_ratio": args.min_aspect_ratio,
        "max_line_width": args.max_width,
        "exclude_bottom": args.exclude_bottom,
        "min_angle": args.min_angle,
        "start_from": args.start_from,
        "end_at": args.end_at,
        "workers": args.workers or multiprocessing.cpu_count(),
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
    print(f"  最小长宽比: {config['min_aspect_ratio']}")
    print(f"  最大线条宽度: {config['max_line_width']}px")
    print(f"  排除底部区域: {config['exclude_bottom']*100:.0f}%")
    print(f"  最小角度: {config['min_angle']}°")
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
