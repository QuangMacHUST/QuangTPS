#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chuyển đổi biểu tượng SVG thành PNG.

Script này sử dụng thư viện cairosvg để chuyển đổi tệp SVG sang PNG
với chất lượng cao, phù hợp cho sử dụng trong giao diện người dùng.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Thiết lập logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def check_dependencies():
    """Kiểm tra các thư viện phụ thuộc."""
    try:
        import cairosvg
        return True
    except ImportError:
        print("Lỗi: Thư viện cairosvg không được cài đặt.")
        print("Vui lòng cài đặt thư viện bằng lệnh: pip install cairosvg")
        return False

def convert_svg_to_png(svg_path: str, png_path: str, 
                      width: Optional[int] = None, 
                      height: Optional[int] = None,
                      scale: float = 1.0) -> bool:
    """
    Chuyển đổi tệp SVG thành PNG.
    
    Args:
        svg_path (str): Đường dẫn đến tệp SVG nguồn.
        png_path (str): Đường dẫn đến tệp PNG đích.
        width (int, optional): Chiều rộng của hình ảnh PNG (điểm ảnh).
        height (int, optional): Chiều cao của hình ảnh PNG (điểm ảnh).
        scale (float, optional): Tỷ lệ chuyển đổi (mặc định: 1.0).
        
    Returns:
        bool: True nếu chuyển đổi thành công, False nếu thất bại.
    """
    try:
        # Import cairosvg trong hàm để tránh lỗi khi không có thư viện
        import cairosvg
        
        # Kiểm tra đường dẫn tệp nguồn
        if not os.path.exists(svg_path):
            logger.error(f"Tệp SVG không tồn tại: {svg_path}")
            return False
        
        # Tạo thư mục đích nếu chưa tồn tại
        png_dir = os.path.dirname(png_path)
        if png_dir and not os.path.exists(png_dir):
            os.makedirs(png_dir, exist_ok=True)
        
        # Chuyển đổi SVG sang PNG
        cairosvg.svg2png(url=svg_path, write_to=png_path, 
                        output_width=width, output_height=height,
                        scale=scale)
        
        logger.info(f"Đã chuyển đổi {svg_path} -> {png_path}")
        return True
    
    except ImportError:
        logger.error("Lỗi: Thư viện cairosvg không được cài đặt.")
        return False
    
    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi {svg_path}: {str(e)}")
        return False

def find_svg_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Tìm tất cả các tệp SVG trong một thư mục.
    
    Args:
        directory (str): Thư mục cần tìm.
        recursive (bool, optional): Tìm kiếm đệ quy trong các thư mục con.
        
    Returns:
        List[str]: Danh sách đường dẫn đến các tệp SVG.
    """
    svg_files = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.svg'):
                    svg_files.append(os.path.join(root, file))
    else:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.lower().endswith('.svg'):
                svg_files.append(entry.path)
    
    return svg_files

def parse_arguments():
    """Phân tích đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description='Chuyển đổi biểu tượng SVG thành PNG'
    )
    
    parser.add_argument('--svg', '-s', type=str,
                       help='Đường dẫn đến tệp SVG đầu vào')
    
    parser.add_argument('--png', '-p', type=str,
                       help='Đường dẫn đến tệp PNG đầu ra')
    
    parser.add_argument('--width', '-W', type=int,
                       help='Chiều rộng của hình ảnh PNG (điểm ảnh)')
    
    parser.add_argument('--height', '-H', type=int,
                       help='Chiều cao của hình ảnh PNG (điểm ảnh)')
    
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Tỷ lệ chuyển đổi (mặc định: 1.0)')
    
    parser.add_argument('--dir', '-d', type=str,
                       help='Thư mục chứa các tệp SVG cần chuyển đổi')
    
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Tìm kiếm đệ quy trong các thư mục con')
    
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Thư mục đầu ra cho các tệp PNG')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Hiển thị thông tin chi tiết')
    
    return parser.parse_args()

def main():
    """Hàm chính của script."""
    # Phân tích đối số
    args = parse_arguments()
    
    # Thiết lập logging
    setup_logging(args.verbose)
    
    # Kiểm tra các thư viện phụ thuộc
    if not check_dependencies():
        return 1
    
    # Chuyển đổi một tệp SVG
    if args.svg:
        if not args.png:
            # Tạo tên file PNG từ tên file SVG
            args.png = os.path.splitext(args.svg)[0] + '.png'
        
        success = convert_svg_to_png(
            args.svg, args.png, args.width, args.height, args.scale
        )
        
        if not success:
            return 1
    
    # Chuyển đổi nhiều tệp SVG trong một thư mục
    elif args.dir:
        # Tìm tất cả các tệp SVG
        svg_files = find_svg_files(args.dir, args.recursive)
        
        if not svg_files:
            logger.error(f"Không tìm thấy tệp SVG nào trong thư mục: {args.dir}")
            return 1
        
        logger.info(f"Tìm thấy {len(svg_files)} tệp SVG để chuyển đổi")
        
        # Xác định thư mục đầu ra
        output_dir = args.output_dir
        
        # Chuyển đổi từng tệp
        success_count = 0
        for svg_path in svg_files:
            # Tạo tên file PNG từ tên file SVG
            if output_dir:
                svg_name = os.path.basename(svg_path)
                png_name = os.path.splitext(svg_name)[0] + '.png'
                png_path = os.path.join(output_dir, png_name)
            else:
                png_path = os.path.splitext(svg_path)[0] + '.png'
            
            # Chuyển đổi tệp
            success = convert_svg_to_png(
                svg_path, png_path, args.width, args.height, args.scale
            )
            
            if success:
                success_count += 1
        
        logger.info(f"Đã chuyển đổi thành công {success_count}/{len(svg_files)} tệp")
        
        if success_count < len(svg_files):
            return 1
    
    else:
        logger.error("Vui lòng chỉ định tệp SVG hoặc thư mục chứa các tệp SVG")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 