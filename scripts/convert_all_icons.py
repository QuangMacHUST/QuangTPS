#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chuyển đổi tất cả biểu tượng SVG trong dự án thành PNG.
Chạy script này để đảm bảo tất cả biểu tượng được chuyển đổi đúng.
"""

import os
import sys
from pathlib import Path

# Thêm thư mục cha vào sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import hàm chuyển đổi
try:
    from scripts.convert_svg_to_png import convert_svg_to_png
except ImportError:
    print("Lỗi: Không thể import hàm convert_svg_to_png")
    print("Vui lòng đảm bảo file convert_svg_to_png.py nằm trong thư mục scripts")
    sys.exit(1)

def main():
    """Hàm chính để chuyển đổi tất cả biểu tượng"""
    # Đường dẫn thư mục gốc
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Thư mục chứa biểu tượng SVG
    icons_dir = root_dir / "quangtps" / "ui" / "icons" / "new_icons"
    
    if not icons_dir.exists():
        print(f"Thư mục biểu tượng không tồn tại: {icons_dir}")
        # Tạo thư mục nếu không tồn tại
        icons_dir.mkdir(parents=True, exist_ok=True)
        print(f"Đã tạo thư mục: {icons_dir}")
    
    # Tìm tất cả file SVG trong thư mục
    svg_files = list(icons_dir.glob("*.svg"))
    
    if not svg_files:
        print(f"Không tìm thấy file SVG nào trong thư mục: {icons_dir}")
        return 1
    
    print(f"Tìm thấy {len(svg_files)} file SVG để chuyển đổi.")
    
    # Chuyển đổi từng file
    for svg_file in svg_files:
        png_file = svg_file.with_suffix('.png')
        print(f"Đang chuyển đổi: {svg_file.name} -> {png_file.name}")
        
        try:
            # Biểu tượng thông thường
            convert_svg_to_png(str(svg_file), str(png_file))
            
            # Biểu tượng lớn (nếu là splash.svg)
            if svg_file.name == "splash.svg":
                splash_png = icons_dir / "splash.png"
                convert_svg_to_png(str(svg_file), str(splash_png), width=800, height=500)
                print(f"Đã tạo splash screen: {splash_png}")
        except Exception as e:
            print(f"Lỗi khi chuyển đổi {svg_file.name}: {str(e)}")
    
    print("Hoàn tất chuyển đổi biểu tượng.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 