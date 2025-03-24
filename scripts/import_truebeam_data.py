#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script nhập dữ liệu chùm tia từ máy gia tốc TrueBeam.

Script này nhập dữ liệu chùm tia từ các file Excel của máy gia tốc TrueBeam,
và lưu dữ liệu vào định dạng JSON để sử dụng trong hệ thống QuangTPS.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Thêm thư mục cha vào sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import các module cần thiết
try:
    from quangtps.treatment.beams.beam_data_importer import TrueBeamDataReader, import_truebeam_data
except ImportError as e:
    print(f"Lỗi: Không thể import module cần thiết: {e}")
    sys.exit(1)

def setup_logging(verbose: bool = False):
    """Thiết lập logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('truebeam_import.log')
        ]
    )

def parse_arguments():
    """Phân tích đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Nhập dữ liệu chùm tia từ máy gia tốc TrueBeam')
    
    parser.add_argument('--source', '-s', type=str, 
                        help='Thư mục chứa dữ liệu TrueBeam dạng Excel')
    
    parser.add_argument('--dest', '-d', type=str,
                        help='Thư mục đích để lưu dữ liệu JSON')
    
    parser.add_argument('--energies', '-e', type=str,
                        help='Danh sách năng lượng cần nhập, cách nhau bởi dấu phẩy (ví dụ: 6MV,10FFF)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Hiển thị thông tin debug chi tiết')
    
    parser.add_argument('--list-only', '-l', action='store_true',
                        help='Chỉ liệt kê các năng lượng có sẵn, không nhập dữ liệu')
    
    return parser.parse_args()

def main():
    """Hàm chính của script."""
    # Phân tích đối số
    args = parse_arguments()
    
    # Thiết lập logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Xác định thư mục nguồn và đích
    source_dir = args.source
    if source_dir is None:
        source_dir = os.path.join(parent_dir, "data", "beam_data", "raw")
    
    dest_dir = args.dest
    if dest_dir is None:
        dest_dir = os.path.join(parent_dir, "data", "beam_data")
    
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(dest_dir, exist_ok=True)
    
    logger.info(f"Thư mục nguồn: {source_dir}")
    logger.info(f"Thư mục đích: {dest_dir}")
    
    # Tạo đối tượng TrueBeamDataReader để quét dữ liệu
    reader = TrueBeamDataReader(source_dir)
    excel_files = reader.scan_data_directory()
    
    if not excel_files:
        logger.error(f"Không tìm thấy file Excel nào trong thư mục {source_dir}")
        return 1
    
    # Lấy danh sách năng lượng có sẵn
    available_energies = reader.available_energies
    logger.info(f"Các năng lượng có sẵn: {available_energies}")
    
    # Nếu chỉ liệt kê các năng lượng, không nhập dữ liệu
    if args.list_only:
        print("\nCác năng lượng có sẵn:")
        for energy in available_energies:
            print(f"  {energy}")
        return 0
    
    # Chuyển đổi danh sách năng lượng từ chuỗi
    energies = None
    if args.energies:
        energies = [e.strip() for e in args.energies.split(',')]
        
        # Kiểm tra xem năng lượng có tồn tại không
        for energy in energies:
            if energy not in available_energies:
                logger.warning(f"Năng lượng {energy} không có trong dữ liệu, sẽ bỏ qua")
        
        # Lọc các năng lượng tồn tại
        energies = [e for e in energies if e in available_energies]
    
    if energies is None:
        energies = available_energies
    
    if not energies:
        logger.error("Không có năng lượng nào để nhập")
        return 1
    
    # Nhập dữ liệu TrueBeam và xuất sang JSON
    logger.info(f"Đang nhập dữ liệu cho các năng lượng: {energies}")
    
    result = import_truebeam_data(source_dir, energies)
    
    # In kết quả
    if result:
        print("\nKết quả nhập dữ liệu TrueBeam:")
        for energy, output_path in result.items():
            print(f"  {energy}: {output_path}")
        print(f"\nĐã nhập {len(result)} năng lượng.")
        return 0
    else:
        print("\nKhông có dữ liệu nào được nhập.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 