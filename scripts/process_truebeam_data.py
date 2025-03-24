#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script xử lý dữ liệu TrueBeam và tạo mô hình chùm tia.

Script này quét thư mục dữ liệu chùm tia, tìm các file Excel chứa dữ liệu TrueBeam,
và tạo các mô hình chùm tia từ dữ liệu này.
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
    from quangtps.treatment.beams.beam_data_processor import BeamDataProcessor
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
            logging.FileHandler('truebeam_processing.log')
        ]
    )

def parse_arguments():
    """Phân tích đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Xử lý dữ liệu TrueBeam và tạo mô hình chùm tia')
    
    parser.add_argument('--source', '-s', type=str, 
                        help='Thư mục chứa dữ liệu TrueBeam dạng Excel')
    
    parser.add_argument('--dest', '-d', type=str,
                        help='Thư mục đích để lưu dữ liệu đã xử lý')
    
    parser.add_argument('--energies', '-e', type=str,
                        help='Danh sách năng lượng cần xử lý, cách nhau bởi dấu phẩy (ví dụ: 6MV,10FFF)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Hiển thị thông tin debug chi tiết')
    
    return parser.parse_args()

def process_truebeam_data(source_dir: str = None, dest_dir: str = None, 
                         energies: List[str] = None, verbose: bool = False) -> Dict[str, str]:
    """
    Xử lý dữ liệu TrueBeam và tạo mô hình chùm tia.
    
    Args:
        source_dir (str, optional): Thư mục chứa dữ liệu TrueBeam dạng Excel.
        dest_dir (str, optional): Thư mục đích để lưu dữ liệu đã xử lý.
        energies (List[str], optional): Danh sách năng lượng cần xử lý.
        verbose (bool, optional): Hiển thị thông tin debug chi tiết.
        
    Returns:
        Dict[str, str]: Dictionary chứa {năng lượng: đường dẫn file mô hình}.
    """
    # Thiết lập logging
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Xác định thư mục nguồn và đích
    if source_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source_dir = os.path.join(base_dir, "data", "beam_data", "raw")
    
    if dest_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest_dir = os.path.join(base_dir, "data", "beam_data")
    
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(dest_dir, exist_ok=True)
    
    logger.info(f"Thư mục nguồn: {source_dir}")
    logger.info(f"Thư mục đích: {dest_dir}")
    
    # Tạo đối tượng TrueBeamDataReader để quét dữ liệu
    reader = TrueBeamDataReader(source_dir)
    excel_files = reader.scan_data_directory()
    
    if not excel_files:
        logger.error(f"Không tìm thấy file Excel nào trong thư mục {source_dir}")
        return {}
    
    # Lấy danh sách năng lượng có sẵn
    available_energies = reader.available_energies
    logger.info(f"Các năng lượng có sẵn: {available_energies}")
    
    # Nếu chỉ định năng lượng cụ thể, chỉ xử lý những năng lượng đó
    if energies:
        # Kiểm tra xem năng lượng có tồn tại không
        for energy in energies:
            if energy not in available_energies:
                logger.warning(f"Năng lượng {energy} không có trong dữ liệu, sẽ bỏ qua")
        
        # Lọc các năng lượng tồn tại
        process_energies = [e for e in energies if e in available_energies]
    else:
        process_energies = available_energies
    
    if not process_energies:
        logger.error("Không có năng lượng nào để xử lý")
        return {}
    
    # Nhập dữ liệu TrueBeam và xuất sang JSON
    logger.info(f"Đang nhập dữ liệu cho các năng lượng: {process_energies}")
    json_files = import_truebeam_data(source_dir, process_energies)
    
    # Tạo mô hình chùm tia từ dữ liệu JSON
    processor = BeamDataProcessor(dest_dir)
    model_files = {}
    
    for energy in process_energies:
        try:
            # Tải mô hình
            model = processor.load_beam_model(energy)
            
            # Xuất mô hình
            model_file = processor.export_beam_model(energy, dest_dir)
            model_files[energy] = model_file
            
            logger.info(f"Đã tạo mô hình chùm tia cho năng lượng {energy}: {model_file}")
        except Exception as e:
            logger.error(f"Lỗi khi tạo mô hình chùm tia cho năng lượng {energy}: {str(e)}")
    
    return model_files

def main():
    """Hàm chính của script."""
    # Phân tích đối số
    args = parse_arguments()
    
    # Chuyển đổi danh sách năng lượng từ chuỗi
    energies = None
    if args.energies:
        energies = [e.strip() for e in args.energies.split(',')]
    
    # Xử lý dữ liệu
    result = process_truebeam_data(
        source_dir=args.source,
        dest_dir=args.dest,
        energies=energies,
        verbose=args.verbose
    )
    
    # In kết quả
    if result:
        print("\nKết quả xử lý dữ liệu TrueBeam:")
        for energy, model_file in result.items():
            print(f"  {energy}: {model_file}")
        print(f"\nĐã xử lý {len(result)} mô hình chùm tia.")
        return 0
    else:
        print("\nKhông có mô hình chùm tia nào được tạo.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 