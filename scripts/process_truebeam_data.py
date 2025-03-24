#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để xử lý dữ liệu chùm tia TrueBeam từ dòng lệnh.

Script này cho phép nhập và xử lý dữ liệu chùm tia từ các file Excel của TrueBeam
và chuyển đổi thành mô hình chùm tia để sử dụng trong QuangTPS.

Sử dụng:
    python process_truebeam_data.py --source <thư mục chứa file Excel> [--dest <thư mục đích>] [--energy <năng lượng>]

Các tham số:
    --source: Thư mục chứa file Excel của TrueBeam
    --dest: Thư mục đích để lưu mô hình chùm tia (mặc định: data/beam_data)
    --energy: Chỉ xử lý file có năng lượng này (mặc định: xử lý tất cả)
"""

import os
import sys
import argparse
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Thêm thư mục gốc vào sys.path để import các module của QuangTPS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quangtps.treatment.beams.truebeam_data_processor import TrueBeamDataProcessor
from quangtps.dose.beam_data_processor import BeamModel
from quangtps.common.paths import get_beam_data_dir, get_project_root
from quangtps.core.logging import setup_logger

# Thiết lập logging
logger = logging.getLogger(__name__)

# Thêm console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def setup_argparse():
    """Thiết lập parser tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Xử lý dữ liệu chùm tia TrueBeam từ file Excel.')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Thư mục chứa file Excel của TrueBeam')
    
    parser.add_argument('--dest', type=str, default=None,
                        help='Thư mục đích để lưu mô hình chùm tia (mặc định: data/beam_data)')
    
    parser.add_argument('--energy', type=str, default=None,
                        help='Chỉ xử lý file có năng lượng này (mặc định: xử lý tất cả)')
    
    parser.add_argument('--force', action='store_true',
                        help='Ghi đè mô hình đã tồn tại')
    
    return parser


def find_excel_files(source_dir: str, energy_filter: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    Tìm tất cả các file Excel trong thư mục và xác định năng lượng.
    
    Parameters
    ----------
    source_dir : str
        Thư mục chứa file Excel
    energy_filter : str, optional
        Lọc theo năng lượng
        
    Returns
    -------
    List[Tuple[str, str, str]]
        Danh sách các tuple (đường dẫn file, loại chùm tia, năng lượng)
    """
    if not os.path.isdir(source_dir):
        logger.error(f"Thư mục không tồn tại: {source_dir}")
        return []
        
    excel_files = []
    processor = TrueBeamDataProcessor()  # Khởi tạo processor để phân tích tên file
    
    for file in os.listdir(source_dir):
        if file.endswith(('.xlsx', '.xls')) and not file.startswith('~$') and not file.startswith('._'):
            file_path = os.path.join(source_dir, file)
            
            # Xác định loại năng lượng từ tên file
            beam_type, energy = processor._determine_beam_type_from_filename(file_path)
            
            # Lọc theo năng lượng nếu có
            if energy_filter and energy_filter.lower() != energy.lower():
                continue
                
            # Thêm vào danh sách
            excel_files.append((file_path, beam_type, energy))
            
    # Sắp xếp file theo năng lượng
    excel_files.sort(key=lambda x: x[2])
    
    return excel_files


def process_excel_file(file_path: str, processor: TrueBeamDataProcessor) -> Optional[BeamModel]:
    """
    Xử lý file Excel và tạo mô hình chùm tia.
    
    Parameters
    ----------
    file_path : str
        Đường dẫn đến file Excel
    processor : TrueBeamDataProcessor
        Processor đã khởi tạo
        
    Returns
    -------
    Optional[BeamModel]
        Mô hình chùm tia, hoặc None nếu không thể tạo
    """
    logger.info(f"Đang xử lý file: {file_path}")
    
    # Đọc file Excel
    logger.info("Đang đọc dữ liệu...")
    success = processor.read_excel_file(file_path)
    
    if not success:
        logger.error("Không thể đọc file Excel.")
        return None
        
    # Tạo mô hình chùm tia
    logger.info("Đang tạo mô hình chùm tia...")
    beam_model = processor.create_beam_model()
    
    if beam_model is None:
        logger.error("Không thể tạo mô hình chùm tia.")
        return None
        
    logger.info(f"Đã tạo mô hình chùm tia: {beam_model.name}")
    
    return beam_model


def save_beam_model(beam_model: BeamModel, dest_dir: Optional[str] = None) -> str:
    """
    Lưu mô hình chùm tia vào file.
    
    Parameters
    ----------
    beam_model : BeamModel
        Mô hình chùm tia
    dest_dir : str, optional
        Thư mục đích
        
    Returns
    -------
    str
        Đường dẫn đến file đã lưu
    """
    # Xác định thư mục đích
    if dest_dir is None:
        beam_data_dir = get_beam_data_dir()
    else:
        beam_data_dir = dest_dir
        
    # Tạo thư mục models nếu chưa tồn tại
    models_dir = os.path.join(beam_data_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Tạo tên file
    file_name = f"TrueBeam_{beam_model.energy.replace(' ', '_')}.json"
    file_path = os.path.join(models_dir, file_name)
    
    # Lưu mô hình
    beam_model.save_to_file(file_path)
    logger.info(f"Đã lưu mô hình chùm tia vào: {file_path}")
    
    return file_path


def main():
    """Hàm chính của script."""
    # Thiết lập logging
    log_dir = os.path.join(get_project_root(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir, level=logging.INFO)
    
    # Parse tham số dòng lệnh
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Xác định thư mục đích
    dest_dir = args.dest or get_beam_data_dir()
    os.makedirs(dest_dir, exist_ok=True)
    
    # Khởi tạo processor
    processor = TrueBeamDataProcessor(dest_dir)
    
    # Tìm các file Excel
    logger.info(f"Đang tìm file Excel trong thư mục: {args.source}")
    excel_files = find_excel_files(args.source, args.energy)
    
    if not excel_files:
        logger.error("Không tìm thấy file Excel phù hợp.")
        return 1
        
    logger.info(f"Đã tìm thấy {len(excel_files)} file Excel phù hợp:")
    for file_path, beam_type, energy in excel_files:
        logger.info(f"  - {os.path.basename(file_path)}: {beam_type} {energy}")
    
    # Xử lý từng file
    success_count = 0
    
    for file_path, beam_type, energy in excel_files:
        logger.info(f"\n{'='*80}\nĐang xử lý file {os.path.basename(file_path)} ({energy})...")
        
        # Kiểm tra xem mô hình đã tồn tại chưa
        model_file = os.path.join(dest_dir, 'models', f"TrueBeam_{energy.replace(' ', '_')}.json")
        if os.path.exists(model_file) and not args.force:
            logger.info(f"Mô hình đã tồn tại: {model_file}, bỏ qua. Sử dụng --force để ghi đè.")
            continue
        
        # Xử lý file
        beam_model = process_excel_file(file_path, processor)
        
        if beam_model:
            # Lưu mô hình
            save_path = save_beam_model(beam_model, dest_dir)
            success_count += 1
            
    logger.info(f"\n{'='*80}\nĐã xử lý thành công {success_count}/{len(excel_files)} file.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 