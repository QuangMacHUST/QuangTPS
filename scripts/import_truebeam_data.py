#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script dòng lệnh để nhập mô hình chùm tia TrueBeam vào hệ thống QuangTPS.
Nhập các mô hình đã xử lý và tạo các mô hình chùm tia cần thiết trong hệ thống.

Sử dụng:
    python import_truebeam_data.py --source-dir /đường/dẫn/đến/mô_hình_chùm_tia --target-dir /đường/dẫn/đến/thư_mục_hệ_thống [--energy 6MV 10FFF] [--convert-format]
"""

import os
import sys
import argparse
import logging
import json
import shutil
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from quangtps.treatment.beams.truebeam_data_processor import TrueBeamDataProcessor
from quangtps.core.exceptions import BeamDataError
from quangtps.core.config import Config
from quangtps.core.logging import setup_logger

def main():
    """Hàm chính nhập mô hình chùm tia TrueBeam vào hệ thống"""
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Nhập mô hình chùm tia TrueBeam vào hệ thống QuangTPS')
    parser.add_argument('--source-dir', type=str, help='Thư mục chứa mô hình chùm tia đã xử lý')
    parser.add_argument('--target-dir', type=str, help='Thư mục đích trong hệ thống (mặc định: thư mục mô hình chùm tia trong cấu hình)')
    parser.add_argument('--energy', type=str, nargs='+', help='Danh sách các năng lượng cần nhập (ví dụ: 6MV 10FFF)')
    parser.add_argument('--convert-format', action='store_true', help='Chuyển đổi mô hình sang định dạng lập kế hoạch')
    parser.add_argument('--verbose', action='store_true', help='Hiển thị thông tin chi tiết')
    
    args = parser.parse_args()
    
    # Thiết lập logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level)
    logger = logging.getLogger(__name__)
    
    # Lấy thư mục đích từ cấu hình nếu không được chỉ định
    config = Config()
    target_dir = args.target_dir or config.get_path('BEAM_MODEL_DIR')
    
    if not target_dir:
        logger.error("Không thể xác định thư mục đích. Vui lòng chỉ định --target-dir")
        sys.exit(1)
    
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(target_dir, exist_ok=True)
    
    # Tạo đối tượng xử lý
    processor = TrueBeamDataProcessor(target_dir)
    
    try:
        # Nếu có thư mục nguồn, sao chép các tệp từ đó
        if args.source_dir:
            if not os.path.exists(args.source_dir):
                logger.error(f"Thư mục nguồn không tồn tại: {args.source_dir}")
                sys.exit(1)
            
            # Tìm tất cả các tệp mô hình chùm tia trong thư mục nguồn
            source_files = []
            for file in os.listdir(args.source_dir):
                if file.endswith('.json') and 'TrueBeam' in file:
                    source_path = os.path.join(args.source_dir, file)
                    source_files.append(source_path)
            
            if not source_files:
                logger.error(f"Không tìm thấy tệp mô hình chùm tia trong thư mục {args.source_dir}")
                sys.exit(1)
            
            # Lọc theo năng lượng nếu có
            if args.energy:
                filtered_files = []
                for file_path in source_files:
                    file_name = os.path.basename(file_path)
                    for energy in args.energy:
                        if energy in file_name:
                            filtered_files.append(file_path)
                            break
                
                source_files = filtered_files
                
                if not source_files:
                    logger.error(f"Không tìm thấy các năng lượng chỉ định trong thư mục {args.source_dir}")
                    sys.exit(1)
            
            # Sao chép các tệp vào thư mục đích
            logger.info(f"Sao chép {len(source_files)} mô hình chùm tia vào {target_dir}...")
            for source_path in source_files:
                file_name = os.path.basename(source_path)
                target_path = os.path.join(target_dir, file_name)
                
                # Sao chép tệp
                shutil.copy2(source_path, target_path)
                logger.info(f"Đã sao chép {file_name} vào {target_dir}")
        
        # Tải các mô hình chùm tia
        beam_models = processor.load_beam_models()
        
        if not beam_models:
            logger.warning(f"Không tìm thấy mô hình chùm tia trong thư mục {target_dir}")
            sys.exit(0)
        
        # Lọc theo năng lượng nếu có
        if args.energy:
            beam_models = {k: v for k, v in beam_models.items() if k in args.energy}
            
            if not beam_models:
                logger.error(f"Không tìm thấy các năng lượng chỉ định trong thư mục {target_dir}")
                sys.exit(1)
        
        logger.info(f"Đã tải {len(beam_models)} mô hình chùm tia: {', '.join(beam_models.keys())}")
        
        # Chuyển đổi sang định dạng lập kế hoạch nếu yêu cầu
        if args.convert_format:
            planning_dir = os.path.join(target_dir, "planning_data")
            os.makedirs(planning_dir, exist_ok=True)
            
            logger.info(f"Chuyển đổi mô hình chùm tia sang định dạng lập kế hoạch trong {planning_dir}...")
            
            for energy_name, beam_model in beam_models.items():
                try:
                    output_path = processor.convert_to_treatment_planning_format(energy_name, planning_dir)
                    logger.info(f"Đã chuyển đổi {energy_name} thành {output_path}")
                except Exception as e:
                    logger.error(f"Lỗi khi chuyển đổi {energy_name}: {str(e)}")
        
        # Kiểm tra và thông báo kết quả
        available_energies = processor.get_available_energies()
        logger.info(f"Đã nhập thành công các mô hình chùm tia: {', '.join(available_energies)}")
        logger.info(f"Các mô hình đã sẵn sàng để sử dụng trong hệ thống QuangTPS")
        
    except BeamDataError as e:
        logger.error(f"Lỗi xử lý dữ liệu chùm tia: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 