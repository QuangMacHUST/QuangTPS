#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script dòng lệnh để xử lý dữ liệu chùm tia TrueBeam từ tệp Excel.
Cho phép tự động chuyển đổi dữ liệu chùm tia từ định dạng Excel cung cấp bởi Varian 
sang mô hình chùm tia sử dụng trong hệ thống QuangTPS.

Sử dụng:
    python process_truebeam_data.py --input-dir /đường/dẫn/đến/thư_mục_excel --output-dir /đường/dẫn/đến/thư_mục_đầu_ra [--energy 6MV 10FFF] [--visualize]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from quangtps.treatment.beams.truebeam_data_processor import TrueBeamDataReader
from quangtps.core.exceptions import BeamDataError
from quangtps.core.logging import setup_logger

def main():
    """Hàm chính xử lý dữ liệu chùm tia TrueBeam"""
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Xử lý dữ liệu chùm tia TrueBeam từ tệp Excel')
    parser.add_argument('--input-dir', type=str, required=True, help='Thư mục chứa tệp Excel dữ liệu chùm tia')
    parser.add_argument('--output-dir', type=str, required=True, help='Thư mục đầu ra cho mô hình chùm tia')
    parser.add_argument('--energy', type=str, nargs='+', help='Danh sách các năng lượng cần xử lý (ví dụ: 6MV 10FFF)')
    parser.add_argument('--visualize', action='store_true', help='Tạo đồ thị trực quan hóa mô hình chùm tia')
    parser.add_argument('--verbose', action='store_true', help='Hiển thị thông tin chi tiết')
    
    args = parser.parse_args()
    
    # Thiết lập logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level)
    logger = logging.getLogger(__name__)
    
    # Kiểm tra thư mục đầu vào
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        logger.error(f"Thư mục đầu vào không tồn tại: {input_dir}")
        sys.exit(1)
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo đối tượng đọc dữ liệu
    reader = TrueBeamDataReader()
    
    try:
        # Quét thư mục để tìm các file dữ liệu
        logger.info(f"Quét thư mục {input_dir} để tìm các file dữ liệu chùm tia...")
        energy_files = reader.scan_directory(input_dir)
        
        if not energy_files:
            logger.error(f"Không tìm thấy file dữ liệu chùm tia trong thư mục {input_dir}")
            sys.exit(1)
        
        # Lọc theo danh sách năng lượng nếu có
        if args.energy:
            logger.info(f"Lọc các năng lượng: {', '.join(args.energy)}")
            energy_files = {k: v for k, v in energy_files.items() if k in args.energy}
            
            if not energy_files:
                logger.error(f"Không tìm thấy các năng lượng chỉ định trong thư mục {input_dir}")
                sys.exit(1)
        
        logger.info(f"Tìm thấy {len(energy_files)} năng lượng: {', '.join(energy_files.keys())}")
        
        # Xử lý từng năng lượng
        for energy_name, file_path in energy_files.items():
            try:
                logger.info(f"Đang xử lý năng lượng {energy_name} từ {file_path}...")
                
                # Đọc dữ liệu
                beam_data = reader.read_beam_data(file_path)
                
                # Tạo mô hình chùm tia
                beam_model = reader.create_beam_model(beam_data)
                
                # Lưu vào file
                output_path = os.path.join(output_dir, f"TrueBeam_{energy_name}_beam_model.json")
                beam_model.save_to_json(output_path)
                logger.info(f"Đã lưu mô hình chùm tia {energy_name} vào {output_path}")
                
                # Tạo đồ thị trực quan hóa nếu yêu cầu
                if args.visualize:
                    logger.info(f"Tạo đồ thị trực quan hóa cho năng lượng {energy_name}...")
                    reader._visualize_beam_model(beam_model, output_dir)
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý năng lượng {energy_name}: {str(e)}")
                if args.verbose:
                    import traceback
                    logger.debug(traceback.format_exc())
        
        logger.info("Hoàn thành xử lý dữ liệu chùm tia")
        
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