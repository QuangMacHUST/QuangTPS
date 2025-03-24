#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để tự động xử lý và nhập dữ liệu chùm tia TrueBeam vào hệ thống QuangTPS.
Script này giúp đơn giản hóa quy trình nhập dữ liệu chùm tia từ các file Excel cung cấp bởi Varian.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Đảm bảo đường dẫn thư mục gốc được thêm vào PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from quangtps.core.logging import setup_logger
from quangtps.core.config import Config
from quangtps.treatment.beams.truebeam_data_processor import TrueBeamDataProcessor, TrueBeamDataReader
from quangtps.dose.beam_data_processor import BeamModel

def setup_truebeam_data(source_dir=None, target_dir=None, energies=None, verbose=False):
    """
    Thiết lập dữ liệu chùm tia TrueBeam cho hệ thống QuangTPS.
    
    Parameters
    ----------
    source_dir : str, optional
        Thư mục chứa dữ liệu chùm tia TrueBeam, by default None (tự động tìm)
    target_dir : str, optional
        Thư mục lưu mô hình chùm tia, by default None (sử dụng thư mục mặc định)
    energies : list, optional
        Danh sách các năng lượng cần xử lý, by default None (xử lý tất cả)
    verbose : bool, optional
        Hiển thị thông tin chi tiết, by default False
    
    Returns
    -------
    dict
        Dictionary chứa các mô hình chùm tia đã tạo
    """
    # Thiết lập mức độ log
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logger(log_level)
    logger = logging.getLogger(__name__)
    
    # Cấu hình
    config = Config()
    
    # Xác định thư mục nguồn
    if source_dir is None:
        # Tìm thư mục dữ liệu chùm tia TrueBeam
        for possible_dir in [
            os.path.join(root_dir, 'Truebeam representative Beam Data for eclipse'),
            os.path.join(root_dir, 'data', 'beam_data', 'truebeam')
        ]:
            if os.path.exists(possible_dir):
                source_dir = possible_dir
                logger.info(f"Tìm thấy thư mục dữ liệu TrueBeam: {source_dir}")
                break
    
    if source_dir is None or not os.path.exists(source_dir):
        logger.error("Không tìm thấy thư mục dữ liệu TrueBeam. Vui lòng chỉ định thư mục nguồn.")
        return {}
    
    # Xác định thư mục đích
    if target_dir is None:
        target_dir = config.get_path('BEAM_MODEL_DIR')
        if not target_dir:
            target_dir = os.path.join(root_dir, 'data', 'beam_data', 'models')
    
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(target_dir, exist_ok=True)
    
    # Tạo đối tượng xử lý dữ liệu
    processor = TrueBeamDataProcessor(target_dir)
    reader = TrueBeamDataReader()
    
    # Quét thư mục nguồn để tìm các file dữ liệu
    energy_files = reader.scan_directory(source_dir)
    
    if not energy_files:
        logger.error(f"Không tìm thấy file dữ liệu chùm tia trong thư mục {source_dir}")
        return {}
    
    logger.info(f"Tìm thấy {len(energy_files)} file dữ liệu chùm tia: {', '.join(energy_files.keys())}")
    
    # Lọc theo danh sách năng lượng nếu có
    if energies:
        energy_files = {k: v for k, v in energy_files.items() if k in energies}
        logger.info(f"Lọc theo năng lượng: {', '.join(energies)}")
    
    # Xử lý từng năng lượng
    beam_models = {}
    for energy_name, file_path in energy_files.items():
        try:
            logger.info(f"Đang xử lý năng lượng {energy_name} từ {file_path}...")
            
            # Đọc dữ liệu
            beam_data = reader.read_beam_data(file_path)
            
            # Tạo mô hình chùm tia
            beam_model = reader.create_beam_model(beam_data)
            
            # Lưu vào dictionary
            beam_models[energy_name] = beam_model
            
            # Lưu vào file
            output_path = os.path.join(target_dir, f"TrueBeam_{energy_name}_beam_model.json")
            beam_model.save_to_json(output_path)
            logger.info(f"Đã lưu mô hình chùm tia {energy_name} vào {output_path}")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý năng lượng {energy_name}: {str(e)}")
            if verbose:
                import traceback
                logger.debug(traceback.format_exc())
    
    if beam_models:
        logger.info(f"Đã tạo thành công {len(beam_models)} mô hình chùm tia: {', '.join(beam_models.keys())}")
    else:
        logger.warning("Không tạo được mô hình chùm tia nào.")
    
    return beam_models

def main():
    """Hàm chính để chạy từ dòng lệnh"""
    parser = argparse.ArgumentParser(description="Thiết lập dữ liệu chùm tia TrueBeam cho QuangTPS")
    parser.add_argument("--source-dir", type=str, help="Thư mục chứa dữ liệu chùm tia TrueBeam")
    parser.add_argument("--target-dir", type=str, help="Thư mục lưu mô hình chùm tia")
    parser.add_argument("--energy", type=str, nargs="+", help="Danh sách các năng lượng cần xử lý (ví dụ: 6MV 10FFF)")
    parser.add_argument("--verbose", action="store_true", help="Hiển thị thông tin chi tiết")
    
    args = parser.parse_args()
    
    setup_truebeam_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        energies=args.energy,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 