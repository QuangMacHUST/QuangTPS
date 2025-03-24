#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để import dữ liệu chùm tia từ TrueBeam vào QuangTPS.

Script này cho phép import các mô hình chùm tia từ các file Excel của TrueBeam
vào hệ thống QuangTPS để sử dụng trong tính toán liều.

Sử dụng:
    python import_truebeam_data.py --source <thư mục chứa file Excel> [--energy <năng lượng>] [--force]

Các tham số:
    --source: Thư mục chứa file Excel của TrueBeam
    --energy: Chỉ import file có năng lượng này (mặc định: import tất cả)
    --force: Ghi đè mô hình đã tồn tại
"""

import os
import sys
import argparse
import logging
import subprocess

# Thêm thư mục gốc vào sys.path để import các module của QuangTPS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quangtps.common.paths import get_beam_data_dir, get_project_root
from quangtps.core.logging import setup_logger

# Thiết lập logging
logger = logging.getLogger(__name__)


def setup_argparse():
    """Thiết lập parser tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Import dữ liệu chùm tia TrueBeam vào QuangTPS.')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Thư mục chứa file Excel của TrueBeam')
    
    parser.add_argument('--energy', type=str, default=None,
                        help='Chỉ import file có năng lượng này (mặc định: import tất cả)')
    
    parser.add_argument('--force', action='store_true',
                        help='Ghi đè mô hình đã tồn tại')
    
    return parser


def main():
    """Hàm chính của script."""
    # Thiết lập logging
    log_dir = os.path.join(get_project_root(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir, level=logging.INFO)
    
    # Parse tham số dòng lệnh
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Xác định đường dẫn đến script process_truebeam_data.py
    script_path = os.path.join(os.path.dirname(__file__), 'process_truebeam_data.py')
    
    # Tạo command line
    cmd = [sys.executable, script_path, '--source', args.source]
    
    if args.energy:
        cmd.extend(['--energy', args.energy])
        
    if args.force:
        cmd.append('--force')
        
    # Gọi script process_truebeam_data.py
    logger.info(f"Chạy script process_truebeam_data.py với các tham số: {' '.join(cmd[2:])}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # In ra output
        logger.info(result.stdout)
        
        # Kiểm tra output để xác định số mô hình đã xử lý
        success_line = [line for line in result.stdout.split('\n') if 'Đã xử lý thành công' in line]
        
        if success_line:
            try:
                # Phân tích số lượng đã xử lý
                success_count = int(success_line[0].split('/')[0].split()[-1])
                total_count = int(success_line[0].split('/')[1].split()[0])
                
                if success_count > 0:
                    logger.info(f"\nĐã import thành công {success_count}/{total_count} mô hình chùm tia vào QuangTPS.")
                    return 0
                else:
                    logger.warning("\nKhông có mô hình nào được import thành công.")
                    return 1
            except:
                logger.error("\nKhông thể xác định số lượng mô hình đã import.")
                return 1
        else:
            logger.error("\nKhông tìm thấy thông tin về kết quả import.")
            return 1
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi gọi script process_truebeam_data.py: {str(e)}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 