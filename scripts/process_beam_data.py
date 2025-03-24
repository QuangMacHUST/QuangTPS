#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để xử lý dữ liệu chùm tia TrueBeam và tạo mô hình cho các năng lượng.

Script này đọc các file Excel của TrueBeam, trích xuất dữ liệu, và tạo mô hình chùm tia
để sử dụng trong QuangTPS.
"""

import os
import sys
import json
import glob
import logging
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các module của QuangTPS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quangtps.dose.beam_data_processor import BeamModel

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('beam_data_processing.log')
    ]
)

logger = logging.getLogger(__name__)


def get_project_root():
    """Lấy đường dẫn đến thư mục gốc của dự án."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    return project_root


def get_beam_data_dir():
    """Lấy đường dẫn đến thư mục lưu trữ dữ liệu chùm tia."""
    beam_data_dir = os.path.join(get_project_root(), "data", "beam_data")
    os.makedirs(beam_data_dir, exist_ok=True)
    return beam_data_dir


def determine_beam_type_from_filename(file_path):
    """
    Xác định loại chùm tia và năng lượng từ tên file.
    
    Parameters
    ----------
    file_path : str
        Đường dẫn đến file Excel
        
    Returns
    -------
    tuple
        (beam_type, energy)
    """
    filename = os.path.basename(file_path)
    
    # Xác định loại chùm tia
    if "Electron" in filename:
        beam_type = "ELECTRON"
    else:
        beam_type = "PHOTON"
    
    # Xác định năng lượng
    if "Electron" in filename:
        energy = "Electron"
    elif "6FFF" in filename:
        energy = "6FFF"
    elif "10FFF" in filename:
        energy = "10FFF"
    elif "6MV" in filename:
        energy = "6MV"
    elif "10MV" in filename:
        energy = "10MV"
    elif "15MV" in filename:
        energy = "15MV"
    elif "4MV" in filename:
        energy = "4MV"
    elif "8MV" in filename:
        energy = "8MV"
    else:
        # Nếu không xác định được, sử dụng tên file
        energy = os.path.splitext(filename)[0]
    
    return beam_type, energy


def process_excel_file(file_path):
    """
    Xử lý file Excel và tạo mô hình chùm tia.
    
    Parameters
    ----------
    file_path : str
        Đường dẫn đến file Excel
        
    Returns
    -------
    BeamModel
        Mô hình chùm tia đã tạo
    """
    logger.info(f"Xử lý file: {file_path}")
    
    # Xác định loại chùm tia và năng lượng
    beam_type, energy = determine_beam_type_from_filename(file_path)
    logger.info(f"Loại chùm tia: {beam_type}, Năng lượng: {energy}")
    
    # Tạo mô hình chùm tia mới
    model_name = f"TrueBeam_{energy}"
    beam_model = BeamModel(model_name, energy, beam_type)
    
    # Thêm thông tin nguồn
    beam_model.set_source_file(file_path)
    
    # Đọc dữ liệu PDD (nếu có)
    try:
        logger.info("Đọc dữ liệu PDD...")
        pdd_df = pd.read_excel(file_path, sheet_name='PDD', engine='openpyxl')
        pdd_data = process_pdd_sheet(pdd_df)
        
        for key, data in pdd_data.items():
            beam_model.add_parameter(f"pdd_{key}", data)
            
        logger.info(f"Đã đọc {len(pdd_data)} đường cong PDD")
    except Exception as e:
        logger.warning(f"Không thể đọc sheet PDD: {str(e)}")
    
    # Đọc dữ liệu Profile (nếu có)
    try:
        logger.info("Đọc dữ liệu Profile...")
        profile_df = pd.read_excel(file_path, sheet_name='Profiles', engine='openpyxl')
        profile_data = process_profile_sheet(profile_df)
        
        for key, data in profile_data.items():
            beam_model.add_parameter(f"profile_{key}", data)
            
        logger.info(f"Đã đọc {len(profile_data)} profile")
    except Exception as e:
        logger.warning(f"Không thể đọc sheet Profiles: {str(e)}")
    
    # Đọc dữ liệu Output Factors (nếu có)
    try:
        logger.info("Đọc dữ liệu Output Factors...")
        output_df = pd.read_excel(file_path, sheet_name='Output Factors', engine='openpyxl')
        output_factors = process_output_factors_sheet(output_df)
        
        beam_model.add_parameter("output_factors", output_factors)
        logger.info(f"Đã đọc {len(output_factors)} output factors")
    except Exception as e:
        logger.warning(f"Không thể đọc sheet Output Factors: {str(e)}")
    
    return beam_model


def process_pdd_sheet(df):
    """
    Xử lý sheet PDD.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu PDD
        
    Returns
    -------
    dict
        Dictionary chứa các đường cong PDD
    """
    pdd_data = {}
    
    if df.empty:
        return pdd_data
    
    # Lấy danh sách cột
    for column in df.columns[1:]:
        # Bỏ qua nếu không phải cột dữ liệu
        if not isinstance(column, str):
            continue
        
        # Trích xuất thông tin field size từ tên cột
        field_size = column.split()[0] if len(column.split()) > 0 else "10x10"
        
        # Lấy dữ liệu
        depths = df.iloc[:, 0].values
        values = df[column].values
        
        valid_indices = ~pd.isna(depths) & ~pd.isna(values)
        if not any(valid_indices):
            continue
            
        # Tạo dictionary dữ liệu
        pdd_values = {
            "depths": depths[valid_indices].tolist(),
            "values": values[valid_indices].tolist(),
            "field_size": field_size,
            "ssd": 100.0  # Mặc định SSD = 100 cm
        }
        
        # Thêm vào dictionary chính
        key = f"{field_size}_SSD=100"
        pdd_data[key] = pdd_values
    
    return pdd_data


def process_profile_sheet(df):
    """
    Xử lý sheet Profile.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu Profile
        
    Returns
    -------
    dict
        Dictionary chứa các profile
    """
    profile_data = {}
    
    if df.empty:
        return profile_data
    
    # Đọc từng cặp cột x và y
    col_index = 0
    while col_index < len(df.columns) - 1:
        x_col = df.columns[col_index]
        y_col = df.columns[col_index + 1]
        
        # Lấy dữ liệu
        x_values = df[x_col].values
        y_values = df[y_col].values
        
        valid_indices = ~pd.isna(x_values) & ~pd.isna(y_values)
        if any(valid_indices):
            # Tạo key cho profile
            profile_key = f"profile_{col_index//2}"
            
            # Tạo dictionary dữ liệu
            profile_values = {
                "positions": x_values[valid_indices].tolist(),
                "values": y_values[valid_indices].tolist(),
                "field_size": "10x10",  # Mặc định
                "depth": 10.0  # Mặc định
            }
            
            # Thêm vào dictionary chính
            profile_data[profile_key] = profile_values
        
        col_index += 2
    
    return profile_data


def process_output_factors_sheet(df):
    """
    Xử lý sheet Output Factors.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu Output Factors
        
    Returns
    -------
    dict
        Dictionary chứa các output factors
    """
    output_factors = {}
    
    if df.empty:
        return output_factors
    
    # Lấy giá trị field size từ hàng đầu tiên
    field_sizes = df.iloc[0, 1:].values
    
    # Đọc từng hàng
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # Lấy kích thước trường
        field_size_y = row.iloc[0]
        
        # Kiểm tra hợp lệ
        if pd.isna(field_size_y):
            continue
        
        # Lấy giá trị output factor cho mỗi kích thước trường
        for j, field_size_x in enumerate(field_sizes):
            if pd.isna(field_size_x):
                continue
                
            output_factor = row.iloc[j + 1]
            
            if pd.isna(output_factor):
                continue
            
            # Tạo key cho output factor
            key = f"{field_size_x}x{field_size_y}"
            
            # Thêm vào dictionary
            output_factors[key] = float(output_factor)
    
    return output_factors


def save_beam_model(beam_model, output_dir=None):
    """
    Lưu mô hình chùm tia vào file.
    
    Parameters
    ----------
    beam_model : BeamModel
        Mô hình chùm tia cần lưu
    output_dir : str, optional
        Thư mục đầu ra, by default None
        
    Returns
    -------
    str
        Đường dẫn đến file đã lưu
    """
    if output_dir is None:
        output_dir = os.path.join(get_beam_data_dir(), "models")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo tên file
    file_name = f"{beam_model.name}.json"
    file_path = os.path.join(output_dir, file_name)
    
    # Lưu mô hình
    beam_model.save_to_file(file_path)
    logger.info(f"Đã lưu mô hình chùm tia vào: {file_path}")
    
    return file_path


def main():
    """Hàm chính."""
    # Lấy thư mục dữ liệu beam data
    beam_data_dir = get_beam_data_dir()
    models_dir = os.path.join(beam_data_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Lấy thư mục chứa file Excel
    truebeam_dir = os.path.join(get_project_root(), "Truebeam representative Beam Data for eclipse")
    
    # Kiểm tra thư mục tồn tại
    if not os.path.isdir(truebeam_dir):
        logger.error(f"Thư mục không tồn tại: {truebeam_dir}")
        return 1
    
    # Tìm các file Excel
    excel_files = glob.glob(os.path.join(truebeam_dir, "*.xlsx"))
    
    # Lọc bỏ các file bắt đầu bằng ._
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith("._")]
    
    logger.info(f"Tìm thấy {len(excel_files)} file Excel:")
    for file in excel_files:
        logger.info(f"  - {os.path.basename(file)}")
    
    # Xử lý từng file
    success_count = 0
    
    for file_path in excel_files:
        try:
            # Xác định loại chùm tia và năng lượng
            beam_type, energy = determine_beam_type_from_filename(file_path)
            
            logger.info(f"\n{'='*80}\nĐang xử lý file {os.path.basename(file_path)} ({beam_type} {energy})...")
            
            # Kiểm tra xem mô hình đã tồn tại chưa
            model_file = os.path.join(models_dir, f"TrueBeam_{energy}.json")
            
            if os.path.exists(model_file):
                logger.info(f"Mô hình đã tồn tại: {model_file}")
                # Ghi đè mô hình
                logger.info("Ghi đè mô hình hiện có...")
            
            # Xử lý file
            beam_model = process_excel_file(file_path)
            
            # Lưu mô hình
            save_path = save_beam_model(beam_model, models_dir)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file {os.path.basename(file_path)}: {str(e)}", exc_info=True)
    
    logger.info(f"\n{'='*80}\nĐã xử lý thành công {success_count}/{len(excel_files)} file.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 