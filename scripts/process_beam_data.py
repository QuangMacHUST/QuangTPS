#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script xử lý dữ liệu chùm tia TrueBeam từ file Excel và nhập vào hệ thống.

Sử dụng:
    python process_beam_data.py --source "thư_mục_chứa_file_excel" --target "thư_mục_đích" --energy 6MV 10MV
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import re
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Thêm thư mục gốc vào PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("beam_processor")

class BeamDataProcessor:
    """Lớp xử lý dữ liệu chùm tia từ file Excel"""
    
    def __init__(self, source_dir, target_dir):
        """
        Khởi tạo processor
        
        Parameters
        ----------
        source_dir : str
            Thư mục chứa file Excel
        target_dir : str
            Thư mục đích để lưu dữ liệu đã xử lý
        """
        self.source_dir = os.path.abspath(source_dir)
        self.target_dir = os.path.abspath(target_dir)
        self.energy_pattern = re.compile(r'(\d+)(MV|FFF|X|E)')
        
        # Tạo thư mục đích nếu chưa tồn tại
        os.makedirs(self.target_dir, exist_ok=True)
        
    def scan_for_beam_data_files(self):
        """
        Quét thư mục nguồn để tìm các file Excel chứa dữ liệu chùm tia
        
        Returns
        -------
        dict
            Dictionary ánh xạ từ tên năng lượng đến đường dẫn file
        """
        energy_files = {}
        
        for file in os.listdir(self.source_dir):
            if file.endswith('.xlsx') and not file.startswith('._'):
                file_path = os.path.join(self.source_dir, file)
                
                # Trích xuất năng lượng từ tên file
                energy_match = self.energy_pattern.search(file)
                if energy_match:
                    energy_value = energy_match.group(1)
                    energy_type = energy_match.group(2)
                    energy_name = f"{energy_value}{energy_type}"
                    energy_files[energy_name] = file_path
                    logger.info(f"Tìm thấy dữ liệu chùm tia: {energy_name} tại {file_path}")
        
        return energy_files
    
    def process_beam_data(self, energy_name, file_path):
        """
        Xử lý dữ liệu chùm tia từ file Excel
        
        Parameters
        ----------
        energy_name : str
            Tên năng lượng (ví dụ: '6MV')
        file_path : str
            Đường dẫn đến file Excel
            
        Returns
        -------
        dict
            Dictionary chứa dữ liệu chùm tia đã xử lý
        """
        logger.info(f"Đang xử lý dữ liệu chùm tia {energy_name} từ {file_path}")
        
        # Đọc file Excel
        beam_data = {
            'energy_name': energy_name,
            'source_file': file_path,
            'pdd_data': {},
            'profile_data': {},
            'output_factors': {}
        }
        
        # Đọc danh sách các sheet trong file
        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            logger.info(f"File có các sheet: {sheet_names}")
            
            # Debug: Hiển thị nội dung một vài dòng đầu của mỗi sheet để xác định cấu trúc
            for sheet_name in sheet_names:
                if any(term in sheet_name.lower() for term in ['depth dose', 'pdd', 'percent']):
                    logger.debug(f"Kiểm tra sheet {sheet_name} cho dữ liệu PDD:")
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=10)
                    logger.debug(f"Các cột: {df.columns.tolist()}")
                    logger.debug(f"Vài dòng đầu:\n{df.head()}")
                    
                    # Kiểm tra dữ liệu trong mỗi cột để tìm mẫu
                    for col in df.columns:
                        unique_values = df[col].dropna().unique()[:5]
                        if len(unique_values) > 0:
                            logger.debug(f"Cột {col} - Giá trị mẫu: {unique_values}")
            
            # Tìm các sheet có thể chứa dữ liệu PDD
            pdd_sheets = []
            profile_sheets = []
            of_sheets = []
            
            for sheet in sheet_names:
                sheet_lower = sheet.lower()
                if 'depth dose' in sheet_lower or 'pdd' in sheet_lower or 'percent' in sheet_lower:
                    pdd_sheets.append(sheet)
                elif 'profile' in sheet_lower or 'cross' in sheet_lower or 'lateral' in sheet_lower:
                    profile_sheets.append(sheet)
                elif 'output' in sheet_lower or 'factor' in sheet_lower or 'of' in sheet_lower:
                    of_sheets.append(sheet)
            
            logger.info(f"Các sheet PDD: {pdd_sheets}")
            logger.info(f"Các sheet Profile: {profile_sheets}")
            logger.info(f"Các sheet Output Factor: {of_sheets}")
            
            # Đọc dữ liệu PDD
            if pdd_sheets:
                self._process_pdd_data(beam_data, file_path, pdd_sheets[0])
            else:
                logger.warning(f"Không tìm thấy sheet chứa dữ liệu PDD")
                
            # Đọc dữ liệu Profile
            if profile_sheets:
                self._process_profile_data(beam_data, file_path, profile_sheets[0])
            else:
                logger.warning(f"Không tìm thấy sheet chứa dữ liệu Profile")
                
            # Đọc dữ liệu Output Factor
            if of_sheets:
                self._process_output_factor_data(beam_data, file_path, of_sheets[0])
            else:
                logger.warning(f"Không tìm thấy sheet chứa dữ liệu Output Factor")
            
        except Exception as e:
            logger.error(f"Lỗi khi đọc file Excel: {str(e)}")
            
        return beam_data
    
    def _process_pdd_data(self, beam_data, file_path, sheet_name):
        """Xử lý dữ liệu PDD từ sheet Excel"""
        try:
            logger.info(f"Đọc dữ liệu PDD từ sheet '{sheet_name}'")
            pdd_df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Hiển thị thông tin chi tiết về sheet
            logger.debug(f"Thông tin sheet PDD ({sheet_name}):")
            logger.debug(f"Kích thước: {pdd_df.shape}")
            logger.debug(f"Các cột: {pdd_df.columns.tolist()}")
            logger.debug(f"5 dòng đầu:\n{pdd_df.head()}")
            
            # Kiểm tra các cột
            columns = pdd_df.columns
            logger.debug(f"Các cột trong sheet PDD: {columns}")
            
            # Tìm các cột cần thiết
            depth_col = None
            dose_col = None
            field_size_col = None
            
            for col in columns:
                col_str = str(col).lower()
                if 'depth' in col_str:
                    depth_col = col
                    logger.debug(f"Tìm thấy cột depth: {col}")
                elif any(term in col_str for term in ['dose', 'pdd', '%', 'percent']):
                    dose_col = col
                    logger.debug(f"Tìm thấy cột dose: {col}")
                elif 'field' in col_str and 'size' in col_str:
                    field_size_col = col
                    logger.debug(f"Tìm thấy cột field size: {col}")
            
            # Hiển thị thông tin để debug
            logger.debug(f"Cột độ sâu (depth): {depth_col}")
            logger.debug(f"Cột liều (dose): {dose_col}")
            logger.debug(f"Cột kích thước trường (field size): {field_size_col}")
            
            # Nếu không tìm thấy cột field size, thử tìm trong sheet
            if field_size_col is None:
                # Kiểm tra các dòng đầu xem có thông tin về field size không
                for i in range(min(10, len(pdd_df))):
                    for col in columns:
                        cell_value = str(pdd_df.iloc[i, columns.get_loc(col)]).lower()
                        if 'field' in cell_value and 'size' in cell_value and 'x' in cell_value:
                            # Trích xuất field size từ giá trị ô
                            match = re.search(r'(\d+(?:\.\d+)?)\s*[x×]\s*\1', cell_value)
                            if match:
                                field_size = float(match.group(1))
                                logger.info(f"Tìm thấy field size cố định: {field_size}x{field_size}")
                                
                                # Tạo cột field size tạm thời
                                pdd_df['field_size'] = field_size
                                field_size_col = 'field_size'
                                break
            
            if depth_col and dose_col:
                # Nếu không có thông tin field size, giả định một field size mặc định
                if field_size_col is None:
                    # Giả định field size 10x10 cm
                    field_size = 10.0
                    pdd_df['field_size'] = field_size
                    field_size_col = 'field_size'
                    logger.warning(f"Không tìm thấy thông tin field size, giả định {field_size}x{field_size} cm")
                
                # Nhóm dữ liệu theo kích thước trường
                field_sizes = pdd_df[field_size_col].dropna().unique()
                
                for field_size in field_sizes:
                    field_data = pdd_df[pdd_df[field_size_col] == field_size]
                    
                    depths = field_data[depth_col].values
                    doses = field_data[dose_col].values
                    
                    # Loại bỏ các giá trị NaN
                    valid_indices = ~(np.isnan(depths) | np.isnan(doses))
                    depths = depths[valid_indices]
                    doses = doses[valid_indices]
                    
                    # Chuẩn hóa đơn vị
                    if np.max(depths) < 100:  # Nếu đơn vị là cm
                        depths = depths * 10  # Chuyển đổi sang mm
                    
                    # Lưu dữ liệu
                    field_key = f"{field_size:.1f}x{field_size:.1f}"
                    beam_data['pdd_data'][field_key] = {
                        'depths': depths.tolist(),
                        'doses': doses.tolist()
                    }
                
                logger.info(f"Đã xử lý dữ liệu PDD cho {len(field_sizes)} kích thước trường")
            else:
                logger.warning(f"Không tìm thấy các cột cần thiết trong sheet PDD")
                # Thử phân tích cấu trúc dữ liệu đặc biệt
                self._analyze_special_sheet_structure(pdd_df, 'PDD')
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu PDD: {str(e)}")
    
    def _process_profile_data(self, beam_data, file_path, sheet_name):
        """Xử lý dữ liệu Profile từ sheet Excel"""
        try:
            logger.info(f"Đọc dữ liệu Profile từ sheet '{sheet_name}'")
            profile_df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Hiển thị thông tin chi tiết về sheet
            logger.debug(f"Thông tin sheet Profile ({sheet_name}):")
            logger.debug(f"Kích thước: {profile_df.shape}")
            logger.debug(f"Các cột: {profile_df.columns.tolist()}")
            logger.debug(f"5 dòng đầu:\n{profile_df.head()}")
            
            # Kiểm tra các cột
            columns = profile_df.columns
            logger.debug(f"Các cột trong sheet Profile: {columns}")
            
            # Tìm các cột cần thiết
            position_col = None
            dose_col = None
            field_size_col = None
            depth_col = None
            
            for col in columns:
                col_str = str(col).lower()
                if any(term in col_str for term in ['position', 'offset', 'distance', 'coord']):
                    position_col = col
                    logger.debug(f"Tìm thấy cột position: {col}")
                elif any(term in col_str for term in ['dose', 'profile', 'value', 'intensity']):
                    dose_col = col
                    logger.debug(f"Tìm thấy cột dose: {col}")
                elif 'field' in col_str and 'size' in col_str:
                    field_size_col = col
                    logger.debug(f"Tìm thấy cột field size: {col}")
                elif 'depth' in col_str:
                    depth_col = col
                    logger.debug(f"Tìm thấy cột depth: {col}")
            
            # Kiểm tra và điền giá trị mặc định nếu cần
            if position_col and dose_col:
                # Nếu không có thông tin field size, giả định một field size mặc định
                if field_size_col is None:
                    field_size = 10.0  # Giả định 10x10 cm
                    profile_df['field_size'] = field_size
                    field_size_col = 'field_size'
                    logger.warning(f"Không tìm thấy thông tin field size, giả định {field_size}x{field_size} cm")
                
                # Nếu không có thông tin depth, giả định một depth mặc định
                if depth_col is None:
                    # Thử trích xuất độ sâu từ tên sheet (ví dụ "Profiles at 10cm")
                    depth_match = re.search(r'at\s+(\d+(?:\.\d+)?)\s*cm', sheet_name, re.IGNORECASE)
                    
                    if depth_match:
                        depth = float(depth_match.group(1)) * 10  # Chuyển đổi từ cm sang mm
                        logger.info(f"Trích xuất thông tin độ sâu từ tên sheet: {depth/10} cm")
                    else:
                        depth = 100.0  # Giả định 10 cm (100 mm)
                        logger.warning(f"Không tìm thấy thông tin depth, giả định {depth/10} cm")
                    
                    profile_df['depth'] = depth
                    depth_col = 'depth'
                
                # Lấy các kích thước trường và độ sâu duy nhất
                field_sizes = profile_df[field_size_col].dropna().unique()
                depths = profile_df[depth_col].dropna().unique()
                
                for field_size in field_sizes:
                    for depth in depths:
                        filtered_data = profile_df[(profile_df[field_size_col] == field_size) & 
                                                 (profile_df[depth_col] == depth)]
                        
                        if not filtered_data.empty:
                            positions = filtered_data[position_col].values
                            doses = filtered_data[dose_col].values
                            
                            # Loại bỏ các giá trị NaN
                            valid_indices = ~(np.isnan(positions) | np.isnan(doses))
                            positions = positions[valid_indices]
                            doses = doses[valid_indices]
                            
                            # Chuẩn hóa đơn vị
                            if np.max(np.abs(positions)) < 100:  # Nếu đơn vị là cm
                                positions = positions * 10  # Chuyển đổi sang mm
                            
                            # Chuẩn hóa đơn vị độ sâu
                            if depth < 100:  # Nếu đơn vị là cm
                                depth_mm = depth * 10  # Chuyển đổi sang mm
                            else:
                                depth_mm = depth
                            
                            # Lưu dữ liệu
                            key = f"{field_size:.1f}x{field_size:.1f}_{depth_mm:.1f}mm"
                            beam_data['profile_data'][key] = {
                                'field_size': float(field_size),
                                'depth': float(depth_mm),
                                'positions': positions.tolist(),
                                'doses': doses.tolist()
                            }
                
                logger.info(f"Đã xử lý dữ liệu Profile cho {len(field_sizes)} kích thước trường ở {len(depths)} độ sâu")
            else:
                logger.warning(f"Không tìm thấy các cột cần thiết trong sheet Profile")
                # Thử phân tích cấu trúc dữ liệu đặc biệt
                self._analyze_special_sheet_structure(profile_df, 'Profile')
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu Profile: {str(e)}")
    
    def _process_output_factor_data(self, beam_data, file_path, sheet_name):
        """Xử lý dữ liệu Output Factor từ sheet Excel"""
        try:
            logger.info(f"Đọc dữ liệu Output Factor từ sheet '{sheet_name}'")
            of_df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Hiển thị thông tin chi tiết về sheet
            logger.debug(f"Thông tin sheet Output Factor ({sheet_name}):")
            logger.debug(f"Kích thước: {of_df.shape}")
            logger.debug(f"Các cột: {of_df.columns.tolist()}")
            logger.debug(f"5 dòng đầu:\n{of_df.head()}")
            
            # Kiểm tra các cột
            columns = of_df.columns
            logger.debug(f"Các cột trong sheet Output Factor: {columns}")
            
            # Tìm các cột cần thiết
            field_size_col = None
            of_col = None
            
            for col in columns:
                col_str = str(col).lower()
                if 'field' in col_str and 'size' in col_str:
                    field_size_col = col
                    logger.debug(f"Tìm thấy cột field size: {col}")
                elif any(term in col_str for term in ['output', 'factor', 'of', 'relative']):
                    of_col = col
                    logger.debug(f"Tìm thấy cột output factor: {col}")
            
            if field_size_col and of_col:
                # Lấy dữ liệu hợp lệ
                valid_data = of_df.dropna(subset=[field_size_col, of_col])
                
                field_sizes = valid_data[field_size_col].values
                output_factors = valid_data[of_col].values
                
                # Tạo dictionary
                for i in range(len(field_sizes)):
                    field_size = field_sizes[i]
                    output_factor = output_factors[i]
                    
                    # Nếu field size là một chuỗi như "10x10", cố gắng trích xuất giá trị
                    if isinstance(field_size, str):
                        match = re.search(r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)', field_size)
                        if match:
                            field_size_x = float(match.group(1))
                            field_size_y = float(match.group(2))
                            
                            # Nếu trường vuông, sử dụng một giá trị
                            if abs(field_size_x - field_size_y) < 0.001:
                                field_size = field_size_x
                            else:
                                # Đối với trường chữ nhật, sử dụng ký hiệu chữ nhật
                                field_key = f"{field_size_x:.1f}x{field_size_y:.1f}"
                                beam_data['output_factors'][field_key] = float(output_factor)
                                continue
                    
                    # Đối với trường vuông
                    field_key = f"{field_size:.1f}x{field_size:.1f}"
                    beam_data['output_factors'][field_key] = float(output_factor)
                
                logger.info(f"Đã xử lý dữ liệu Output Factor cho {len(beam_data['output_factors'])} kích thước trường")
            else:
                logger.warning(f"Không tìm thấy các cột cần thiết trong sheet Output Factor")
                # Thử phân tích cấu trúc dữ liệu đặc biệt
                self._analyze_special_sheet_structure(of_df, 'Output Factor')
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu Output Factor: {str(e)}")
    
    def save_beam_data(self, beam_data):
        """
        Lưu dữ liệu chùm tia đã xử lý vào file JSON
        
        Parameters
        ----------
        beam_data : dict
            Dictionary chứa dữ liệu chùm tia đã xử lý
            
        Returns
        -------
        str
            Đường dẫn đến file đã lưu
        """
        energy_name = beam_data['energy_name']
        output_file = os.path.join(self.target_dir, f"TrueBeam_{energy_name}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(beam_data, f, indent=2)
        
        logger.info(f"Đã lưu dữ liệu chùm tia {energy_name} vào {output_file}")
        return output_file
    
    def visualize_beam_data(self, beam_data):
        """
        Tạo các biểu đồ trực quan cho dữ liệu chùm tia
        
        Parameters
        ----------
        beam_data : dict
            Dictionary chứa dữ liệu chùm tia đã xử lý
            
        Returns
        -------
        dict
            Dictionary chứa đường dẫn đến các file hình ảnh đã tạo
        """
        energy_name = beam_data['energy_name']
        visualization_dir = os.path.join(self.target_dir, f"{energy_name}_visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
        
        result = {'pdd': [], 'profile': [], 'output_factors': None}
        
        # Vẽ đồ thị PDD
        fig, ax = plt.subplots(figsize=(10, 6))
        for field_size, data in beam_data['pdd_data'].items():
            ax.plot(data['depths'], data['doses'], label=f"Field size: {field_size}")
        
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Percentage Depth Dose (%)')
        ax.set_title(f"PDD for {energy_name}")
        ax.grid(True)
        ax.legend()
        
        pdd_file = os.path.join(visualization_dir, f"{energy_name}_pdd.png")
        plt.savefig(pdd_file)
        plt.close(fig)
        
        result['pdd'].append(pdd_file)
        
        # Vẽ đồ thị Profile
        for key, data in beam_data['profile_data'].items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['positions'], data['doses'])
            
            ax.set_xlabel('Position (mm)')
            ax.set_ylabel('Relative Dose')
            ax.set_title(f"Profile for {energy_name}, Field: {data['field_size']}x{data['field_size']}, Depth: {data['depth']} mm")
            ax.grid(True)
            
            profile_file = os.path.join(visualization_dir, f"{energy_name}_profile_{key}.png")
            plt.savefig(profile_file)
            plt.close(fig)
            
            result['profile'].append(profile_file)
        
        # Vẽ đồ thị Output Factor
        if beam_data['output_factors']:
            fig, ax = plt.subplots(figsize=(10, 6))
            field_sizes = []
            ofs = []
            
            for field_size, of in beam_data['output_factors'].items():
                size = float(field_size.split('x')[0])
                field_sizes.append(size)
                ofs.append(of)
            
            # Sắp xếp theo kích thước trường
            indices = np.argsort(field_sizes)
            field_sizes = [field_sizes[i] for i in indices]
            ofs = [ofs[i] for i in indices]
            
            ax.plot(field_sizes, ofs, 'o-')
            
            ax.set_xlabel('Field Size (cm)')
            ax.set_ylabel('Output Factor')
            ax.set_title(f"Output Factors for {energy_name}")
            ax.grid(True)
            
            of_file = os.path.join(visualization_dir, f"{energy_name}_output_factors.png")
            plt.savefig(of_file)
            plt.close(fig)
            
            result['output_factors'] = of_file
        
        logger.info(f"Đã tạo các hình ảnh trực quan cho {energy_name} trong {visualization_dir}")
        return result

    def _analyze_special_sheet_structure(self, df, data_type):
        """Phân tích cấu trúc sheet không theo tiêu chuẩn"""
        logger.debug(f"Phân tích cấu trúc đặc biệt cho {data_type}")
        
        # In ra tên các cột và 5 hàng đầu tiên để kiểm tra cấu trúc
        logger.debug(f"Các cột: {df.columns.tolist()}")
        logger.debug(f"Dữ liệu mẫu:\n{df.head()}")
        
        # Phân tích ô đầu tiên để tìm tên của các cột
        try:
            # Kiểm tra 20 dòng đầu có thông tin meta
            for i in range(min(20, len(df))):
                row = df.iloc[i]
                row_str = str(row.iloc[0]).lower() if not pd.isna(row.iloc[0]) else ""
                
                if data_type == 'PDD' and ('depth' in row_str or 'pdd' in row_str):
                    logger.debug(f"Tìm thấy dòng tiêu đề PDD tại dòng {i}: {row.tolist()}")
                elif data_type == 'Profile' and ('position' in row_str or 'profile' in row_str):
                    logger.debug(f"Tìm thấy dòng tiêu đề Profile tại dòng {i}: {row.tolist()}")
                elif data_type == 'Output Factor' and ('factor' in row_str or 'output' in row_str):
                    logger.debug(f"Tìm thấy dòng tiêu đề Output Factor tại dòng {i}: {row.tolist()}")
        except Exception as e:
            logger.error(f"Lỗi khi phân tích cấu trúc đặc biệt: {str(e)}")

    def process_all_beam_data(self, source_dir, target_dir, energy_filter=None):
        """
        Xử lý tất cả dữ liệu chùm tia từ thư mục nguồn và lưu vào thư mục đích
        
        Parameters
        ----------
        source_dir : str
            Thư mục chứa dữ liệu chùm tia
        target_dir : str
            Thư mục đích để lưu dữ liệu đã xử lý
        energy_filter : str, optional
            Lọc theo năng lượng (ví dụ: '6MV')
            
        Returns
        -------
        list
            Danh sách các file đã tạo
        """
        # Quét tìm các file dữ liệu
        energy_files = self.scan_for_beam_data_files()
        
        if not energy_files:
            logger.warning(f"Không tìm thấy file dữ liệu chùm tia nào trong {source_dir}")
            return []
        
        # Lọc theo năng lượng nếu có
        if energy_filter:
            filtered_files = {}
            for energy, file_path in energy_files.items():
                if energy_filter.lower() in energy.lower():
                    filtered_files[energy] = file_path
            
            energy_files = filtered_files
            
            if not energy_files:
                logger.warning(f"Không tìm thấy dữ liệu cho năng lượng {energy_filter}")
                return []
        
        # Xử lý từng file dữ liệu
        output_files = []
        for energy_name, file_path in energy_files.items():
            logger.info(f"Đang xử lý {energy_name} từ {file_path}")
            
            # Xử lý dữ liệu
            beam_data = self.process_beam_data(energy_name, file_path)
            
            # Lưu dữ liệu đã xử lý
            output_file = self.save_beam_data(beam_data)
            output_files.append(output_file)
        
        logger.info(f"Đã xử lý thành công {len(output_files)} năng lượng")
        logger.info(f"Các file đầu ra: {output_files}")
        
        return output_files

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Xử lý dữ liệu chùm tia từ TrueBeam')
    parser.add_argument('--source', required=True, help='Thư mục chứa dữ liệu chùm tia')
    parser.add_argument('--target', required=True, help='Thư mục đích để lưu dữ liệu đã xử lý')
    parser.add_argument('--energy', help='Lọc theo năng lượng (ví dụ: 6MV)')
    parser.add_argument('--debug', action='store_true', help='Bật chế độ debug')
    parser.add_argument('--verbose', action='store_true', help='Hiển thị thêm thông tin')
    parser.add_argument('--visualize', action='store_true', help='Tạo đồ thị trực quan hóa dữ liệu')
    
    args = parser.parse_args()
    
    # Bật chế độ debug nếu được yêu cầu
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Đảm bảo handler cũng có mức log thấp
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        
    # Xử lý dữ liệu chùm tia
    processor = BeamDataProcessor(args.source, args.target)
    processor.process_all_beam_data(args.source, args.target, args.energy)

if __name__ == "__main__":
    main() 