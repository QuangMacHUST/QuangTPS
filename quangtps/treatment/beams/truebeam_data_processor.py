#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module xử lý và nhập dữ liệu chùm tia từ máy gia tốc TrueBeam.
Hỗ trợ đọc dữ liệu từ các file Excel cung cấp bởi Varian, bao gồm PDD, Profile, và Output Factor.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import re
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from quangtps.common.paths import get_beam_data_dir
from quangtps.core.exceptions import BeamDataError
from quangtps.dose.beam_data_processor import BeamModelParameter, BeamModel
from quangtps.core.config import ConfigManager

logger = logging.getLogger(__name__)

class TrueBeamDataReader:
    """Lớp đọc và xử lý dữ liệu chùm tia từ máy TrueBeam"""
    
    def __init__(self):
        """Khởi tạo đối tượng đọc dữ liệu TrueBeam"""
        self.energy_pattern = re.compile(r'(\d+)(MV|FFF|X|E)') # Mẫu để trích xuất năng lượng từ tên file
        self.beam_data_types = {
            'PDD': 'Percent Depth Dose',
            'PROFILE': 'Beam Profile',
            'OF': 'Output Factor'
        }
    
    def scan_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Quét thư mục để tìm các file dữ liệu chùm tia TrueBeam
        
        Parameters
        ----------
        directory_path : str
            Đường dẫn đến thư mục chứa dữ liệu
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping energy names to file paths
        """
        energy_files = {}
        
        # Kiểm tra đường dẫn tồn tại
        if not os.path.exists(directory_path):
            raise BeamDataError(f"Đường dẫn không tồn tại: {directory_path}")
        
        # Duyệt qua tất cả các file trong thư mục
        for file in os.listdir(directory_path):
            filepath = os.path.join(directory_path, file)
            
            # Chỉ xử lý file Excel
            if file.endswith('.xlsx') and os.path.isfile(filepath):
                # Bỏ qua các file ._* (file ẩn trên macOS)
                if file.startswith('._'):
                    continue
                    
                # Trích xuất năng lượng từ tên file
                energy_match = self.energy_pattern.search(file)
                if energy_match:
                    energy_value = energy_match.group(1)
                    energy_type = energy_match.group(2)
                    energy_name = f"{energy_value}{energy_type}"
                    energy_files[energy_name] = filepath
                    logger.info(f"Tìm thấy dữ liệu chùm tia: {energy_name} tại {filepath}")
        
        if not energy_files:
            logger.warning(f"Không tìm thấy file dữ liệu chùm tia nào trong {directory_path}")
            
        return energy_files
    
    def read_beam_data(self, file_path: str) -> Dict[str, Any]:
        """
        Đọc dữ liệu chùm tia từ file Excel
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file Excel chứa dữ liệu
            
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa dữ liệu chùm tia đã đọc
        """
        try:
            # Xác định năng lượng từ tên file
            filename = os.path.basename(file_path)
            energy_match = self.energy_pattern.search(filename)
            if not energy_match:
                raise BeamDataError(f"Không thể xác định năng lượng từ tên file: {filename}")
                
            energy_value = energy_match.group(1)
            energy_type = energy_match.group(2)
            energy_name = f"{energy_value}{energy_type}"
            
            logger.info(f"Đang đọc dữ liệu chùm tia cho {energy_name} từ {file_path}")
            
            # Đọc file Excel
            beam_data = {}
            
            # Đọc thông tin cơ bản (thông thường ở sheet đầu tiên)
            try:
                info_df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=10)
                beam_info = {}
                
                # Trích xuất thông tin cơ bản nếu có
                for _, row in info_df.iterrows():
                    if len(row) >= 2 and isinstance(row[0], str):
                        key = row[0].strip()
                        value = row[1]
                        beam_info[key] = value
                
                beam_data['info'] = beam_info
            except Exception as e:
                logger.warning(f"Không thể đọc thông tin cơ bản: {str(e)}")
            
            # Đọc dữ liệu PDD
            try:
                pdd_data = {}
                # Thử đọc từ các sheet có tên liên quan đến PDD
                for sheet_name in ['PDD', 'Percent Depth Dose', 'PDDs']:
                    try:
                        pdd_df = pd.read_excel(file_path, sheet_name=sheet_name)
                        
                        # Tổ chức dữ liệu theo kích thước trường (field size)
                        field_sizes = []
                        for col in pdd_df.columns:
                            if 'field' in str(col).lower() and 'size' in str(col).lower():
                                field_size_col = col
                                # Trích xuất các kích thước trường duy nhất
                                field_sizes = pdd_df[field_size_col].dropna().unique()
                                break
                        
                        # Tạo dictionary cho mỗi kích thước trường
                        for field_size in field_sizes:
                            field_data = pdd_df[pdd_df[field_size_col] == field_size]
                            
                            # Tìm cột độ sâu và phần trăm liều
                            depth_col = None
                            dose_col = None
                            
                            for col in field_data.columns:
                                col_str = str(col).lower()
                                if 'depth' in col_str:
                                    depth_col = col
                                elif 'dose' in col_str or 'pdd' in col_str:
                                    dose_col = col
                            
                            if depth_col is not None and dose_col is not None:
                                depths = field_data[depth_col].values
                                doses = field_data[dose_col].values
                                
                                # Lọc ra các giá trị hợp lệ
                                valid_indices = ~(np.isnan(depths) | np.isnan(doses))
                                depths = depths[valid_indices]
                                doses = doses[valid_indices]
                                
                                # Chuẩn hóa về mm nếu cần
                                if np.max(depths) < 100:  # Có thể đơn vị là cm
                                    depths = depths * 10  # Chuyển đổi từ cm sang mm
                                
                                pdd_data[f"{field_size:.1f}x{field_size:.1f}"] = {
                                    'depths': depths,
                                    'doses': doses
                                }
                        
                        break  # Nếu đọc thành công từ sheet này, không cần đọc các sheet khác
                        
                    except Exception as e:
                        logger.debug(f"Không thể đọc dữ liệu PDD từ sheet {sheet_name}: {str(e)}")
                
                beam_data['pdd'] = pdd_data
                
            except Exception as e:
                logger.warning(f"Không thể đọc dữ liệu PDD: {str(e)}")
            
            # Đọc dữ liệu Profile
            try:
                profile_data = {}
                # Thử đọc từ các sheet có tên liên quan đến Profile
                for sheet_name in ['Profile', 'Profiles', 'Beam Profile']:
                    try:
                        profile_df = pd.read_excel(file_path, sheet_name=sheet_name)
                        
                        # Tìm các cột chứa thông tin về field size, depth, và profile
                        field_size_col = None
                        depth_col = None
                        position_col = None
                        dose_col = None
                        
                        for col in profile_df.columns:
                            col_str = str(col).lower()
                            if 'field' in col_str and 'size' in col_str:
                                field_size_col = col
                            elif 'depth' in col_str:
                                depth_col = col
                            elif 'position' in col_str or 'distance' in col_str or 'offset' in col_str:
                                position_col = col
                            elif 'dose' in col_str or 'profile' in col_str:
                                dose_col = col
                        
                        # Nếu tìm thấy đủ cột cần thiết
                        if field_size_col is not None and depth_col is not None and position_col is not None and dose_col is not None:
                            # Lấy các kích thước trường và độ sâu duy nhất
                            field_sizes = profile_df[field_size_col].dropna().unique()
                            depths = profile_df[depth_col].dropna().unique()
                            
                            # Tạo dictionary cho mỗi kích thước trường và độ sâu
                            for field_size in field_sizes:
                                for depth in depths:
                                    # Lọc dữ liệu cho kích thước trường và độ sâu cụ thể
                                    filtered_data = profile_df[(profile_df[field_size_col] == field_size) & 
                                                              (profile_df[depth_col] == depth)]
                                    
                                    if not filtered_data.empty:
                                        positions = filtered_data[position_col].values
                                        doses = filtered_data[dose_col].values
                                        
                                        # Lọc ra các giá trị hợp lệ
                                        valid_indices = ~(np.isnan(positions) | np.isnan(doses))
                                        positions = positions[valid_indices]
                                        doses = doses[valid_indices]
                                        
                                        # Chuẩn hóa về mm nếu cần
                                        if np.max(np.abs(positions)) < 100:  # Có thể đơn vị là cm
                                            positions = positions * 10  # Chuyển đổi từ cm sang mm
                                        
                                        # Chuẩn hóa độ sâu về mm nếu cần
                                        if depth < 100:  # Có thể đơn vị là cm
                                            depth_mm = depth * 10
                                        else:
                                            depth_mm = depth
                                        
                                        key = f"{field_size:.1f}x{field_size:.1f}_{depth_mm:.1f}mm"
                                        profile_data[key] = {
                                            'field_size': field_size,
                                            'depth': depth_mm,
                                            'positions': positions,
                                            'doses': doses
                                        }
                            
                            break  # Nếu đọc thành công từ sheet này, không cần đọc các sheet khác
                            
                    except Exception as e:
                        logger.debug(f"Không thể đọc dữ liệu Profile từ sheet {sheet_name}: {str(e)}")
                
                beam_data['profile'] = profile_data
                
            except Exception as e:
                logger.warning(f"Không thể đọc dữ liệu Profile: {str(e)}")
            
            # Đọc dữ liệu Output Factor
            try:
                of_data = {}
                # Thử đọc từ các sheet có tên liên quan đến Output Factor
                for sheet_name in ['OF', 'Output Factor', 'Output Factors']:
                    try:
                        of_df = pd.read_excel(file_path, sheet_name=sheet_name)
                        
                        # Tìm các cột chứa thông tin về field size và output factor
                        field_size_col = None
                        of_col = None
                        
                        for col in of_df.columns:
                            col_str = str(col).lower()
                            if 'field' in col_str and 'size' in col_str:
                                field_size_col = col
                            elif 'factor' in col_str or 'of' in col_str or 'output' in col_str:
                                of_col = col
                        
                        # Nếu tìm thấy đủ cột cần thiết
                        if field_size_col is not None and of_col is not None:
                            # Lọc dữ liệu hợp lệ
                            valid_data = of_df[[field_size_col, of_col]].dropna()
                            
                            # Đọc dữ liệu
                            field_sizes = valid_data[field_size_col].values
                            factors = valid_data[of_col].values
                            
                            # Lưu dữ liệu
                            of_data = {
                                'field_sizes': field_sizes,
                                'factors': factors
                            }
                            
                            break  # Nếu đọc thành công từ sheet này, không cần đọc các sheet khác
                            
                    except Exception as e:
                        logger.debug(f"Không thể đọc dữ liệu Output Factor từ sheet {sheet_name}: {str(e)}")
                
                beam_data['output_factor'] = of_data
                
            except Exception as e:
                logger.warning(f"Không thể đọc dữ liệu Output Factor: {str(e)}")
            
            # Thêm thông tin năng lượng
            beam_data['energy'] = {
                'name': energy_name,
                'value': energy_value,
                'type': energy_type
            }
            
            return beam_data
            
        except Exception as e:
            raise BeamDataError(f"Lỗi khi đọc dữ liệu chùm tia từ {file_path}: {str(e)}")
    
    def create_beam_model(self, beam_data: Dict[str, Any]) -> BeamModel:
        """
        Tạo mô hình chùm tia từ dữ liệu đã đọc
        
        Parameters
        ----------
        beam_data : Dict[str, Any]
            Dữ liệu chùm tia đã đọc từ file Excel
            
        Returns
        -------
        BeamModel
            Mô hình chùm tia
        """
        # Lấy thông tin năng lượng
        energy_info = beam_data.get('energy', {})
        energy_name = energy_info.get('name', 'Unknown')
        
        # Tạo mô hình chùm tia
        beam_model = BeamModel(
            name=f"TrueBeam_{energy_name}",
            description=f"TrueBeam {energy_name} beam model",
            source="Varian TrueBeam"
        )
        
        # Thêm thông số PDD
        if 'pdd' in beam_data:
            for field_size, pdd_info in beam_data['pdd'].items():
                depths = pdd_info['depths']
                doses = pdd_info['doses']
                
                # Tạo tham số PDD
                pdd_param = BeamModelParameter(
                    name=f"PDD_{field_size}",
                    type="PDD",
                    field_size=field_size,
                    x_values=depths,
                    y_values=doses,
                    unit="mm",
                    description=f"Percent Depth Dose for {field_size} field size"
                )
                
                # Thêm vào mô hình
                beam_model.add_parameter(pdd_param)
        
        # Thêm thông số Profile
        if 'profile' in beam_data:
            for key, profile_info in beam_data['profile'].items():
                field_size = profile_info['field_size']
                depth = profile_info['depth']
                positions = profile_info['positions']
                doses = profile_info['doses']
                
                # Tạo tham số Profile
                profile_param = BeamModelParameter(
                    name=f"Profile_{field_size:.1f}x{field_size:.1f}_{depth:.1f}mm",
                    type="PROFILE",
                    field_size=f"{field_size:.1f}x{field_size:.1f}",
                    depth=depth,
                    x_values=positions,
                    y_values=doses,
                    unit="mm",
                    description=f"Beam Profile for {field_size:.1f}x{field_size:.1f} field size at {depth:.1f}mm depth"
                )
                
                # Thêm vào mô hình
                beam_model.add_parameter(profile_param)
        
        # Thêm thông số Output Factor
        if 'output_factor' in beam_data and 'field_sizes' in beam_data['output_factor']:
            field_sizes = beam_data['output_factor']['field_sizes']
            factors = beam_data['output_factor']['factors']
            
            # Tạo tham số Output Factor
            of_param = BeamModelParameter(
                name=f"OF_{energy_name}",
                type="OUTPUT_FACTOR",
                x_values=field_sizes,
                y_values=factors,
                unit="",
                description=f"Output Factors for {energy_name}"
            )
            
            # Thêm vào mô hình
            beam_model.add_parameter(of_param)
        
        # Thêm thông tin cơ bản
        if 'info' in beam_data:
            for key, value in beam_data['info'].items():
                if isinstance(value, (str, int, float, bool)):
                    beam_model.add_metadata(key, value)
        
        return beam_model
    
    def process_all_energies(self, directory_path: str, output_directory: str = None) -> Dict[str, BeamModel]:
        """
        Xử lý tất cả các năng lượng trong thư mục
        
        Parameters
        ----------
        directory_path : str
            Đường dẫn đến thư mục chứa dữ liệu
        output_directory : str, optional
            Đường dẫn đến thư mục lưu kết quả, by default None
            
        Returns
        -------
        Dict[str, BeamModel]
            Dictionary mapping energy names to beam models
        """
        # Quét thư mục để tìm các file dữ liệu
        energy_files = self.scan_directory(directory_path)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Xử lý từng năng lượng
        beam_models = {}
        
        for energy_name, file_path in energy_files.items():
            try:
                # Đọc dữ liệu
                beam_data = self.read_beam_data(file_path)
                
                # Tạo mô hình chùm tia
                beam_model = self.create_beam_model(beam_data)
                
                # Lưu vào dictionary
                beam_models[energy_name] = beam_model
                
                # Lưu vào file nếu có thư mục đầu ra
                if output_directory:
                    output_path = os.path.join(output_directory, f"TrueBeam_{energy_name}_beam_model.json")
                    beam_model.save_to_json(output_path)
                    logger.info(f"Đã lưu mô hình chùm tia {energy_name} vào {output_path}")
                    
                    # Tạo đồ thị trực quan hóa
                    self._visualize_beam_model(beam_model, output_directory)
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý năng lượng {energy_name}: {str(e)}")
        
        return beam_models
    
    def _visualize_beam_model(self, beam_model: BeamModel, output_directory: str):
        """
        Trực quan hóa mô hình chùm tia
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia cần trực quan hóa
        output_directory : str
            Thư mục lưu đồ thị
        """
        try:
            # Tạo thư mục con cho đồ thị
            viz_dir = os.path.join(output_directory, "visualization")
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Trực quan hóa PDD
            pdd_params = [param for param in beam_model.parameters if param.type == "PDD"]
            if pdd_params:
                plt.figure(figsize=(10, 6))
                for param in pdd_params:
                    plt.plot(param.x_values, param.y_values, label=param.name)
                
                plt.title(f"PDD Curves - {beam_model.name}")
                plt.xlabel("Depth (mm)")
                plt.ylabel("Percent Dose (%)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                plt.savefig(os.path.join(viz_dir, f"{beam_model.name}_PDD.png"))
                plt.close()
            
            # Trực quan hóa Profile
            profile_params = [param for param in beam_model.parameters if param.type == "PROFILE"]
            if profile_params:
                # Tổ chức profile theo độ sâu
                depths = set()
                for param in profile_params:
                    if hasattr(param, 'depth'):
                        depths.add(param.depth)
                
                for depth in depths:
                    plt.figure(figsize=(10, 6))
                    depth_profiles = [param for param in profile_params if hasattr(param, 'depth') and param.depth == depth]
                    
                    for param in depth_profiles:
                        plt.plot(param.x_values, param.y_values, label=param.field_size)
                    
                    plt.title(f"Beam Profiles at {depth}mm - {beam_model.name}")
                    plt.xlabel("Off-axis Distance (mm)")
                    plt.ylabel("Relative Dose (%)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, f"{beam_model.name}_Profile_{depth}mm.png"))
                    plt.close()
            
            # Trực quan hóa Output Factor
            of_params = [param for param in beam_model.parameters if param.type == "OUTPUT_FACTOR"]
            if of_params:
                plt.figure(figsize=(10, 6))
                for param in of_params:
                    plt.plot(param.x_values, param.y_values, 'o-')
                
                plt.title(f"Output Factors - {beam_model.name}")
                plt.xlabel("Field Size (cm)")
                plt.ylabel("Output Factor")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(os.path.join(viz_dir, f"{beam_model.name}_OutputFactors.png"))
                plt.close()
                
        except Exception as e:
            logger.warning(f"Không thể tạo đồ thị cho mô hình chùm tia {beam_model.name}: {str(e)}")


class TrueBeamDataProcessor:
    """
    Lớp xử lý dữ liệu chùm tia từ máy TrueBeam và tích hợp vào hệ thống
    """
    
    def __init__(self, output_dir: str = None):
        """
        Khởi tạo processor dữ liệu TrueBeam.
        
        Parameters
        ----------
        output_dir : str, optional
            Thư mục đầu ra để lưu các mô hình chùm tia, mặc định là data/beam_data
        """
        self.output_dir = output_dir or get_beam_data_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tạo thư mục models nếu chưa tồn tại
        self.models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Lưu trữ các mô hình đã được tải
        self.beam_models = {}
        
        # Khởi tạo reader để đọc dữ liệu
        self.reader = TrueBeamDataReader()
    
    def get_available_energies(self) -> List[str]:
        """
        Lấy danh sách các năng lượng có sẵn trong processor.
        
        Returns
        -------
        List[str]
            Danh sách tên năng lượng
        """
        return list(self.beam_models.keys())
    
    def load_beam_models(self) -> Dict[str, BeamModel]:
        """
        Tải các mô hình chùm tia từ thư mục đầu ra.
        
        Returns
        -------
        Dict[str, BeamModel]
            Dictionary chứa các mô hình chùm tia đã tải
        """
        self.beam_models = {}
        
        if not self.output_dir or not os.path.exists(self.output_dir):
            logger.warning(f"Thư mục đầu ra không tồn tại: {self.output_dir}")
            return self.beam_models
        
        # Tìm tất cả các file JSON trong thư mục
        json_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json') and 'TrueBeam' in f]
        
        for json_file in json_files:
            try:
                file_path = os.path.join(self.output_dir, json_file)
                
                # Trích xuất tên năng lượng từ tên file
                energy_match = self.reader.energy_pattern.search(json_file)
                if energy_match:
                    energy_name = f"{energy_match.group(1)}{energy_match.group(2)}"
                    
                    # Tải mô hình
                    beam_model = BeamModel.load_from_json(file_path)
                    self.beam_models[energy_name] = beam_model
                    logger.info(f"Đã tải mô hình chùm tia {energy_name} từ {json_file}")
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình từ {json_file}: {str(e)}")
        
        return self.beam_models
    
    def create_beam_model(self, energy: str) -> Optional[BeamModel]:
        """
        Tạo mô hình chùm tia cho năng lượng cụ thể.
        
        Parameters
        ----------
        energy : str
            Tên năng lượng (ví dụ: "6MV", "10FFF")
            
        Returns
        -------
        Optional[BeamModel]
            Mô hình chùm tia đã tạo, hoặc None nếu không thành công
        """
        try:
            # Quét tìm file dữ liệu phù hợp với năng lượng
            energy_files = self.scan_for_beam_data_files()
            
            if energy not in energy_files:
                logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
                return None
            
            file_path = energy_files[energy]
            
            # Đọc dữ liệu
            beam_data = self.reader.read_beam_data(file_path)
            
            # Tạo mô hình
            beam_model = self.reader.create_beam_model(beam_data)
            
            # Lưu vào dictionary
            self.beam_models[energy] = beam_model
            
            # Lưu vào file nếu có thư mục đầu ra
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(self.output_dir, f"TrueBeam_{energy}_beam_model.json")
                beam_model.save_to_json(output_path)
                logger.info(f"Đã lưu mô hình chùm tia {energy} vào {output_path}")
            
            return beam_model
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo mô hình chùm tia cho {energy}: {str(e)}")
            return None
    
    def scan_for_beam_data_files(self) -> Dict[str, str]:
        """
        Quét các thư mục để tìm file dữ liệu chùm tia TrueBeam.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping energy names to file paths
        """
        # Tìm trong thư mục dữ liệu mặc định
        config = ConfigManager().get_config()
        beam_data_dirs = [
            os.path.join(config.get('data_directory', 'data'), 'beam_data'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'beam_data'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'Truebeam representative Beam Data for eclipse')
        ]
        
        # Thêm thư mục hiện tại
        beam_data_dirs.append(os.getcwd())
        
        energy_files = {}
        
        for directory in beam_data_dirs:
            if os.path.exists(directory):
                try:
                    logger.debug(f"Quét thư mục {directory} để tìm dữ liệu chùm tia")
                    files = self.reader.scan_directory(directory)
                    energy_files.update(files)
                except Exception as e:
                    logger.debug(f"Lỗi khi quét thư mục {directory}: {str(e)}")
        
        return energy_files
    
    def process_beam_data(self, input_directory: str, energies: List[str] = None) -> Dict[str, BeamModel]:
        """
        Xử lý dữ liệu chùm tia từ thư mục đầu vào
        
        Parameters
        ----------
        input_directory : str
            Thư mục chứa dữ liệu chùm tia
        energies : List[str], optional
            Danh sách các năng lượng cần xử lý, by default None (xử lý tất cả)
            
        Returns
        -------
        Dict[str, BeamModel]
            Dictionary mapping energy names to beam models
        """
        # Quét thư mục để tìm các file dữ liệu
        energy_files = self.scan_for_beam_data_files()
        
        # Lọc theo danh sách năng lượng nếu có
        if energies:
            energy_files = {k: v for k, v in energy_files.items() if k in energies}
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Xử lý từng năng lượng
        for energy_name, file_path in energy_files.items():
            try:
                logger.info(f"Đang xử lý dữ liệu chùm tia cho năng lượng {energy_name}")
                
                # Đọc dữ liệu
                beam_data = self.reader.read_beam_data(file_path)
                
                # Tạo mô hình chùm tia
                beam_model = self.create_beam_model(energy_name)
                
                # Lưu vào dictionary
                self.beam_models[energy_name] = beam_model
                
                # Lưu vào file
                output_path = os.path.join(self.output_dir, f"TrueBeam_{energy_name}_beam_model.json")
                beam_model.save_to_json(output_path)
                logger.info(f"Đã lưu mô hình chùm tia {energy_name} vào {output_path}")
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý năng lượng {energy_name}: {str(e)}")
        
        return self.beam_models
    
    def get_beam_model(self, energy_name: str) -> Optional[BeamModel]:
        """
        Lấy mô hình chùm tia cho một năng lượng cụ thể
        
        Parameters
        ----------
        energy_name : str
            Tên năng lượng (ví dụ: "6MV", "10FFF")
            
        Returns
        -------
        Optional[BeamModel]
            Mô hình chùm tia hoặc None nếu không tìm thấy
        """
        if not self.beam_models:
            self.load_beam_models()
        
        return self.beam_models.get(energy_name)
    
    def convert_to_treatment_planning_format(self, energy_name: str, output_directory: str) -> str:
        """
        Chuyển đổi mô hình chùm tia sang định dạng cho lập kế hoạch điều trị
        
        Parameters
        ----------
        energy_name : str
            Tên năng lượng
        output_directory : str
            Thư mục đầu ra
            
        Returns
        -------
        str
            Đường dẫn đến file đã chuyển đổi
        """
        beam_model = self.get_beam_model(energy_name)
        if not beam_model:
            raise BeamDataError(f"Không tìm thấy mô hình chùm tia cho năng lượng {energy_name}")
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Tạo file đầu ra
        output_path = os.path.join(output_directory, f"TrueBeam_{energy_name}_planning_data.txt")
        
        try:
            with open(output_path, 'w') as f:
                # Viết thông tin cơ bản
                f.write(f"# TrueBeam Beam Model for {energy_name}\n")
                f.write(f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Source: {beam_model.source}\n")
                f.write(f"# Description: {beam_model.description}\n\n")
                
                # Viết thông tin PDD
                pdd_params = [param for param in beam_model.parameters if param.type == "PDD"]
                f.write("# PERCENT DEPTH DOSE DATA\n")
                f.write("# Format: field_size depth dose\n")
                
                for param in pdd_params:
                    f.write(f"# Field Size: {param.field_size} cm\n")
                    for i in range(len(param.x_values)):
                        f.write(f"{param.field_size} {param.x_values[i]:.2f} {param.y_values[i]:.4f}\n")
                    f.write("\n")
                
                # Viết thông tin Profile
                profile_params = [param for param in beam_model.parameters if param.type == "PROFILE"]
                f.write("# BEAM PROFILE DATA\n")
                f.write("# Format: field_size depth position dose\n")
                
                for param in profile_params:
                    if hasattr(param, 'depth') and hasattr(param, 'field_size'):
                        f.write(f"# Field Size: {param.field_size} cm, Depth: {param.depth} mm\n")
                        for i in range(len(param.x_values)):
                            f.write(f"{param.field_size} {param.depth:.2f} {param.x_values[i]:.2f} {param.y_values[i]:.4f}\n")
                        f.write("\n")
                
                # Viết thông tin Output Factor
                of_params = [param for param in beam_model.parameters if param.type == "OUTPUT_FACTOR"]
                f.write("# OUTPUT FACTOR DATA\n")
                f.write("# Format: field_size factor\n")
                
                for param in of_params:
                    for i in range(len(param.x_values)):
                        f.write(f"{param.x_values[i]:.2f} {param.y_values[i]:.4f}\n")
                
                # Viết metadata
                f.write("\n# METADATA\n")
                for key, value in beam_model.metadata.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Đã chuyển đổi mô hình chùm tia {energy_name} sang {output_path}")
            return output_path
            
        except Exception as e:
            raise BeamDataError(f"Lỗi khi chuyển đổi mô hình chùm tia {energy_name}: {str(e)}")

    def read_excel_file(self, file_path: str) -> bool:
        """
        Đọc dữ liệu từ file Excel của TrueBeam.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file Excel cần đọc
            
        Returns
        -------
        bool
            True nếu đọc thành công, False nếu thất bại
        """
        try:
            import pandas as pd
            
            # Kiểm tra file tồn tại
            if not os.path.exists(file_path):
                logger.error(f"File không tồn tại: {file_path}")
                return False
                
            # Lấy tên file để xác định loại chùm tia và năng lượng
            beam_type, energy = self._determine_beam_type_from_filename(file_path)
            
            # Lưu thông tin cơ bản
            self.current_file = file_path
            self.current_energy = energy
            self.current_beam_type = beam_type
            
            # Đọc dữ liệu từ các sheet
            logger.info(f"Đọc file Excel: {file_path}")
            
            # Đọc sheet PDD data
            try:
                pdd_df = pd.read_excel(file_path, sheet_name='PDD', engine='openpyxl')
                self.pdd_data = self._process_pdd_sheet(pdd_df)
                logger.info(f"Đã đọc dữ liệu PDD: {len(self.pdd_data)} điểm dữ liệu")
            except Exception as e:
                logger.warning(f"Không thể đọc sheet PDD: {str(e)}")
                self.pdd_data = {}
                
            # Đọc sheet Profile data
            try:
                profile_df = pd.read_excel(file_path, sheet_name='Profiles', engine='openpyxl')
                self.profile_data = self._process_profile_sheet(profile_df)
                logger.info(f"Đã đọc dữ liệu Profile: {len(self.profile_data)} profile")
            except Exception as e:
                logger.warning(f"Không thể đọc sheet Profiles: {str(e)}")
                self.profile_data = {}
                
            # Đọc sheet Output factors
            try:
                output_df = pd.read_excel(file_path, sheet_name='Output Factors', engine='openpyxl')
                self.output_factors = self._process_output_factors_sheet(output_df)
                logger.info(f"Đã đọc dữ liệu Output Factors: {len(self.output_factors)} kích thước trường")
            except Exception as e:
                logger.warning(f"Không thể đọc sheet Output Factors: {str(e)}")
                self.output_factors = {}
                
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi đọc file Excel: {str(e)}", exc_info=True)
            return False
            
    def _process_pdd_sheet(self, df):
        """Xử lý dữ liệu từ sheet PDD."""
        pdd_data = {}
        
        try:
            # Dùng hàng đầu tiên làm tên cột
            if df.shape[0] < 2:
                return pdd_data
                
            # Chuyển đổi DataFrame
            for col in df.columns:
                if 'field' in col.lower() or 'ssd' in col.lower() or 'depth' in col.lower():
                    continue
                    
                # Lấy thông tin trường từ tên cột
                # Ví dụ: "10x10" hoặc "10x10 SSD=100"
                field_info = col.strip()
                field_parts = field_info.split()
                
                field_size = field_parts[0] if field_parts else "Unknown"
                ssd = 100.0  # Mặc định
                
                # Tìm SSD từ tên cột
                for part in field_parts:
                    if 'ssd=' in part.lower():
                        try:
                            ssd = float(part.lower().replace('ssd=', ''))
                        except:
                            pass
                            
                # Lấy dữ liệu depth và PDD
                depths = df['Depth'].values if 'Depth' in df.columns else df.iloc[:, 0].values
                pdds = df[col].values
                
                # Tạo dictionary dữ liệu PDD
                pdd_values = {
                    'depths': depths.tolist(),
                    'values': pdds.tolist(),
                    'ssd': ssd,
                    'field_size': field_size
                }
                
                # Thêm vào dictionary chính
                key = f"{field_size}_SSD={ssd}"
                pdd_data[key] = pdd_values
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý sheet PDD: {str(e)}", exc_info=True)
            
        return pdd_data
            
    def _process_profile_sheet(self, df):
        """Xử lý dữ liệu từ sheet Profile."""
        profile_data = {}
        
        try:
            # Phân tích cấu trúc sheet
            # Sheet có thể có nhiều profile, mỗi profile được đánh dấu bởi hàng tiêu đề
            
            current_profile = None
            profile_meta = {}
            profile_positions = []
            profile_values = []
            
            for i, row in df.iterrows():
                # Kiểm tra xem dòng này có phải là tiêu đề của profile mới không
                if not pd.isna(row.iloc[0]) and 'field' in str(row.iloc[0]).lower():
                    # Lưu profile trước đó nếu có
                    if current_profile and profile_positions and profile_values:
                        profile_data[current_profile] = {
                            'meta': profile_meta,
                            'positions': profile_positions,
                            'values': profile_values
                        }
                        
                    # Bắt đầu profile mới
                    try:
                        meta_text = str(row.iloc[0])
                        meta_parts = meta_text.split(',')
                        
                        profile_meta = {}
                        for part in meta_parts:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                profile_meta[key.strip().lower()] = value.strip()
                        
                        # Tạo tên profile từ meta
                        field_size = profile_meta.get('field', 'Unknown')
                        depth = profile_meta.get('depth', 'Unknown')
                        axis = profile_meta.get('axis', 'Unknown')
                        
                        current_profile = f"{field_size}_D={depth}_{axis}"
                        profile_positions = []
                        profile_values = []
                    except:
                        current_profile = None
                        
                # Nếu đang trong profile, đọc dữ liệu
                elif current_profile and not pd.isna(row.iloc[0]) and not pd.isna(row.iloc[1]):
                    try:
                        position = float(row.iloc[0])
                        value = float(row.iloc[1])
                        
                        profile_positions.append(position)
                        profile_values.append(value)
                    except:
                        pass
            
            # Lưu profile cuối cùng
            if current_profile and profile_positions and profile_values:
                profile_data[current_profile] = {
                    'meta': profile_meta,
                    'positions': profile_positions,
                    'values': profile_values
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý sheet Profile: {str(e)}", exc_info=True)
            
        return profile_data
            
    def _process_output_factors_sheet(self, df):
        """Xử lý dữ liệu từ sheet Output Factors."""
        output_factors = {}
        
        try:
            # Thường sheet Output Factors có cấu trúc ma trận
            # Với hàng là kích thước X, cột là kích thước Y
            
            # Tìm các cột có dữ liệu
            data_columns = [col for col in df.columns if isinstance(col, (int, float)) or 
                            (isinstance(col, str) and col.replace('.', '', 1).isdigit())]
            
            # Chuyển thành float
            data_columns = [float(col) if isinstance(col, str) else col for col in data_columns]
            
            # Đọc từng hàng
            for i, row in df.iterrows():
                try:
                    # Kiểm tra hàng có dữ liệu không
                    if pd.isna(row.iloc[0]) or not isinstance(row.iloc[0], (int, float)):
                        continue
                        
                    x_size = float(row.iloc[0])
                    
                    for j, col in enumerate(data_columns):
                        y_size = float(col)
                        
                        # Lấy giá trị output factor
                        try:
                            value = float(row.iloc[j+1])
                            
                            # Tạo key cho output factor
                            if x_size == y_size:
                                key = f"{x_size}x{y_size}"
                            else:
                                key = f"{x_size}x{y_size}"
                                
                            output_factors[key] = value
                        except:
                            pass
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Lỗi khi xử lý sheet Output Factors: {str(e)}", exc_info=True)
            
        return output_factors
    
    def create_beam_model(self) -> Optional[BeamModel]:
        """
        Tạo mô hình chùm tia từ dữ liệu đã đọc.
        
        Returns
        -------
        Optional[BeamModel]
            Mô hình chùm tia, hoặc None nếu không có dữ liệu
        """
        if not hasattr(self, 'current_energy') or not self.current_energy:
            logger.error("Chưa đọc dữ liệu từ file Excel")
            return None
            
        try:
            # Tạo mô hình chùm tia mới
            model_name = f"TrueBeam_{self.current_energy}"
            beam_model = BeamModel(model_name, self.current_energy, self.current_beam_type)
            
            # Thêm thông tin nguồn
            beam_model.set_source_file(self.current_file)
            
            # Thêm dữ liệu PDD
            for key, pdd_data in getattr(self, 'pdd_data', {}).items():
                param_name = f"pdd_{key}"
                beam_model.add_parameter(param_name, pdd_data)
                
            # Thêm dữ liệu Profile
            for key, profile_data in getattr(self, 'profile_data', {}).items():
                param_name = f"profile_{key}"
                beam_model.add_parameter(param_name, profile_data)
                
            # Thêm Output Factors
            beam_model.add_parameter("output_factors", getattr(self, 'output_factors', {}))
            
            # Lưu model
            model_file = os.path.join(self.models_dir, f"{model_name}.json")
            beam_model.save_to_file(model_file)
            
            # Lưu vào cache
            self.beam_models[self.current_energy] = beam_model
            
            return beam_model
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo mô hình chùm tia: {str(e)}", exc_info=True)
            return None 