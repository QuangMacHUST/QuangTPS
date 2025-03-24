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
        Khởi tạo processor.
        
        Parameters
        ----------
        output_dir : str, optional
            Thư mục đầu ra để lưu mô hình chùm tia, by default None
        """
        self.output_dir = output_dir
        self.reader = TrueBeamDataReader()
        self.beam_models = {}
        
        # Nếu có thư mục đầu ra, tải mô hình sẵn có
        if output_dir and os.path.exists(output_dir):
            self.load_beam_models()
    
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