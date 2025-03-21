#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý mô hình chùm tia Truebeam.

Module này cung cấp các lớp và hàm để xử lý dữ liệu chùm tia Truebeam,
đọc dữ liệu từ các file Excel, và tạo mô hình chùm tia cho việc tính toán liều.
"""

import os
import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from quangtps.core.exceptions import DataProcessingError
from quangtps.core.utils import ensure_directory
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter, BeamModelFactory

logger = logging.getLogger(__name__)

class TruebeamEnergyModelBuilder:
    """
    Lớp xây dựng mô hình năng lượng cho máy Truebeam từ dữ liệu Excel.
    
    Lớp này đọc và xử lý dữ liệu từ Excel để tạo mô hình chùm tia
    cho các năng lượng khác nhau của Truebeam.
    """
    
    ENERGY_NAMES = {
        "4MV": "4X",
        "6MV": "6X",
        "8MV": "8X", 
        "10MV": "10X",
        "15MV": "15X",
        "6FFF": "6X-FFF",
        "10FFF": "10X-FFF"
    }
    
    def __init__(self, data_directory: str):
        """
        Khởi tạo builder với thư mục chứa dữ liệu.
        
        Parameters
        ----------
        data_directory : str
            Đường dẫn đến thư mục chứa các file Excel dữ liệu chùm tia
        """
        self.data_directory = data_directory
        self.energy_files = self._find_energy_files()
        logger.info(f"Tìm thấy {len(self.energy_files)} file dữ liệu năng lượng trong {data_directory}")
    
    def _find_energy_files(self) -> Dict[str, str]:
        """
        Tìm các file Excel chứa dữ liệu cho từng năng lượng.
        
        Returns
        -------
        Dict[str, str]
            Dictionary với khóa là tên năng lượng và giá trị là đường dẫn đến file
        """
        energy_files = {}
        
        # Danh sách các mẫu tên file cho từng năng lượng
        energy_patterns = {
            "4MV": ["4MV", "4 MV"],
            "6MV": ["6MV", "6 MV"],
            "8MV": ["8MV", "8 MV"],
            "10MV": ["10MV", "10 MV"],
            "15MV": ["15MV", "15 MV"],
            "6FFF": ["6FFF", "6 FFF"],
            "10FFF": ["10FFF", "10 FFF"]
        }
        
        # Kiểm tra các file trong thư mục
        for file in os.listdir(self.data_directory):
            if file.endswith(".xlsx") and "Beam Data" in file:
                file_path = os.path.join(self.data_directory, file)
                
                # Kiểm tra xem file thuộc năng lượng nào
                for energy, patterns in energy_patterns.items():
                    if any(pattern in file for pattern in patterns):
                        energy_files[energy] = file_path
                        logger.info(f"Tìm thấy file dữ liệu cho năng lượng {energy}: {file}")
                        break
        
        return energy_files
    
    def build_model(self, energy: str, output_directory: Optional[str] = None) -> Optional[BeamModel]:
        """
        Xây dựng mô hình chùm tia cho một năng lượng cụ thể.
        
        Parameters
        ----------
        energy : str
            Năng lượng cần xây dựng mô hình (ví dụ: "6MV", "10FFF")
        output_directory : str, optional
            Thư mục đầu ra để lưu mô hình, nếu None sẽ không lưu
            
        Returns
        -------
        Optional[BeamModel]
            Mô hình chùm tia được tạo, hoặc None nếu không thể tạo
        
        Raises
        ------
        DataProcessingError
            Nếu có lỗi trong quá trình xử lý dữ liệu
        """
        if energy not in self.energy_files:
            logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
            return None
        
        file_path = self.energy_files[energy]
        logger.info(f"Đang xây dựng mô hình cho năng lượng {energy} từ file {file_path}")
        
        try:
            # Tạo mô hình chùm tia mới
            beam_model = BeamModel(
                name=f"Truebeam {self.ENERGY_NAMES.get(energy, energy)}",
                energy=energy,
                beam_type="PHOTON"
            )
            
            # Đọc và xử lý dữ liệu từ Excel
            self._process_pdd_data(beam_model, file_path, energy)
            self._process_profile_data(beam_model, file_path, energy)
            self._process_output_factors(beam_model, file_path, energy)
            
            # Lưu mô hình nếu cần
            if output_directory:
                self._save_model(beam_model, output_directory, energy)
            
            return beam_model
            
        except Exception as e:
            logger.error(f"Lỗi khi xây dựng mô hình cho năng lượng {energy}: {str(e)}")
            raise DataProcessingError(f"Không thể xây dựng mô hình cho năng lượng {energy}") from e
    
    def _process_pdd_data(self, beam_model: BeamModel, file_path: str, energy: str):
        """
        Xử lý dữ liệu PDD (Percentage Depth Dose) từ file Excel.
        
        Parameters
        ----------
        beam_model : BeamModel
            Đối tượng mô hình chùm tia cần cập nhật
        file_path : str
            Đường dẫn đến file Excel
        energy : str
            Năng lượng đang xử lý
        """
        try:
            # Đọc dữ liệu PDD từ sheet tương ứng
            sheet_name = "PDD"
            
            # Cố gắng đọc Excel với pandas
            try:
                pdd_data = pd.read_excel(file_path, sheet_name=sheet_name)
            except Exception as e:
                logger.warning(f"Không thể đọc sheet {sheet_name} từ file {file_path}: {str(e)}")
                logger.info("Đang thử đọc sheet 'PDDs' thay thế...")
                pdd_data = pd.read_excel(file_path, sheet_name="PDDs")
            
            # Tìm các cột cần thiết
            depth_col = next((col for col in pdd_data.columns if "Depth" in col), None)
            
            # Tìm các cột chứa dữ liệu PDD cho các kích thước trường khác nhau
            pdd_cols = {}
            for col in pdd_data.columns:
                if isinstance(col, str) and "x" in col.lower() and "cm" in col.lower():
                    # Trích xuất kích thước trường từ tên cột (vd: "10x10 cm")
                    try:
                        size_str = col.lower().split("cm")[0].strip()
                        if "x" in size_str:
                            x, y = size_str.split("x")
                            size = (float(x.strip()), float(y.strip()))
                            pdd_cols[size] = col
                    except Exception:
                        continue
            
            if not depth_col or not pdd_cols:
                logger.warning(f"Không tìm thấy dữ liệu PDD hợp lệ trong file {file_path}")
                return
            
            # Trích xuất dữ liệu cho từng kích thước trường
            for field_size, col_name in pdd_cols.items():
                # Lọc các hàng có giá trị hợp lệ
                valid_data = pdd_data[[depth_col, col_name]].dropna()
                
                if len(valid_data) < 10:  # Kiểm tra đủ điểm dữ liệu
                    logger.warning(f"Không đủ dữ liệu PDD cho kích thước trường {field_size}")
                    continue
                
                # Trích xuất mảng dữ liệu
                depths = valid_data[depth_col].values
                pdds = valid_data[col_name].values
                
                # Tạo tham số mô hình
                param_name = f"pdd_{field_size[0]}x{field_size[1]}"
                parameter = BeamModelParameter(
                    name=param_name,
                    value_grid=pdds,
                    dimensions=["depth"],
                    units=["cm"],
                    dimension_values=[depths],
                    interpolation_method="cubic"
                )
                
                # Thêm vào mô hình
                beam_model.add_parameter(parameter)
                logger.info(f"Đã thêm dữ liệu PDD cho kích thước trường {field_size}")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu PDD: {str(e)}")
            raise DataProcessingError(f"Không thể xử lý dữ liệu PDD cho năng lượng {energy}") from e
    
    def _process_profile_data(self, beam_model: BeamModel, file_path: str, energy: str):
        """
        Xử lý dữ liệu profile (beam profiles) từ file Excel.
        
        Parameters
        ----------
        beam_model : BeamModel
            Đối tượng mô hình chùm tia cần cập nhật
        file_path : str
            Đường dẫn đến file Excel
        energy : str
            Năng lượng đang xử lý
        """
        try:
            # Đọc dữ liệu profile từ sheet tương ứng
            possible_sheets = ["Profiles", "Profile", "Beam Profiles", "Cross-Plane Profiles"]
            
            profile_data = None
            used_sheet = None
            
            # Thử từng sheet có thể chứa dữ liệu profile
            for sheet_name in possible_sheets:
                try:
                    profile_data = pd.read_excel(file_path, sheet_name=sheet_name)
                    used_sheet = sheet_name
                    break
                except Exception:
                    continue
            
            if profile_data is None:
                logger.warning(f"Không tìm thấy sheet chứa dữ liệu profile trong file {file_path}")
                return
            
            logger.info(f"Đọc dữ liệu profile từ sheet '{used_sheet}'")
            
            # Xử lý dữ liệu profile dựa trên cấu trúc của file Excel
            # Đây chỉ là ví dụ, cần điều chỉnh tùy theo cấu trúc thực tế của file
            
            # Tìm các độ sâu khác nhau trong file
            depths = set()
            field_sizes = set()
            
            for col in profile_data.columns:
                col_str = str(col).lower()
                
                # Tìm thông tin độ sâu và kích thước trường từ tên cột
                if "cm" in col_str:
                    try:
                        # Tìm độ sâu dmax, 5cm, 10cm, 20cm, 30cm
                        if "dmax" in col_str:
                            depths.add(0)  # dmax tạm thời coi là độ sâu 0
                        elif "cm" in col_str:
                            depth_parts = col_str.split("cm")
                            depth = float(depth_parts[0].strip().split()[-1])
                            depths.add(depth)
                        
                        # Tìm kích thước trường 10x10, 20x20, 30x30, 40x40
                        if "x" in col_str:
                            size_parts = col_str.split("x")
                            if len(size_parts) > 1 and "cm" in size_parts[1]:
                                size_x = float(size_parts[0].strip().split()[-1])
                                size_y = float(size_parts[1].split("cm")[0].strip())
                                field_sizes.add((size_x, size_y))
                    except Exception:
                        continue
            
            logger.info(f"Tìm thấy {len(depths)} độ sâu và {len(field_sizes)} kích thước trường cho profile")
            
            # Xử lý dữ liệu profile cho mỗi độ sâu và kích thước trường
            for depth in sorted(depths):
                for field_size in field_sizes:
                    # Tìm cột phù hợp
                    off_axis_col = None
                    profile_col = None
                    
                    depth_str = "dmax" if depth == 0 else f"{depth}cm"
                    field_str = f"{field_size[0]}x{field_size[1]}"
                    
                    for col in profile_data.columns:
                        col_str = str(col).lower()
                        if "off" in col_str and "axis" in col_str:
                            off_axis_col = col
                        
                        if depth_str.lower() in col_str and field_str.lower() in col_str:
                            profile_col = col
                            break
                    
                    if off_axis_col is None or profile_col is None:
                        continue
                    
                    # Trích xuất dữ liệu
                    valid_data = profile_data[[off_axis_col, profile_col]].dropna()
                    
                    if len(valid_data) < 10:  # Kiểm tra đủ điểm dữ liệu
                        continue
                    
                    # Trích xuất mảng dữ liệu
                    off_axis = valid_data[off_axis_col].values
                    profile = valid_data[profile_col].values
                    
                    # Tạo tham số mô hình
                    param_name = f"profile_{field_size[0]}x{field_size[1]}_{depth}cm"
                    parameter = BeamModelParameter(
                        name=param_name,
                        value_grid=profile,
                        dimensions=["off_axis"],
                        units=["cm"],
                        dimension_values=[off_axis],
                        interpolation_method="cubic"
                    )
                    
                    # Thêm vào mô hình
                    beam_model.add_parameter(parameter)
                    logger.info(f"Đã thêm dữ liệu profile cho kích thước trường {field_size} tại độ sâu {depth_str}")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu profile: {str(e)}")
            raise DataProcessingError(f"Không thể xử lý dữ liệu profile cho năng lượng {energy}") from e
    
    def _process_output_factors(self, beam_model: BeamModel, file_path: str, energy: str):
        """
        Xử lý dữ liệu hệ số đầu ra (output factors) từ file Excel.
        
        Parameters
        ----------
        beam_model : BeamModel
            Đối tượng mô hình chùm tia cần cập nhật
        file_path : str
            Đường dẫn đến file Excel
        energy : str
            Năng lượng đang xử lý
        """
        try:
            # Đọc dữ liệu output factor từ sheet tương ứng
            possible_sheets = ["Output Factors", "OF", "Output Factor"]
            
            of_data = None
            used_sheet = None
            
            # Thử từng sheet có thể chứa dữ liệu output factor
            for sheet_name in possible_sheets:
                try:
                    of_data = pd.read_excel(file_path, sheet_name=sheet_name)
                    used_sheet = sheet_name
                    break
                except Exception:
                    continue
            
            if of_data is None:
                logger.warning(f"Không tìm thấy sheet chứa dữ liệu output factor trong file {file_path}")
                return
            
            logger.info(f"Đọc dữ liệu output factor từ sheet '{used_sheet}'")
            
            # Xử lý dữ liệu dựa trên cấu trúc của file Excel
            # Đối với output factors, thường là bảng chéo với X và Y là kích thước trường
            
            # Tìm cột và hàng chứa kích thước trường
            x_sizes = None
            y_index = None
            
            for i, col_name in enumerate(of_data.columns):
                if "field size" in str(col_name).lower() or "size" in str(col_name).lower():
                    y_index = i
                    break
            
            if y_index is not None:
                # Cột đầu tiên là kích thước Y
                y_sizes = of_data.iloc[:, y_index].dropna().values
                # Các cột còn lại là kích thước X và giá trị OF
                x_sizes = of_data.columns[y_index+1:].values
                
                # Tạo lưới output factor
                of_values = []
                x_values = []
                y_values = []
                
                for i, y in enumerate(y_sizes):
                    if not isinstance(y, (int, float)):
                        continue
                    
                    row_data = of_data.iloc[i, y_index+1:].values
                    for j, x in enumerate(x_sizes):
                        if j < len(row_data) and not np.isnan(row_data[j]):
                            if isinstance(x, (int, float)):
                                x_values.append(float(x))
                                y_values.append(float(y))
                                of_values.append(float(row_data[j]))
                
                if len(x_values) > 0:
                    # Tạo tham số output factor 2D
                    unique_x = np.array(sorted(set(x_values)))
                    unique_y = np.array(sorted(set(y_values)))
                    
                    # Tạo lưới 2D
                    of_grid = np.zeros((len(unique_y), len(unique_x)))
                    of_grid.fill(np.nan)
                    
                    # Điền giá trị vào lưới
                    for i in range(len(x_values)):
                        x_idx = np.where(unique_x == x_values[i])[0][0]
                        y_idx = np.where(unique_y == y_values[i])[0][0]
                        of_grid[y_idx, x_idx] = of_values[i]
                    
                    # Điền các giá trị NaN bằng nội suy
                    for i in range(of_grid.shape[0]):
                        for j in range(of_grid.shape[1]):
                            if np.isnan(of_grid[i, j]):
                                # Tìm giá trị tương ứng gần nhất
                                valid_indices = np.where(~np.isnan(of_grid))
                                if len(valid_indices[0]) > 0:
                                    nearest_idx = np.argmin(
                                        (valid_indices[0] - i)**2 + (valid_indices[1] - j)**2
                                    )
                                    of_grid[i, j] = of_grid[valid_indices[0][nearest_idx], valid_indices[1][nearest_idx]]
                    
                    # Tạo tham số mô hình
                    parameter = BeamModelParameter(
                        name="output_factors",
                        value_grid=of_grid,
                        dimensions=["field_size_y", "field_size_x"],
                        units=["cm", "cm"],
                        dimension_values=[unique_y, unique_x],
                        interpolation_method="linear"
                    )
                    
                    # Thêm vào mô hình
                    beam_model.add_parameter(parameter)
                    logger.info("Đã thêm dữ liệu output factor")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu output factor: {str(e)}")
            raise DataProcessingError(f"Không thể xử lý dữ liệu output factor cho năng lượng {energy}") from e
    
    def _save_model(self, beam_model: BeamModel, output_directory: str, energy: str):
        """
        Lưu mô hình chùm tia vào file.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia cần lưu
        output_directory : str
            Thư mục đầu ra
        energy : str
            Năng lượng của mô hình
        """
        try:
            # Tạo thư mục đầu ra nếu chưa tồn tại
            ensure_directory(output_directory)
            
            # Lưu thành file JSON
            output_file = os.path.join(output_directory, f"truebeam_{energy.lower()}_model.json")
            
            # Chuyển mô hình sang dictionary
            model_dict = beam_model.to_dict()
            
            # Lưu file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(model_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Đã lưu mô hình chùm tia {energy} vào {output_file}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            raise DataProcessingError(f"Không thể lưu mô hình cho năng lượng {energy}") from e

class TruebeamModelManager:
    """
    Lớp quản lý các mô hình Truebeam.
    
    Lớp này quản lý việc tạo, lưu trữ và truy xuất các mô hình chùm tia Truebeam
    từ dữ liệu đo được.
    """
    
    def __init__(self, data_directory: str, model_directory: str):
        """
        Khởi tạo manager với thư mục dữ liệu và thư mục mô hình.
        
        Parameters
        ----------
        data_directory : str
            Đường dẫn đến thư mục chứa dữ liệu chùm tia
        model_directory : str
            Đường dẫn đến thư mục lưu trữ mô hình
        """
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.builder = TruebeamEnergyModelBuilder(data_directory)
        
        # Tạo thư mục mô hình nếu chưa tồn tại
        ensure_directory(model_directory)
        
        # Tải các mô hình đã có
        self.available_models = self._scan_available_models()
        logger.info(f"Tìm thấy {len(self.available_models)} mô hình có sẵn trong {model_directory}")
    
    def _scan_available_models(self) -> Dict[str, str]:
        """
        Quét thư mục mô hình để tìm các mô hình có sẵn.
        
        Returns
        -------
        Dict[str, str]
            Dictionary với khóa là năng lượng và giá trị là đường dẫn đến file mô hình
        """
        available_models = {}
        
        if not os.path.exists(self.model_directory):
            return available_models
        
        for file in os.listdir(self.model_directory):
            if file.endswith(".json") and file.startswith("truebeam_"):
                # Trích xuất năng lượng từ tên file
                energy = file.replace("truebeam_", "").replace("_model.json", "").upper()
                if energy:
                    available_models[energy] = os.path.join(self.model_directory, file)
        
        return available_models
    
    def get_available_energies(self) -> List[str]:
        """
        Lấy danh sách các năng lượng có sẵn cho mô hình.
        
        Returns
        -------
        List[str]
            Danh sách tên các năng lượng
        """
        # Kết hợp các năng lượng có sẵn trong dữ liệu và mô hình
        all_energies = set(self.builder.energy_files.keys()) | set(self.available_models.keys())
        return sorted(list(all_energies))
    
    def load_model(self, energy: str) -> Optional[BeamModel]:
        """
        Tải mô hình cho một năng lượng cụ thể.
        
        Parameters
        ----------
        energy : str
            Năng lượng cần tải mô hình (ví dụ: "6MV", "10FFF")
            
        Returns
        -------
        Optional[BeamModel]
            Mô hình chùm tia được tải, hoặc None nếu không tìm thấy
        """
        energy_key = energy.upper()
        
        # Kiểm tra xem đã có mô hình lưu sẵn chưa
        if energy_key in self.available_models:
            file_path = self.available_models[energy_key]
            try:
                # Đọc file JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    model_dict = json.load(f)
                
                # Tạo mô hình từ dictionary
                beam_model = BeamModel.from_dict(model_dict)
                logger.info(f"Đã tải mô hình {energy} từ {file_path}")
                
                return beam_model
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình từ {file_path}: {str(e)}")
        
        # Nếu không có mô hình có sẵn, thử tạo mới từ dữ liệu
        if energy_key in self.builder.energy_files:
            logger.info(f"Không tìm thấy mô hình có sẵn cho {energy}, đang tạo mới...")
            try:
                beam_model = self.builder.build_model(energy_key, self.model_directory)
                
                # Cập nhật danh sách mô hình có sẵn
                if beam_model:
                    self.available_models = self._scan_available_models()
                
                return beam_model
            except Exception as e:
                logger.error(f"Không thể tạo mô hình cho {energy}: {str(e)}")
        
        logger.warning(f"Không tìm thấy dữ liệu hoặc mô hình cho năng lượng {energy}")
        return None
    
    def build_all_models(self):
        """
        Xây dựng mô hình cho tất cả các năng lượng có sẵn trong dữ liệu.
        """
        for energy in self.builder.energy_files.keys():
            try:
                logger.info(f"Đang xây dựng mô hình cho năng lượng {energy}...")
                self.builder.build_model(energy, self.model_directory)
                logger.info(f"Đã xây dựng xong mô hình cho năng lượng {energy}")
            except Exception as e:
                logger.error(f"Lỗi khi xây dựng mô hình cho năng lượng {energy}: {str(e)}")
        
        # Cập nhật danh sách mô hình có sẵn
        self.available_models = self._scan_available_models()
        logger.info(f"Đã xây dựng tổng cộng {len(self.available_models)} mô hình") 