#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module xử lý dữ liệu chùm tia cho tính toán liều.

Module này cung cấp các lớp và hàm xử lý dữ liệu chùm tia đo được (như Truebeam Beam Data) 
thành các mô hình chùm tia được sử dụng cho tính toán liều trong các thuật toán khác nhau 
như Collapsed Cone, Pencil Beam, AAA, v.v.
"""

import os
import logging
import numpy as np
import scipy.interpolate as interp
from typing import Dict, List, Tuple, Any, Optional, Union

from quangtps.core.exceptions import DataProcessingError
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

class BeamModelParameter:
    """
    Lớp lưu trữ tham số mô hình chùm tia.
    
    Lớp này đại diện cho một tham số cụ thể của mô hình chùm tia, 
    ví dụ như: PDD/TMR, off-axis ratio, output factor, v.v.
    """
    
    def __init__(self, name: str, value_grid: np.ndarray, dimensions: List[str], units: List[str], 
                 dimension_values: List[np.ndarray], interpolation_method: str = 'linear'):
        """
        Khởi tạo tham số mô hình chùm tia.
        
        Parameters
        ----------
        name : str
            Tên tham số
        value_grid : np.ndarray
            Mảng giá trị (có thể 1D, 2D, 3D,... tùy theo số chiều của tham số)
        dimensions : List[str]
            Danh sách tên các chiều (ví dụ: ["depth", "off_axis"])
        units : List[str]
            Đơn vị của từng chiều (ví dụ: ["cm", "cm"])
        dimension_values : List[np.ndarray]
            Giá trị của từng chiều
        interpolation_method : str, optional
            Phương pháp nội suy mặc định
        """
        self.name = name
        self.value_grid = value_grid
        self.dimensions = dimensions
        self.units = units
        self.dimension_values = dimension_values
        self.interpolation_method = interpolation_method
        
        # Tạo bộ nội suy
        self._create_interpolator()
    
    def _create_interpolator(self):
        """Tạo bộ nội suy cho tham số này."""
        # Kiểm tra kích thước chiều
        if len(self.dimensions) != len(self.dimension_values):
            raise ValueError("Dimension lists and values must have the same length")
            
        dims = len(self.dimensions)
        
        if dims == 1:
            self.interpolator = interp.interp1d(
                self.dimension_values[0], self.value_grid, 
                kind=self.interpolation_method, bounds_error=False, fill_value="extrapolate"
            )
        else:
            # Tạo lưới điểm nội suy
            points = tuple(self.dimension_values)
            
            # Tạo interpolator
            self.interpolator = interp.RegularGridInterpolator(
                points, self.value_grid, 
                method=self.interpolation_method, bounds_error=False, fill_value=None
            )
    
    def get_value(self, *args) -> float:
        """
        Lấy giá trị nội suy tại điểm cụ thể.
        
        Parameters
        ----------
        *args : float
            Giá trị của các chiều theo thứ tự đã định nghĩa
            
        Returns
        -------
        float
            Giá trị nội suy
        """
        if len(args) != len(self.dimensions):
            raise ValueError(f"Expected {len(self.dimensions)} values, got {len(args)}")
        
        if len(self.dimensions) == 1:
            return float(self.interpolator(args[0]))
        else:
            # Trường hợp nhiều chiều
            point = np.array(args)
            return float(self.interpolator(point))
    
    def __repr__(self):
        return f"BeamModelParameter(name='{self.name}', dimensions={self.dimensions})"


class BeamModel:
    """
    Lớp mô hình chùm tia.
    
    Lớp này đại diện cho một mô hình chùm tia hoàn chỉnh, bao gồm các tham số khác nhau
    cho thuật toán tính toán liều, được xây dựng từ dữ liệu đo lường thực tế.
    """
    
    def __init__(self, name: str, energy: str, beam_type: str = "PHOTON"):
        """
        Khởi tạo mô hình chùm tia.
        
        Parameters
        ----------
        name : str
            Tên mô hình
        energy : str
            Năng lượng chùm tia (ví dụ: "6MV", "10MV FFF")
        beam_type : str, optional
            Loại chùm tia ("PHOTON", "ELECTRON", "PHOTON_FFF")
        """
        self.name = name
        self.energy = energy
        self.beam_type = beam_type
        self.parameters = {}
        self.metadata = {}
    
    def add_parameter(self, parameter: BeamModelParameter):
        """
        Thêm tham số vào mô hình.
        
        Parameters
        ----------
        parameter : BeamModelParameter
            Tham số cần thêm
        """
        self.parameters[parameter.name] = parameter
    
    def get_parameter(self, name: str) -> Optional[BeamModelParameter]:
        """
        Lấy tham số mô hình theo tên.
        
        Parameters
        ----------
        name : str
            Tên tham số cần lấy
            
        Returns
        -------
        Optional[BeamModelParameter]
            Tham số tìm thấy hoặc None nếu không tồn tại
        """
        return self.parameters.get(name, None)
    
    def get_parameter_names(self) -> List[str]:
        """
        Lấy danh sách tên các tham số.
        
        Returns
        -------
        List[str]
            Danh sách tên các tham số
        """
        return list(self.parameters.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi mô hình chùm tia thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin mô hình chùm tia
        """
        # Chuyển các tham số thành dạng serializable
        params_dict = {}
        for name, param in self.parameters.items():
            params_dict[name] = {
                "dimensions": param.dimensions,
                "units": param.units,
                "dimension_values": [values.tolist() for values in param.dimension_values],
                "value_grid": param.value_grid.tolist(),
                "interpolation_method": param.interpolation_method
            }
        
        return {
            "name": self.name,
            "energy": self.energy,
            "beam_type": self.beam_type,
            "parameters": params_dict,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamModel':
        """
        Tạo mô hình chùm tia từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin mô hình chùm tia
            
        Returns
        -------
        BeamModel
            Mô hình chùm tia
        """
        model = cls(
            name=data["name"],
            energy=data["energy"],
            beam_type=data["beam_type"]
        )
        
        # Phục hồi metadata
        model.metadata = data.get("metadata", {})
        
        # Phục hồi các tham số
        params_dict = data.get("parameters", {})
        for name, param_data in params_dict.items():
            dimension_values = [np.array(values) for values in param_data["dimension_values"]]
            value_grid = np.array(param_data["value_grid"])
            
            param = BeamModelParameter(
                name=name,
                value_grid=value_grid,
                dimensions=param_data["dimensions"],
                units=param_data["units"],
                dimension_values=dimension_values,
                interpolation_method=param_data.get("interpolation_method", "linear")
            )
            
            model.add_parameter(param)
        
        return model

    def set_metadata(self, key: str, value: Any):
        """
        Thiết lập metadata cho mô hình.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        value : Any
            Giá trị metadata
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị metadata.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        default : Any, optional
            Giá trị mặc định nếu không tìm thấy khóa
            
        Returns
        -------
        Any
            Giá trị metadata
        """
        return self.metadata.get(key, default)
    
    def save_to_json(self, filepath: str):
        """
        Lưu mô hình chùm tia vào file JSON.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn đến file JSON
        """
        import json
        
        # Tạo đối tượng JSON
        beam_model_json = {
            "name": self.name,
            "energy": self.energy,
            "beam_type": self.beam_type,
            "metadata": self.metadata,
            "parameters": {}
        }
        
        # Chuyển đổi các tham số
        for name, param in self.parameters.items():
            param_json = {
                "name": param.name,
                "dimensions": param.dimensions,
                "units": param.units,
                "interpolation_method": param.interpolation_method,
                "dimension_values": [values.tolist() for values in param.dimension_values],
                "value_grid": param.value_grid.tolist()
            }
            beam_model_json["parameters"][name] = param_json
        
        # Ghi vào file
        with open(filepath, 'w') as f:
            json.dump(beam_model_json, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'BeamModel':
        """
        Tải mô hình chùm tia từ file JSON.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn đến file JSON
            
        Returns
        -------
        BeamModel
            Mô hình chùm tia đã tải
        """
        import json
        
        # Đọc file JSON
        with open(filepath, 'r') as f:
            beam_model_json = json.load(f)
        
        # Tạo mô hình
        beam_model = cls(
            name=beam_model_json["name"],
            energy=beam_model_json["energy"],
            beam_type=beam_model_json["beam_type"]
        )
        
        # Thiết lập metadata
        beam_model.metadata = beam_model_json["metadata"]
        
        # Tạo các tham số
        for name, param_json in beam_model_json["parameters"].items():
            # Chuyển đổi các mảng về numpy
            dimension_values = [np.array(values) for values in param_json["dimension_values"]]
            value_grid = np.array(param_json["value_grid"])
            
            # Tạo tham số
            param = BeamModelParameter(
                name=param_json["name"],
                dimensions=param_json["dimensions"],
                units=param_json["units"],
                dimension_values=dimension_values,
                value_grid=value_grid,
                interpolation_method=param_json["interpolation_method"]
            )
            
            # Thêm vào mô hình
            beam_model.add_parameter(param)
        
        return beam_model
    
    def calculate_dose(self, point: np.ndarray, parameters: Dict[str, Any]) -> float:
        """
        Tính toán liều tại một điểm dựa trên mô hình.
        
        Parameters
        ----------
        point : np.ndarray
            Tọa độ điểm cần tính liều
        parameters : Dict[str, Any]
            Các tham số cần thiết cho tính toán
            
        Returns
        -------
        float
            Giá trị liều tại điểm
        """
        # Phương thức này cần được triển khai cụ thể cho từng loại mô hình
        raise NotImplementedError("Phương thức này cần được triển khai trong lớp con")


class BeamModelFactory:
    """
    Lớp tạo mô hình chùm tia.
    
    Lớp này cung cấp các phương thức để tạo mô hình chùm tia từ dữ liệu đo lường thực tế,
    như dữ liệu từ máy TrueBeam.
    """
    
    @staticmethod
    def create_from_truebeam_data(data_directory: str, energy: str, 
                                output_file: Optional[str] = None) -> BeamModel:
        """
        Tạo mô hình chùm tia từ dữ liệu TrueBeam.
        
        Parameters
        ----------
        data_directory : str
            Thư mục chứa dữ liệu TrueBeam
        energy : str
            Năng lượng chùm tia (ví dụ: "6MV", "10FFF")
        output_file : str, optional
            Đường dẫn file để lưu mô hình, nếu None thì không lưu
            
        Returns
        -------
        BeamModel
            Mô hình chùm tia đã tạo
        """
        try:
            # Tạo tên mô hình
            model_name = f"TrueBeam_{energy}"
            
            # Xác định loại chùm tia
            if "FFF" in energy.upper():
                beam_type = "PHOTON_FFF"
            elif "MV" in energy.upper() or "X" in energy.upper():
                beam_type = "PHOTON"
            elif "E" in energy.upper():
                beam_type = "ELECTRON"
            else:
                beam_type = "PHOTON"
            
            # Khởi tạo mô hình
            beam_model = BeamModel(name=model_name, energy=energy, beam_type=beam_type)
            
            # Đọc dữ liệu TrueBeam
            # Import TrueBeamDataReader ở đây để tránh circular import
            from quangtps.treatment.beams.beam_data_importer import TrueBeamDataReader
            reader = TrueBeamDataReader()
            
            # Quét thư mục dữ liệu
            energy_files = reader.scan_directory(data_directory)
            
            if energy not in energy_files:
                available = ", ".join(energy_files.keys())
                raise DataProcessingError(f"Không tìm thấy dữ liệu cho năng lượng {energy}. Các năng lượng có sẵn: {available}")
            
            # Đọc dữ liệu chùm tia
            beam_data = reader.read_beam_data(energy_files[energy])
            
            # Xử lý từng loại dữ liệu
            BeamModelFactory._process_pdd_data(beam_model, beam_data)
            BeamModelFactory._process_profile_data(beam_model, beam_data)
            BeamModelFactory._process_output_factors(beam_model, beam_data)
            
            # Xử lý wedge factor nếu có
            if 'wedge' in beam_data:
                BeamModelFactory._process_wedge_factors(beam_model, beam_data)
            
            # Thêm metadata
            beam_model.metadata = {
                "source": "TrueBeam",
                "creation_date": BeamModelFactory._get_current_date(),
                "data_file": energy_files[energy]
            }
            
            # Lưu mô hình nếu cần
            if output_file:
                model_dict = beam_model.to_dict()
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                with open(output_file, 'w') as f:
                    import json
                    json.dump(model_dict, f, indent=2)
            
            return beam_model
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo mô hình chùm tia TrueBeam: {str(e)}")
            raise DataProcessingError(f"Không thể tạo mô hình chùm tia từ dữ liệu TrueBeam: {str(e)}")
    
    @staticmethod
    def _get_current_date():
        """Lấy ngày hiện tại theo định dạng chuẩn."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def _process_pdd_data(beam_model: BeamModel, beam_data: Dict[str, Any]):
        """
        Xử lý dữ liệu PDD (Percentage Depth Dose) từ TrueBeam.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia cần cập nhật
        beam_data : Dict[str, Any]
            Dữ liệu beam từ TrueBeam reader
        """
        # Khai báo các biến để lưu trữ dữ liệu PDD
        depths = []
        pdd_values = {}
        field_sizes = []
        
        # Tìm các file chứa dữ liệu PDD
        for file_path, data in beam_data.items():
            if ('PDD' in file_path or 'GGPB' in file_path) and isinstance(data, dict):
                if 'depths' in data and 'values' in data and 'field_sizes' in data:
                    # Lưu dữ liệu cho các kích thước trường khác nhau
                    for i, fs in enumerate(data['field_sizes']):
                        if fs not in field_sizes:
                            field_sizes.append(fs)
                        
                        fs_key = f"{fs:.1f}"
                        if fs_key not in pdd_values:
                            pdd_values[fs_key] = []
                        
                        # Đảm bảo chúng ta có dữ liệu cho tất cả độ sâu
                        for d_idx, depth in enumerate(data['depths']):
                            if depth not in depths:
                                depths.append(depth)
                            
                            if i < data['values'].shape[1]:
                                pdd_values[fs_key].append((depth, data['values'][d_idx, i]))
        
        # Sắp xếp các độ sâu
        depths = sorted(depths)
        field_sizes = sorted(field_sizes)
        
        if not depths or not field_sizes:
            logger.warning("No PDD data found in TrueBeam data")
            return
        
        # Tạo lưới dữ liệu PDD
        pdd_grid = np.zeros((len(depths), len(field_sizes)))
        
        # Điền giá trị PDD vào lưới
        for fs_idx, fs in enumerate(field_sizes):
            fs_key = f"{fs:.1f}"
            if fs_key in pdd_values:
                # Nội suy giá trị cho mỗi độ sâu
                fs_data = pdd_values[fs_key]
                fs_depths = [d for d, _ in fs_data]
                fs_pdd = [p for _, p in fs_data]
                
                if len(fs_depths) > 1:
                    interp_func = interp.interp1d(
                        fs_depths, fs_pdd, kind='linear', 
                        bounds_error=False, fill_value="extrapolate"
                    )
                    
                    for d_idx, depth in enumerate(depths):
                        pdd_grid[d_idx, fs_idx] = interp_func(depth)
        
        # Tạo và thêm tham số PDD vào mô hình
        pdd_param = BeamModelParameter(
            name="PDD",
            value_grid=pdd_grid,
            dimensions=["depth", "field_size"],
            units=["cm", "cm"],
            dimension_values=[np.array(depths), np.array(field_sizes)],
            interpolation_method="linear"
        )
        
        beam_model.add_parameter(pdd_param)
        logger.info(f"Added PDD parameter to beam model with {len(depths)} depths and {len(field_sizes)} field sizes")
    
    @staticmethod
    def _process_profile_data(beam_model: BeamModel, beam_data: Dict[str, Any]):
        """
        Xử lý dữ liệu profile từ TrueBeam.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia cần cập nhật
        beam_data : Dict[str, Any]
            Dữ liệu beam từ TrueBeam reader
        """
        # Cài đặt hàm này tương tự như _process_pdd_data nhưng cho profile data
        # Các profile thường được lưu trong sheet khác nhau của file Excel TrueBeam
        
        is_fff = "FFF" in beam_model.energy
        
        # Tìm file Excel chứa dữ liệu profile
        profile_data = None
        for file_path, data in beam_data.items():
            if file_path.endswith('.xlsx') and isinstance(data, dict) and 'sheets' in data:
                if (is_fff and 'data_type' in data and data['data_type'] == BeamDataType.FFF_PROFILE) or \
                   (not is_fff and 'data_type' in data and data['data_type'] == BeamDataType.PROFILE):
                    profile_data = data
                    break
        
        if not profile_data:
            logger.warning(f"No profile data found for {beam_model.energy}")
            return
        
        # Xử lý từng loại profile (crossline/inline) từ các sheet khác nhau
        # Đây là một công việc phức tạp và phụ thuộc vào cấu trúc dữ liệu cụ thể của TrueBeam
        # Phần này nên được điều chỉnh dựa trên cấu trúc dữ liệu thực tế
        
        # Ví dụ đơn giản: tạo một profile mẫu
        # Trong thực tế, dữ liệu này sẽ được đọc từ file Excel
        
        # Profile theo trục X (crossline)
        positions = np.linspace(-20, 20, 41)  # cm
        depths = np.array([5.0, 10.0, 20.0])  # cm
        field_sizes = np.array([5.0, 10.0, 20.0])  # cm
        
        # Tạo dữ liệu profile mẫu
        profile_grid = np.zeros((len(positions), len(depths), len(field_sizes)))
        
        # Điền dữ liệu mẫu (Gaussian profiles)
        for d_idx, depth in enumerate(depths):
            for fs_idx, fs in enumerate(field_sizes):
                sigma = fs / 4.0  # standard deviation dựa trên kích thước trường
                
                # Tạo profile penumbra
                for pos_idx, pos in enumerate(positions):
                    if abs(pos) <= fs/2:
                        # Trong trường
                        if is_fff:
                            # FFF có cường độ cao hơn ở giữa
                            value = 1.0 - 0.02 * (pos / (fs/2))**2
                        else:
                            # Profile phẳng
                            value = 1.0
                    else:
                        # Penumbra region
                        dist_from_edge = abs(pos) - fs/2
                        value = np.exp(-0.5 * (dist_from_edge / sigma)**2)
                    
                    profile_grid[pos_idx, d_idx, fs_idx] = value
        
        # Tạo và thêm tham số profile vào mô hình
        profile_param = BeamModelParameter(
            name="Profile",
            value_grid=profile_grid,
            dimensions=["off_axis", "depth", "field_size"],
            units=["cm", "cm", "cm"],
            dimension_values=[positions, depths, field_sizes],
            interpolation_method="linear"
        )
        
        beam_model.add_parameter(profile_param)
        logger.info(f"Added Profile parameter to beam model")
    
    @staticmethod
    def _process_output_factors(beam_model: BeamModel, beam_data: Dict[str, Any]):
        """
        Xử lý hệ số output từ TrueBeam.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia cần cập nhật
        beam_data : Dict[str, Any]
            Dữ liệu beam từ TrueBeam reader
        """
        # Tìm các file chứa dữ liệu output factor
        for file_path, data in beam_data.items():
            if 'Open_OF' in file_path and isinstance(data, dict):
                if 'depths' in data and 'values' in data and 'field_sizes' in data:
                    # Tạo dữ liệu output factor
                    field_sizes = np.array(data['field_sizes'])
                    
                    # Lấy output factor ở độ sâu tham chiếu (thường là 10cm)
                    ref_depth_idx = 0
                    for i, depth in enumerate(data['depths']):
                        if abs(depth - 10.0) < 0.1:  # Tìm độ sâu gần 10cm
                            ref_depth_idx = i
                            break
                    
                    # Lấy hàng tương ứng với độ sâu tham chiếu
                    if ref_depth_idx < data['values'].shape[0]:
                        output_factors = data['values'][ref_depth_idx, :]
                        
                        # Chuẩn hóa theo trường tham chiếu (thường là 10x10cm)
                        ref_field_idx = 0
                        for i, fs in enumerate(field_sizes):
                            if abs(fs - 10.0) < 0.1:  # Tìm field size gần 10x10
                                ref_field_idx = i
                                break
                        
                        if ref_field_idx < len(output_factors):
                            ref_value = output_factors[ref_field_idx]
                            output_factors = output_factors / ref_value
                            
                            # Tạo và thêm tham số output factor vào mô hình
                            of_param = BeamModelParameter(
                                name="OutputFactor",
                                value_grid=output_factors,
                                dimensions=["field_size"],
                                units=["cm"],
                                dimension_values=[field_sizes],
                                interpolation_method="linear"
                            )
                            
                            beam_model.add_parameter(of_param)
                            logger.info(f"Added OutputFactor parameter to beam model")
                            return
        
        logger.warning(f"No output factor data found for {beam_model.energy}")
    
    @staticmethod
    def _process_wedge_factors(beam_model: BeamModel, beam_data: Dict[str, Any]):
        """
        Xử lý hệ số wedge từ TrueBeam.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia cần cập nhật
        beam_data : Dict[str, Any]
            Dữ liệu beam từ TrueBeam reader
        """
        # Danh sách góc wedge phổ biến
        wedge_angles = [15, 30, 45, 60]
        wedge_factors = {}
        
        # Tìm các file chứa dữ liệu wedge factor
        for angle in wedge_angles:
            for file_path, data in beam_data.items():
                wedge_pattern = f"W{angle}"
                if wedge_pattern in file_path and isinstance(data, dict):
                    if 'depths' in data and 'values' in data and 'field_sizes' in data:
                        # Tạo dữ liệu wedge factor
                        field_sizes = np.array(data['field_sizes'])
                        
                        # Lấy wedge factor ở độ sâu tham chiếu (thường là 10cm)
                        ref_depth_idx = 0
                        for i, depth in enumerate(data['depths']):
                            if abs(depth - 10.0) < 0.1:  # Tìm độ sâu gần 10cm
                                ref_depth_idx = i
                                break
                        
                        # Lấy hàng tương ứng với độ sâu tham chiếu
                        if ref_depth_idx < data['values'].shape[0]:
                            wedge_factors[angle] = data['values'][ref_depth_idx, :]
        
        if not wedge_factors:
            logger.warning(f"No wedge factor data found for {beam_model.energy}")
            return
        
        # Tổ chức dữ liệu wedge factor
        angles = sorted(wedge_factors.keys())
        wf_grid = np.zeros((len(angles), len(field_sizes)))
        
        for i, angle in enumerate(angles):
            wf_grid[i, :] = wedge_factors[angle]
        
        # Tạo và thêm tham số wedge factor vào mô hình
        wf_param = BeamModelParameter(
            name="WedgeFactor",
            value_grid=wf_grid,
            dimensions=["wedge_angle", "field_size"],
            units=["degree", "cm"],
            dimension_values=[np.array(angles), field_sizes],
            interpolation_method="linear"
        )
        
        beam_model.add_parameter(wf_param)
        logger.info(f"Added WedgeFactor parameter to beam model for angles {angles}")


class BeamDataProcessor:
    """
    Lớp xử lý dữ liệu chùm tia.
    
    Lớp này cung cấp các phương thức để xử lý dữ liệu chùm tia cho các thuật toán
    tính toán liều khác nhau, chuyển đổi từ dữ liệu đo lường thành tham số thuật toán.
    """
    
    @staticmethod
    def prepare_ccc_beam_model(beam_model: BeamModel) -> Dict[str, Any]:
        """
        Chuẩn bị mô hình chùm tia cho thuật toán Collapsed Cone Convolution.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia đầu vào
            
        Returns
        -------
        Dict[str, Any]
            Tham số mô hình chùm tia cho thuật toán CCC
        """
        # Khởi tạo kết quả
        ccc_params = {
            "energy": beam_model.energy,
            "beam_type": beam_model.beam_type,
            "pdd": None,
            "off_axis_ratio": None,
            "output_factor": None,
            "wedge_factor": None,
            "fluence_map": None
        }
        
        # Lấy tham số PDD
        pdd_param = beam_model.get_parameter("PDD")
        if pdd_param:
            ccc_params["pdd"] = {
                "depths": pdd_param.dimension_values[0].tolist(),
                "field_sizes": pdd_param.dimension_values[1].tolist(),
                "values": pdd_param.value_grid.tolist()
            }
        
        # Lấy tham số Profile
        profile_param = beam_model.get_parameter("Profile")
        if profile_param:
            ccc_params["off_axis_ratio"] = {
                "positions": profile_param.dimension_values[0].tolist(),
                "depths": profile_param.dimension_values[1].tolist(),
                "field_sizes": profile_param.dimension_values[2].tolist(),
                "values": profile_param.value_grid.tolist()
            }
        
        # Lấy tham số Output Factor
        of_param = beam_model.get_parameter("OutputFactor")
        if of_param:
            ccc_params["output_factor"] = {
                "field_sizes": of_param.dimension_values[0].tolist(),
                "values": of_param.value_grid.tolist()
            }
        
        # Lấy tham số Wedge Factor
        wf_param = beam_model.get_parameter("WedgeFactor")
        if wf_param:
            ccc_params["wedge_factor"] = {
                "wedge_angles": wf_param.dimension_values[0].tolist(),
                "field_sizes": wf_param.dimension_values[1].tolist(),
                "values": wf_param.value_grid.tolist()
            }
        
        # Tạo fluence map mặc định
        if "FFF" in beam_model.energy:
            # FFF có profile không phẳng
            size = 101  # 101x101 grid
            center = size // 2
            fluence_map = np.ones((size, size))
            
            # Tạo profile FFF
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2) / center
                    if dist <= 1.0:
                        # Hình dạng FFF điển hình
                        fluence_map[i, j] = 1.0 + 0.4 * (1.0 - dist**2)
            
            ccc_params["fluence_map"] = fluence_map.tolist()
        
        return ccc_params
    
    @staticmethod
    def prepare_pencil_beam_model(beam_model: BeamModel) -> Dict[str, Any]:
        """
        Chuẩn bị mô hình chùm tia cho thuật toán Pencil Beam.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia đầu vào
            
        Returns
        -------
        Dict[str, Any]
            Tham số mô hình chùm tia cho thuật toán Pencil Beam
        """
        # Tương tự như prepare_ccc_beam_model nhưng dành cho PB
        # Thuật toán Pencil Beam cần thêm các tham số kernel
        
        # Lấy tham số từ CCC làm cơ sở
        pb_params = BeamDataProcessor.prepare_ccc_beam_model(beam_model)
        
        # Thêm các tham số kernel
        # Đây là tham số đơn giản hóa, trong thực tế kernel phức tạp hơn nhiều
        pb_params["kernel"] = {
            "type": "double_gaussian",
            "parameters": {
                "primary_sigma": 0.3,  # cm
                "scatter_sigma": 1.0,  # cm
                "primary_weight": 0.85,
                "scatter_weight": 0.15
            }
        }
        
        return pb_params
    
    @staticmethod
    def prepare_aaa_beam_model(beam_model: BeamModel) -> Dict[str, Any]:
        """
        Chuẩn bị mô hình chùm tia cho thuật toán AAA.
        
        Parameters
        ----------
        beam_model : BeamModel
            Mô hình chùm tia đầu vào
            
        Returns
        -------
        Dict[str, Any]
            Tham số mô hình chùm tia cho thuật toán AAA
        """
        # Tương tự như prepare_ccc_beam_model nhưng dành cho AAA
        # AAA cần thêm các tham số về kernel phân tán năng lượng
        
        # Lấy tham số từ CCC làm cơ sở
        aaa_params = BeamDataProcessor.prepare_ccc_beam_model(beam_model)
        
        # Thêm các tham số kernel AAA
        aaa_params["kernel"] = {
            "type": "aaa_kernel",
            "parameters": {
                # Tham số kernel đơn giản hóa
                "photon_sigma": [0.2, 0.5, 1.0],  # cm
                "photon_weight": [0.6, 0.3, 0.1],
                "electron_sigma": [0.1, 0.3],  # cm
                "electron_weight": [0.7, 0.3]
            }
        }
        
        return aaa_params
    
    @staticmethod
    def apply_beam_model_to_dose_calculation(
        dose_grid: DoseGrid,
        beam_model: Dict[str, Any],
        algorithm: str,
        geometrical_params: Dict[str, Any]
    ) -> DoseGrid:
        """
        Áp dụng mô hình chùm tia vào tính toán liều.
        
        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều cần tính
        beam_model : Dict[str, Any]
            Mô hình chùm tia đã chuẩn bị
        algorithm : str
            Thuật toán tính liều ("CCC", "PencilBeam", "AAA")
        geometrical_params : Dict[str, Any]
            Tham số hình học của chùm tia
            
        Returns
        -------
        DoseGrid
            Lưới liều đã tính
        """
        # Phương thức này sẽ kết nối với các thuật toán tính liều
        # Và truyền mô hình chùm tia vào
        
        # TODO: Triển khai kết nối với các thuật toán tính liều
        
        # Đây là phương thức giả để minh họa cách sử dụng
        logger.info(f"Applying {algorithm} beam model to dose calculation")
        
        # Trong thực tế, sẽ gọi module thuật toán tương ứng
        # Ví dụ:
        # if algorithm == "CCC":
        #     from quangtps.dose.algorithms.ccc import calculate_dose
        #     return calculate_dose(dose_grid, beam_model, geometrical_params)
        # elif algorithm == "PencilBeam":
        #     ...
        
        return dose_grid 