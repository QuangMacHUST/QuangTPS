"""
Module phân tích DVH (Dose Volume Histogram) cho đánh giá kế hoạch xạ trị.

Module này cung cấp các hàm phân tích nâng cao cho dữ liệu DVH, bao gồm tính toán các
chỉ số đánh giá kế hoạch, so sánh kế hoạch, và phân tích theo các ràng buộc lâm sàng.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pandas as pd
from scipy.optimize import minimize_scalar

from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh, _get_dose_at_volume, _get_volume_at_dose

logger = logging.getLogger(__name__)

class DVHAnalysis:
    """
    Lớp cung cấp các phương thức phân tích DVH nâng cao.
    """
    
    def __init__(self, dvh_data: Dict[str, Any], structure_name: str = None):
        """
        Khởi tạo đối tượng phân tích DVH.
        
        Parameters:
            dvh_data (Dict[str, Any]): Dữ liệu DVH từ hàm calculate_dvh
            structure_name (str, optional): Tên cấu trúc
        """
        self.dvh_data = dvh_data
        self.structure_name = structure_name
        
        # Trích xuất dữ liệu cơ bản từ dvh_data
        self.dose_bins = dvh_data['dose_bins']
        self.cumulative_dvh = dvh_data['cumulative']
        self.differential_dvh = dvh_data['differential']
        self.dose_unit = dvh_data['dose_unit']
        self.volume_type = dvh_data['volume_type']
        
        # Kiểm tra xem có phải DVH rỗng không
        self.is_empty = np.all(self.cumulative_dvh == 0)
    
    def get_dx(self, volume_percent: float) -> float:
        """
        Lấy giá trị liều phủ x% thể tích (Dx).
        
        Parameters:
            volume_percent (float): Phần trăm thể tích (0-100)
            
        Returns:
            float: Giá trị liều tại phần trăm thể tích
            
        Raises:
            ValueError: Nếu phần trăm thể tích nằm ngoài phạm vi 0-100
        """
        if self.is_empty:
            return 0.0
            
        if volume_percent < 0 or volume_percent > 100:
            raise ValueError(f"Volume percentage must be between 0 and 100, got {volume_percent}")
        
        return _get_dose_at_volume(self.dose_bins, self.cumulative_dvh, volume_percent)
    
    def get_vx(self, dose: float, relative_to_prescription: bool = False, prescription_dose: float = None) -> float:
        """
        Lấy phần trăm thể tích nhận liều >= x Gy (Vx).
        
        Parameters:
            dose (float): Giá trị liều
            relative_to_prescription (bool, optional): Liều là phần trăm của liều kê đơn
            prescription_dose (float, optional): Liều kê đơn nếu relative_to_prescription là True
            
        Returns:
            float: Phần trăm thể tích nhận liều >= x
            
        Raises:
            ValueError: Nếu relative_to_prescription là True nhưng không cung cấp prescription_dose
        """
        if self.is_empty:
            return 0.0
            
        if relative_to_prescription and prescription_dose is None:
            raise ValueError("prescription_dose must be provided when relative_to_prescription is True")
        
        # Chuyển đổi liều nếu cần
        target_dose = dose * prescription_dose / 100 if relative_to_prescription else dose
        
        return _get_volume_at_dose(self.dose_bins, self.cumulative_dvh, target_dose)
    
    def get_effective_volume(self, parameter_a: float) -> float:
        """
        Tính thể tích hiệu quả (veff) cho mô hình gEUD.
        
        veff = (Σ(vi * Di^a))^(1/a)
        
        Parameters:
            parameter_a (float): Tham số a trong mô hình gEUD
            
        Returns:
            float: Thể tích hiệu quả
        """
        if self.is_empty:
            return 0.0
            
        # Lấy thể tích vi phân từ DVH
        diff_volumes = self.differential_dvh
        
        # Chuẩn hóa thể tích vi phân để tổng = 1
        if self.volume_type == 'relative':
            norm_volumes = diff_volumes / 100.0
        else:
            norm_volumes = diff_volumes / np.sum(diff_volumes)
        
        # Tính veff
        if parameter_a == 0:
            # Trường hợp đặc biệt: lim(a->0) = exp(Σ(vi * ln(Di)))
            # Tránh log(0) bằng cách chỉ tính các bin có liều > 0
            mask = self.dose_bins > 0
            if np.any(mask):
                log_dose = np.log(self.dose_bins[mask])
                geo_mean = np.exp(np.sum(norm_volumes[mask] * log_dose))
                return geo_mean
            else:
                return 0.0
        else:
            # Công thức thông thường
            veff = np.power(np.sum(norm_volumes * np.power(self.dose_bins, parameter_a)), 1.0/parameter_a)
            return veff
    
    def get_equivalent_uniform_dose(self, parameter_a: float) -> float:
        """
        Tính liều đồng nhất tương đương (EUD).
        
        EUD = (Σ(vi * Di^a))^(1/a)
        
        Parameters:
            parameter_a (float): Tham số a trong mô hình EUD (âm cho cơ quan song song, dương cho cơ quan nối tiếp)
            
        Returns:
            float: Giá trị EUD
        """
        if self.is_empty:
            return 0.0
            
        # Lấy thể tích vi phân từ DVH
        diff_volumes = self.differential_dvh
        
        # Chuẩn hóa thể tích vi phân để tổng = 1
        if self.volume_type == 'relative':
            norm_volumes = diff_volumes / 100.0
        else:
            norm_volumes = diff_volumes / np.sum(diff_volumes)
        
        # Tính EUD
        return self.get_effective_volume(parameter_a)
    
    def get_homogeneity_index(self, prescription_dose: float, method: str = 'icru83') -> float:
        """
        Tính chỉ số đồng nhất (Homogeneity Index - HI).
        
        Parameters:
            prescription_dose (float): Liều kê đơn
            method (str, optional): Phương pháp tính HI ('icru83', 'rtog', 'paddick')
            
        Returns:
            float: Chỉ số đồng nhất
            
        Raises:
            ValueError: Nếu phương pháp không được hỗ trợ
        """
        if self.is_empty:
            return float('nan')
            
        if method.lower() == 'icru83':
            # HI = (D2% - D98%) / D50%
            d2 = self.get_dx(2)
            d98 = self.get_dx(98)
            d50 = self.get_dx(50)
            
            if d50 == 0:
                return float('nan')
                
            return (d2 - d98) / d50
            
        elif method.lower() == 'rtog':
            # HI = Dmax / prescription_dose
            dmax = self.dvh_data['max_dose']
            
            if prescription_dose == 0:
                return float('nan')
                
            return dmax / prescription_dose
            
        elif method.lower() == 'paddick':
            # HI = (D5% - D95%) / prescription_dose
            d5 = self.get_dx(5)
            d95 = self.get_dx(95)
            
            if prescription_dose == 0:
                return float('nan')
                
            return (d5 - d95) / prescription_dose
            
        else:
            raise ValueError(f"Unsupported homogeneity index method: {method}")
    
    def get_conformity_index(self, prescription_dose: float, reference_volume: Optional[float] = None, method: str = 'paddick') -> float:
        """
        Tính chỉ số phù hợp (Conformity Index - CI).
        
        Parameters:
            prescription_dose (float): Liều kê đơn
            reference_volume (float, optional): Thể tích tham chiếu (thường là thể tích PTV)
            method (str, optional): Phương pháp tính CI ('paddick', 'rtog', 'lomax')
            
        Returns:
            float: Chỉ số phù hợp
            
        Raises:
            ValueError: Nếu phương pháp không được hỗ trợ
        """
        if self.is_empty:
            return float('nan')
            
        # Tính thể tích nhận được ít nhất là liều kê đơn
        v_rx = self.get_vx(prescription_dose)
        
        # Chuyển đổi từ phần trăm sang thể tích tuyệt đối nếu cần
        if self.volume_type == 'relative':
            v_rx_abs = v_rx * self.dvh_data['structure_volume'] / 100.0
        else:
            v_rx_abs = v_rx
        
        if method.lower() == 'rtog':
            # CI = V_rx / V_ref
            if reference_volume is None:
                logger.warning("reference_volume not provided for RTOG CI, using structure volume")
                reference_volume = self.dvh_data['structure_volume']
                
            if reference_volume == 0:
                return float('nan')
                
            return v_rx_abs / reference_volume
            
        elif method.lower() == 'paddick':
            # CI = (TV_rx)^2 / (TV * V_rx)
            # TV_rx: thể tích target nhận được ít nhất là liều kê đơn
            # TV: tổng thể tích target
            # V_rx: tổng thể tích nhận được ít nhất là liều kê đơn
            
            if reference_volume is None:
                logger.warning("reference_volume not provided for Paddick CI, assuming the structure is the target")
                tv_rx = v_rx_abs
                tv = self.dvh_data['structure_volume']
            else:
                # Giả định structure_volume là thể tích nằm trong intersection của target và isodose
                tv_rx = v_rx_abs
                tv = reference_volume
            
            if tv == 0 or v_rx_abs == 0:
                return float('nan')
                
            return (tv_rx ** 2) / (tv * v_rx_abs)
            
        elif method.lower() == 'lomax':
            # CI = 1 - |1 - V_rx/V_ref|
            if reference_volume is None:
                logger.warning("reference_volume not provided for Lomax CI, using structure volume")
                reference_volume = self.dvh_data['structure_volume']
                
            if reference_volume == 0:
                return float('nan')
                
            return 1 - abs(1 - v_rx_abs / reference_volume)
            
        else:
            raise ValueError(f"Unsupported conformity index method: {method}")
    
    def get_gradient_index(self, high_dose: float, low_dose: float = None, ratio: float = 0.5) -> float:
        """
        Tính chỉ số gradient (Gradient Index - GI).
        
        GI = V_low / V_high
        
        Parameters:
            high_dose (float): Liều cao (thường là liều kê đơn)
            low_dose (float, optional): Liều thấp, nếu None thì sẽ là high_dose * ratio
            ratio (float, optional): Tỉ lệ của low_dose so với high_dose nếu low_dose không được cung cấp
            
        Returns:
            float: Chỉ số gradient
        """
        if self.is_empty:
            return float('nan')
            
        # Tính low_dose nếu không được cung cấp
        if low_dose is None:
            low_dose = high_dose * ratio
        
        # Tính thể tích tương ứng
        v_high = self.get_vx(high_dose)
        v_low = self.get_vx(low_dose)
        
        if v_high == 0:
            return float('nan')
            
        return v_low / v_high
    
    def get_dose_spillage(self, prescription_dose: float, r50_reference_volume: Optional[float] = None) -> Dict[str, float]:
        """
        Tính liều spillage (tràn) theo tiêu chuẩn RTOG.
        
        Parameters:
            prescription_dose (float): Liều kê đơn
            r50_reference_volume (float, optional): Thể tích tham chiếu cho R50% (thường là thể tích PTV)
            
        Returns:
            Dict[str, float]: Các chỉ số liều spillage
        """
        if self.is_empty:
            return {
                'R50%': float('nan'),
                'R27%': float('nan'),
                'D2cm': float('nan')
            }
            
        # Tính thể tích nhận được các liều khác nhau
        v_100 = self.get_vx(prescription_dose)
        v_50 = self.get_vx(prescription_dose * 0.5)
        v_27 = self.get_vx(prescription_dose * 0.27)
        
        # Sử dụng structure_volume nếu không cung cấp r50_reference_volume
        if r50_reference_volume is None:
            r50_reference_volume = self.dvh_data['structure_volume']
        
        # Chuyển đổi từ phần trăm sang thể tích tuyệt đối nếu cần
        if self.volume_type == 'relative':
            v_100_abs = v_100 * self.dvh_data['structure_volume'] / 100.0
            v_50_abs = v_50 * self.dvh_data['structure_volume'] / 100.0
            v_27_abs = v_27 * self.dvh_data['structure_volume'] / 100.0
        else:
            v_100_abs = v_100
            v_50_abs = v_50
            v_27_abs = v_27
        
        # Tính các chỉ số
        if v_100_abs == 0 or r50_reference_volume == 0:
            r_50 = float('nan')
        else:
            r_50 = v_50_abs / r50_reference_volume
            
        if v_100_abs == 0:
            r_27 = float('nan')
        else:
            r_27 = v_27_abs / v_100_abs
        
        # Chú ý: D2cm đòi hỏi thông tin không gian 3D mà không có trong DVH đơn thuần
        # Ở đây chỉ trả về NaN
        d_2cm = float('nan')
        
        return {
            'R50%': r_50,
            'R27%': r_27,
            'D2cm': d_2cm
        }
    
    def get_integral_dose(self, density: float = 1.0) -> float:
        """
        Tính liều tích phân (Integral Dose).
        
        ID = Σ(Di * vi * ρ)
        
        Parameters:
            density (float, optional): Mật độ mô (g/cm³)
            
        Returns:
            float: Liều tích phân (Gy*cc hoặc J/kg)
        """
        if self.is_empty:
            return 0.0
            
        # Lấy DVH vi phân
        diff_volumes = self.differential_dvh
        
        # Chuyển đổi sang thể tích tuyệt đối (cc) nếu cần
        if self.volume_type == 'relative':
            structure_volume = self.dvh_data.get('structure_volume_cc', 
                                               self.dvh_data['structure_volume'])
            volumes_cc = diff_volumes * structure_volume / 100.0
        else:
            volumes_cc = diff_volumes
        
        # Tính liều tích phân
        integral_dose = np.sum(self.dose_bins * volumes_cc) * density
        
        return integral_dose
    
    def get_conformation_number(self, prescription_dose: float, target_volume: float) -> float:
        """
        Tính chỉ số conformation (Conformation Number - CN).
        
        CN = (TV_rx / TV) * (TV_rx / V_rx)
        
        Parameters:
            prescription_dose (float): Liều kê đơn
            target_volume (float): Thể tích target (cấu trúc mục tiêu)
            
        Returns:
            float: Chỉ số conformation
        """
        if self.is_empty or target_volume == 0:
            return float('nan')
            
        # Tính thể tích nhận được ít nhất là liều kê đơn
        v_rx = self.get_vx(prescription_dose)
        
        # Chuyển đổi từ phần trăm sang thể tích tuyệt đối nếu cần
        if self.volume_type == 'relative':
            structure_volume = self.dvh_data.get('structure_volume_cc', 
                                               self.dvh_data['structure_volume'])
            v_rx_abs = v_rx * structure_volume / 100.0
        else:
            v_rx_abs = v_rx
        
        # Giả định structure là target, nên TV_rx = v_rx_abs
        tv_rx = v_rx_abs
        
        # Tính CN
        if v_rx_abs == 0:
            return 0.0
            
        return (tv_rx / target_volume) * (tv_rx / v_rx_abs)
    
    def get_dose_statistics(self) -> Dict[str, float]:
        """
        Lấy các thống kê liều từ DVH.
        
        Returns:
            Dict[str, float]: Thống kê liều
        """
        if self.is_empty:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'modal': 0.0,
                'std': 0.0
            }
            
        # Trích xuất các giá trị từ dvh_data
        min_dose = self.dvh_data['min_dose']
        max_dose = self.dvh_data['max_dose']
        mean_dose = self.dvh_data['mean_dose']
        median_dose = self.dvh_data['median_dose']
        modal_dose = self.dvh_data.get('modal_dose', 0.0)
        
        # Tính độ lệch chuẩn từ DVH vi phân
        diff_volumes = self.differential_dvh
        
        # Chuẩn hóa thể tích vi phân để tổng = 1
        if self.volume_type == 'relative':
            norm_volumes = diff_volumes / 100.0
        else:
            norm_volumes = diff_volumes / np.sum(diff_volumes)
        
        # Tính độ lệch chuẩn
        variance = np.sum(norm_volumes * (self.dose_bins - mean_dose)**2)
        std_dose = np.sqrt(variance)
        
        return {
            'min': min_dose,
            'max': max_dose,
            'mean': mean_dose,
            'median': median_dose,
            'modal': modal_dose,
            'std': std_dose
        }
    
    def check_dose_constraints(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Kiểm tra ràng buộc liều cho cấu trúc.
        
        Parameters:
            constraints (List[Dict]): Danh sách các ràng buộc, mỗi ràng buộc là một dict với các key:
                - type: Loại ràng buộc ('Dx', 'Vx', 'Max', 'Mean', 'Min', 'EUD', ...)
                - value: Giá trị giới hạn
                - direction: Hướng so sánh ('≤', '≥', '=')
                - priority: Độ ưu tiên ('mandatory', 'high', 'medium', 'low')
                - unit: Đơn vị ('Gy', '%', 'cc', ...)
                - params: Tham số bổ sung (nếu cần)
            
        Returns:
            List[Dict]: Danh sách các ràng buộc với kết quả kiểm tra
        """
        if self.is_empty:
            return [{**constraint, 'result': float('nan'), 'passed': False} 
                    for constraint in constraints]
            
        results = []
        
        for constraint in constraints:
            constraint_type = constraint.get('type', '')
            constraint_value = constraint.get('value', 0)
            constraint_direction = constraint.get('direction', '≤')
            constraint_params = constraint.get('params', {})
            
            # Tính giá trị thực tế
            actual_value = float('nan')
            
            if constraint_type.startswith('D') and constraint_type[1:].isdigit():
                # Dx - Liều phủ x% thể tích
                volume_percent = float(constraint_type[1:])
                actual_value = self.get_dx(volume_percent)
                
            elif constraint_type.startswith('V'):
                # Vx - Phần trăm thể tích nhận liều >= x
                dose_value = float(constraint_type[1:])
                rx_dose = constraint_params.get('prescription_dose')
                relative = constraint_params.get('relative_to_prescription', False)
                actual_value = self.get_vx(dose_value, relative, rx_dose)
                
            elif constraint_type == 'Max' or constraint_type == 'Maximum':
                actual_value = self.dvh_data['max_dose']
                
            elif constraint_type == 'Mean':
                actual_value = self.dvh_data['mean_dose']
                
            elif constraint_type == 'Min' or constraint_type == 'Minimum':
                actual_value = self.dvh_data['min_dose']
                
            elif constraint_type == 'Median':
                actual_value = self.dvh_data['median_dose']
                
            elif constraint_type == 'EUD':
                parameter_a = constraint_params.get('a', 1.0)
                actual_value = self.get_equivalent_uniform_dose(parameter_a)
                
            elif constraint_type == 'HI':
                prescription_dose = constraint_params.get('prescription_dose', 0)
                method = constraint_params.get('method', 'icru83')
                actual_value = self.get_homogeneity_index(prescription_dose, method)
                
            elif constraint_type == 'CI':
                prescription_dose = constraint_params.get('prescription_dose', 0)
                ref_volume = constraint_params.get('reference_volume')
                method = constraint_params.get('method', 'paddick')
                actual_value = self.get_conformity_index(prescription_dose, ref_volume, method)
                
            elif constraint_type == 'GI':
                high_dose = constraint_params.get('high_dose', 0)
                low_dose = constraint_params.get('low_dose')
                ratio = constraint_params.get('ratio', 0.5)
                actual_value = self.get_gradient_index(high_dose, low_dose, ratio)
            
            # Kiểm tra kết quả so với ràng buộc
            passed = False
            if constraint_direction == '≤' or constraint_direction == '<=':
                passed = actual_value <= constraint_value
            elif constraint_direction == '≥' or constraint_direction == '>=':
                passed = actual_value >= constraint_value
            elif constraint_direction == '=':
                # Cho phép sai số nhỏ
                tolerance = constraint_params.get('tolerance', 0.01)
                passed = abs(actual_value - constraint_value) <= tolerance
            
            # Thêm kết quả vào danh sách
            results.append({
                **constraint,
                'actual': actual_value,
                'passed': passed
            })
        
        return results
    
    def find_dose_for_volume(self, target_volume: float, initial_guess: Optional[float] = None) -> float:
        """
        Tìm liều tại thể tích mục tiêu cụ thể.
        
        Parameters:
            target_volume (float): Phần trăm thể tích mục tiêu (0-100)
            initial_guess (float, optional): Giá trị liều ban đầu để tìm kiếm
            
        Returns:
            float: Giá trị liều tại phần trăm thể tích mục tiêu
        """
        if self.is_empty or target_volume < 0 or target_volume > 100:
            return float('nan')
            
        # Dùng nội suy tuyến tính đơn giản vì đã có hàm _get_dose_at_volume
        return self.get_dx(target_volume)
    
    def find_volume_for_dose(self, target_dose: float) -> float:
        """
        Tìm thể tích nhận liều >= mục tiêu cụ thể.
        
        Parameters:
            target_dose (float): Giá trị liều mục tiêu
            
        Returns:
            float: Phần trăm thể tích nhận liều >= target_dose
        """
        if self.is_empty:
            return 0.0
            
        # Dùng nội suy tuyến tính đơn giản vì đã có hàm _get_volume_at_dose
        return self.get_vx(target_dose)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu DVH thành DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu DVH
        """
        # Tạo DataFrame
        df = pd.DataFrame({
            'Dose': self.dose_bins,
            'Differential_Volume': self.differential_dvh,
            'Cumulative_Volume': self.cumulative_dvh
        })
        
        # Thêm metadata
        df.attrs['structure_name'] = self.structure_name
        df.attrs['dose_unit'] = self.dose_unit
        df.attrs['volume_type'] = self.volume_type
        df.attrs['min_dose'] = self.dvh_data['min_dose']
        df.attrs['max_dose'] = self.dvh_data['max_dose']
        df.attrs['mean_dose'] = self.dvh_data['mean_dose']
        df.attrs['median_dose'] = self.dvh_data['median_dose']
        
        return df
