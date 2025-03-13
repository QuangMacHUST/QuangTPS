"""
Module định nghĩa các hàm mục tiêu (objective functions) cho tối ưu hóa kế hoạch xạ trị.

Module này cung cấp các hàm mục tiêu khác nhau được sử dụng trong quá trình tối ưu hóa kế hoạch 
xạ trị để đạt được phân bố liều mong muốn cho cấu trúc đích và bảo vệ các cơ quan nguy cấp.
Các hàm mục tiêu này được sử dụng bởi các thuật toán tối ưu hóa để tìm ra kế hoạch tối ưu.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field

from quangtps.evaluation.dvh import calculate_dvh, calculate_dvh_metrics
from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.constants import EPSILON

logger = logging.getLogger(__name__)

@dataclass
class ObjectiveBase:
    """Lớp cơ sở cho các hàm mục tiêu tối ưu hóa."""
    structure_name: str
    weight: float = 1.0
    is_enabled: bool = True
    objective_type: str = "None"
    
    def __post_init__(self):
        """Xác thực các tham số sau khi khởi tạo."""
        if self.weight < 0:
            raise ValueError(f"Trọng số phải là giá trị không âm, nhận được: {self.weight}")
    
    def evaluate(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> float:
        """
        Đánh giá hàm mục tiêu với phân bố liều và cấu trúc hiện tại.
        
        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            Giá trị của hàm mục tiêu (cost)
        """
        if not self.is_enabled:
            return 0.0
            
        if self.structure_name not in structures:
            logger.warning(f"Cấu trúc '{self.structure_name}' không tồn tại trong structures")
            return 0.0
            
        return self._calculate_cost(dose_grid, structures[self.structure_name])
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính toán giá trị cost dựa trên phân bố liều và mặt nạ cấu trúc.
        
        Phương thức này cần được ghi đè trong các lớp con.
        """
        raise NotImplementedError("Phương thức này phải được triển khai trong lớp con")
    
    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin mô tả về hàm mục tiêu."""
        return {
            "structure_name": self.structure_name,
            "type": self.objective_type,
            "weight": self.weight,
            "is_enabled": self.is_enabled
        }

@dataclass
class MinDose(ObjectiveBase):
    """Mục tiêu liều tối thiểu cho cấu trúc (thường dùng cho PTV)."""
    dose: float  # Liều mục tiêu, đơn vị Gy
    objective_type: str = "MinDose"
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty cho các voxel trong cấu trúc có liều nhỏ hơn dose.
        
        Công thức: sum((dose - D_i)^2) cho các D_i < dose
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        # Tính toán penalty cho các voxel dưới liều mong muốn
        under_dose = np.maximum(0, self.dose - structure_dose)
        
        # Tính tổng bình phương của các vi phạm
        cost = np.sum(under_dose**2)
        
        # Chuẩn hóa theo số voxel
        if structure_mask.sum() > 0:
            cost /= structure_mask.sum()
        
        return cost * self.weight

@dataclass
class MaxDose(ObjectiveBase):
    """Mục tiêu liều tối đa cho cấu trúc (thường dùng cho OAR)."""
    dose: float  # Liều giới hạn, đơn vị Gy
    objective_type: str = "MaxDose"
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty cho các voxel trong cấu trúc có liều lớn hơn dose.
        
        Công thức: sum((D_i - dose)^2) cho các D_i > dose
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        # Tính toán penalty cho các voxel vượt liều mong muốn
        over_dose = np.maximum(0, structure_dose - self.dose)
        
        # Tính tổng bình phương của các vi phạm
        cost = np.sum(over_dose**2)
        
        # Chuẩn hóa theo số voxel
        if structure_mask.sum() > 0:
            cost /= structure_mask.sum()
        
        return cost * self.weight

@dataclass
class UniformDose(ObjectiveBase):
    """Mục tiêu liều đồng nhất cho cấu trúc (thường dùng cho PTV)."""
    dose: float  # Liều mong muốn, đơn vị Gy
    objective_type: str = "UniformDose"
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty cho các voxel trong cấu trúc có liều khác dose.
        
        Công thức: sum((D_i - dose)^2) cho tất cả D_i
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        # Tính độ lệch so với liều mong muốn
        dose_diff = structure_dose - self.dose
        
        # Tính tổng bình phương của độ lệch
        cost = np.sum(dose_diff**2)
        
        # Chuẩn hóa theo số voxel
        if structure_mask.sum() > 0:
            cost /= structure_mask.sum()
        
        return cost * self.weight

@dataclass
class MeanDose(ObjectiveBase):
    """Mục tiêu giới hạn liều trung bình cho cấu trúc (thường dùng cho OAR)."""
    dose: float  # Liều trung bình mục tiêu, đơn vị Gy
    objective_type: str = "MeanDose"
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty khi liều trung bình của cấu trúc vượt quá dose.
        
        Công thức: (mean_dose - dose)^2 nếu mean_dose > dose
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        # Tính liều trung bình
        mean_dose = np.mean(structure_dose) if len(structure_dose) > 0 else 0
        
        # Tính penalty nếu vượt quá liều mong muốn
        over_dose = max(0, mean_dose - self.dose)
        cost = over_dose**2
        
        return cost * self.weight

@dataclass
class DoseVolume(ObjectiveBase):
    """Mục tiêu giới hạn thể tích nhận liều mức nào đó (DVH constraint)."""
    dose: float  # Liều đòi hỏi, đơn vị Gy
    volume_percent: float  # Phần trăm thể tích
    direction: str = "upper"  # "upper" hoặc "lower"
    objective_type: str = "DoseVolume"
    
    def __post_init__(self):
        """Xác thực các tham số sau khi khởi tạo."""
        super().__post_init__()
        if self.volume_percent < 0 or self.volume_percent > 100:
            raise ValueError(f"volume_percent phải nằm trong khoảng [0, 100], nhận được: {self.volume_percent}")
        
        if self.direction not in ["upper", "lower"]:
            raise ValueError(f"direction phải là 'upper' hoặc 'lower', nhận được: {self.direction}")
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty khi vi phạm ràng buộc liều-thể tích.
        
        Upper: penalty khi có quá {volume_percent}% thể tích nhận liều >= {dose}
        Lower: penalty khi có ít hơn {volume_percent}% thể tích nhận liều >= {dose}
        """
        # Tính DVH
        dvh = calculate_dvh(
            dose_array=dose_grid.dose_array,
            structure_mask=structure_mask,
            volume_type='relative'
        )
        
        # Tính thể tích thực tế nhận liều >= dose
        volume_at_dose = np.interp(
            self.dose,
            dvh['dose_bins'],
            100 - dvh['cumulative_volume']
        )
        
        if self.direction == "upper":
            # Penalty khi vượt quá thể tích cho phép
            violation = max(0, volume_at_dose - self.volume_percent)
        else:  # "lower"
            # Penalty khi không đạt thể tích yêu cầu
            violation = max(0, self.volume_percent - volume_at_dose)
        
        # Bình phương vi phạm
        cost = violation**2
        
        return cost * self.weight

@dataclass
class ConformityIndex(ObjectiveBase):
    """Mục tiêu tối ưu chỉ số đồng dạng (thường dùng cho PTV)."""
    reference_dose: float  # Liều tham chiếu, thường là liều chỉ định, đơn vị Gy
    target_ci: float = 1.0  # Chỉ số đồng dạng mục tiêu, thường là 1.0 (lý tưởng)
    objective_type: str = "ConformityIndex"
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty dựa trên độ lệch của chỉ số đồng dạng so với target_ci.
        
        CI = (V_ref / V_target) * (V_ref / V_ref_total)
        Trong đó:
            V_ref: thể tích PTV nhận >= liều tham chiếu
            V_target: tổng thể tích PTV
            V_ref_total: tổng thể tích (cả PTV và ngoài PTV) nhận >= liều tham chiếu
        """
        # Tính thể tích PTV
        v_target = np.sum(structure_mask)
        
        # Tạo mặt nạ cho vùng nhận liều >= reference_dose
        ref_dose_mask = dose_grid.dose_array >= self.reference_dose
        
        # Tính V_ref_total - tổng thể tích nhận liều >= reference_dose
        v_ref_total = np.sum(ref_dose_mask)
        
        # Tính V_ref - thể tích PTV nhận liều >= reference_dose
        v_ref = np.sum(ref_dose_mask & structure_mask)
        
        # Tránh chia cho 0
        if v_target == 0 or v_ref_total == 0:
            return 0
        
        # Tính chỉ số đồng dạng (Paddick CI)
        ci = (v_ref * v_ref) / (v_target * v_ref_total)
        
        # Tính penalty dựa trên độ lệch so với target_ci
        cost = (ci - self.target_ci)**2
        
        return cost * self.weight

@dataclass
class HomogeneityIndex(ObjectiveBase):
    """Mục tiêu tối ưu chỉ số đồng nhất (thường dùng cho PTV)."""
    prescription_dose: float  # Liều chỉ định, đơn vị Gy
    target_hi: float = 0.0  # Chỉ số đồng nhất mục tiêu, 0 là lý tưởng (hoàn toàn đồng nhất)
    method: str = "icru83"  # Phương pháp tính HI: "icru83" hoặc "d5_d95"
    objective_type: str = "HomogeneityIndex"
    
    def __post_init__(self):
        """Xác thực các tham số sau khi khởi tạo."""
        super().__post_init__()
        if self.method not in ["icru83", "d5_d95"]:
            raise ValueError(f"method phải là 'icru83' hoặc 'd5_d95', nhận được: {self.method}")
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty dựa trên độ lệch của chỉ số đồng nhất so với target_hi.
        
        HI ICRU83 = (D2% - D98%) / D50%
        HI d5_d95 = (D5% - D95%) / prescription_dose
        """
        # Tính DVH
        dvh = calculate_dvh(
            dose_array=dose_grid.dose_array,
            structure_mask=structure_mask,
            volume_type='relative'
        )
        
        # Tính các giá trị liều cần thiết
        metrics = ['D2', 'D5', 'D50', 'D95', 'D98']
        dvh_metrics = calculate_dvh_metrics(dvh, metrics, self.prescription_dose)
        
        # Tính chỉ số đồng nhất theo phương pháp được chọn
        if self.method == "icru83":
            if dvh_metrics['D50'] > EPSILON:
                hi = (dvh_metrics['D2'] - dvh_metrics['D98']) / dvh_metrics['D50']
            else:
                hi = 0
        else:  # "d5_d95"
            if self.prescription_dose > EPSILON:
                hi = (dvh_metrics['D5'] - dvh_metrics['D95']) / self.prescription_dose
            else:
                hi = 0
        
        # Tính penalty dựa trên độ lệch so với target_hi
        cost = (hi - self.target_hi)**2
        
        return cost * self.weight

@dataclass
class GradientIndex(ObjectiveBase):
    """Mục tiêu tối ưu chỉ số gradient (thường dùng cho SRS/SBRT)."""
    reference_dose: float  # Liều tham chiếu cao, đơn vị Gy
    low_dose: Optional[float] = None  # Liều thấp, đơn vị Gy, mặc định = reference_dose/2
    target_gi: float = 3.0  # Chỉ số gradient mục tiêu
    objective_type: str = "GradientIndex"
    
    def __post_init__(self):
        """Xác thực và hoàn thiện các tham số sau khi khởi tạo."""
        super().__post_init__()
        if self.low_dose is None:
            self.low_dose = self.reference_dose / 2.0
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty dựa trên độ lệch của chỉ số gradient so với target_gi.
        
        GI = V(low_dose) / V(reference_dose)
        """
        # Tạo mặt nạ cho vùng nhận liều >= reference_dose
        high_dose_mask = dose_grid.dose_array >= self.reference_dose
        
        # Tạo mặt nạ cho vùng nhận liều >= low_dose
        low_dose_mask = dose_grid.dose_array >= self.low_dose
        
        # Tính thể tích tương ứng
        v_high = np.sum(high_dose_mask)
        v_low = np.sum(low_dose_mask)
        
        # Tránh chia cho 0
        if v_high == 0:
            return 0
        
        # Tính chỉ số gradient
        gi = v_low / v_high
        
        # Tính penalty dựa trên độ lệch so với target_gi
        cost = (gi - self.target_gi)**2
        
        return cost * self.weight

@dataclass
class EUDObjective(ObjectiveBase):
    """Mục tiêu dựa trên liều đồng nhất tương đương (EUD)."""
    target_eud: float  # EUD mục tiêu, đơn vị Gy
    parameter_a: float  # Tham số a điều chỉnh độ nhạy (a > 0 cho PTV, a < 0 cho OAR)
    direction: str = "upper"  # "upper" hoặc "lower"
    objective_type: str = "EUD"
    
    def __post_init__(self):
        """Xác thực các tham số sau khi khởi tạo."""
        super().__post_init__()
        if self.direction not in ["upper", "lower"]:
            raise ValueError(f"direction phải là 'upper' hoặc 'lower', nhận được: {self.direction}")
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty dựa trên độ lệch của EUD so với target_eud.
        
        EUD = (1/N * sum(D_i^a))^(1/a)
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        # Nếu không có voxel nào, trả về 0
        if len(structure_dose) == 0:
            return 0
        
        # Tính giá trị EUD
        # Thêm EPSILON để tránh lỗi khi dose = 0
        dose_plus_eps = structure_dose + EPSILON
        
        # Tính trung bình của dose^a
        mean_pow_dose = np.mean(dose_plus_eps**self.parameter_a)
        
        # Tính EUD
        eud = mean_pow_dose**(1.0/self.parameter_a)
        
        # Tính penalty dựa trên hướng và đích
        if self.direction == "upper":
            # Penalty khi EUD vượt quá target_eud
            violation = max(0, eud - self.target_eud)
        else:  # "lower"
            # Penalty khi EUD thấp hơn target_eud
            violation = max(0, self.target_eud - eud)
        
        # Bình phương vi phạm
        cost = violation**2
        
        return cost * self.weight

@dataclass
class FalloffObjective(ObjectiveBase):
    """Mục tiêu kiểm soát gradient liều xung quanh cấu trúc đích (dose falloff)."""
    high_dose: float  # Liều cao, thường là liều chỉ định, đơn vị Gy
    low_dose: float  # Liều thấp, đơn vị Gy
    falloff_distance: float  # Khoảng cách mong muốn (mm), liều giảm từ high_dose xuống low_dose
    objective_type: str = "Falloff"
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính penalty cho gradient liều quá dốc hoặc quá thoải.
        
        Công thức: sử dụng khoảng cách dự kiến và so sánh với gradient thực tế
        """
        # Chưa triển khai đầy đủ - cần thêm thuật toán tính khoảng cách 3D
        # Đây là phiên bản đơn giản hóa
        
        # Tạo mặt nạ cho vùng nhận liều >= high_dose
        high_dose_mask = dose_grid.dose_array >= self.high_dose
        
        # Tạo mặt nạ cho vùng nhận liều >= low_dose
        low_dose_mask = dose_grid.dose_array >= self.low_dose
        
        # Tính thể tích tương ứng
        v_high = np.sum(high_dose_mask)
        v_low = np.sum(low_dose_mask)
        
        # Tránh chia cho 0
        if v_high == 0:
            return 0
        
        # Tính tỷ lệ thể tích
        vol_ratio = v_low / v_high
        
        # Tính penalty dựa trên tỷ lệ thể tích
        # Trong phiên bản đầy đủ, sẽ tính toán dựa trên khoảng cách thực
        falloff_cost = vol_ratio * self.falloff_distance / 10.0
        
        return falloff_cost * self.weight

# Dictionary chứa tất cả các loại objective có sẵn để dễ dàng tạo mới
OBJECTIVE_TYPES = {
    "MinDose": MinDose,
    "MaxDose": MaxDose,
    "UniformDose": UniformDose,
    "MeanDose": MeanDose,
    "DoseVolume": DoseVolume,
    "ConformityIndex": ConformityIndex,
    "HomogeneityIndex": HomogeneityIndex,
    "GradientIndex": GradientIndex,
    "EUD": EUDObjective,
    "Falloff": FalloffObjective
}

def create_objective(objective_type: str, **kwargs) -> ObjectiveBase:
    """
    Tạo đối tượng objective từ loại và tham số.
    
    Args:
        objective_type: Loại objective ('MinDose', 'MaxDose', etc.)
        **kwargs: Các tham số cần thiết cho loại objective đó
        
    Returns:
        Đối tượng objective đã được tạo
        
    Raises:
        ValueError: Nếu objective_type không được hỗ trợ
    """
    if objective_type not in OBJECTIVE_TYPES:
        raise ValueError(f"Loại objective không hợp lệ: {objective_type}. "
                         f"Các loại được hỗ trợ: {list(OBJECTIVE_TYPES.keys())}")
    
    return OBJECTIVE_TYPES[objective_type](**kwargs)

class ObjectiveCollection:
    """Tập hợp nhiều hàm mục tiêu để đánh giá tổng thể kế hoạch."""
    
    def __init__(self):
        """Khởi tạo danh sách hàm mục tiêu trống."""
        self.objectives: List[ObjectiveBase] = []
    
    def add_objective(self, objective: ObjectiveBase) -> None:
        """Thêm một hàm mục tiêu vào danh sách."""
        self.objectives.append(objective)
    
    def remove_objective(self, index: int) -> None:
        """Xóa một hàm mục tiêu từ danh sách theo chỉ số."""
        if 0 <= index < len(self.objectives):
            del self.objectives[index]
        else:
            raise IndexError(f"Chỉ số không hợp lệ: {index}")
    
    def enable_objective(self, index: int, enabled: bool = True) -> None:
        """Bật/tắt một hàm mục tiêu theo chỉ số."""
        if 0 <= index < len(self.objectives):
            self.objectives[index].is_enabled = enabled
        else:
            raise IndexError(f"Chỉ số không hợp lệ: {index}")
    
    def evaluate_all(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Đánh giá tất cả các hàm mục tiêu với phân bố liều và cấu trúc hiện tại.
        
        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            Dictionary chứa kết quả đánh giá của từng hàm mục tiêu và tổng cost
        """
        results = {}
        total_cost = 0.0
        
        for i, objective in enumerate(self.objectives):
            if not objective.is_enabled:
                results[f"objective_{i}"] = 0.0
                continue
                
            cost = objective.evaluate(dose_grid, structures)
            results[f"objective_{i}"] = cost
            total_cost += cost
        
        results["total_cost"] = total_cost
        return results
    
    def get_objectives_info(self) -> List[Dict[str, Any]]:
        """Trả về thông tin mô tả về tất cả các hàm mục tiêu."""
        return [obj.get_info() for obj in self.objectives]
    
    def __len__(self) -> int:
        """Trả về số lượng hàm mục tiêu trong danh sách."""
        return len(self.objectives)
