"""
Module định nghĩa các ràng buộc (constraints) cho kế hoạch xạ trị.

Module này cung cấp các loại ràng buộc khác nhau được sử dụng trong quá trình tối ưu hóa kế hoạch 
xạ trị để đảm bảo liều cho cơ quan nguy cấp (OAR) nằm trong giới hạn cho phép và liều
cho vùng điều trị (PTV) đạt được mục tiêu.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field

from quangtps.evaluation.dvh import calculate_dvh, calculate_dvh_metrics
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

@dataclass
class ConstraintBase:
    """Lớp cơ sở cho các ràng buộc kế hoạch xạ trị."""
    structure_name: str
    is_enabled: bool = True
    priority: int = 1  # Mức độ ưu tiên: 1 (cao nhất) - 5 (thấp nhất)
    constraint_type: str = "None"
    is_hard_constraint: bool = False  # True nếu là ràng buộc bắt buộc
    
    def __post_init__(self):
        """Xác thực các tham số sau khi khởi tạo."""
        if self.priority < 1 or self.priority > 5:
            raise ValueError(f"Mức độ ưu tiên phải trong khoảng [1, 5], nhận được: {self.priority}")
    
    def check(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> Tuple[bool, float]:
        """
        Kiểm tra xem ràng buộc có được thỏa mãn không.
        
        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        if not self.is_enabled:
            return True, 0.0
            
        if self.structure_name not in structures:
            logger.warning(f"Cấu trúc '{self.structure_name}' không tồn tại trong structures")
            return True, 0.0
            
        return self._evaluate_constraint(dose_grid, structures[self.structure_name])
    
    def _evaluate_constraint(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Đánh giá ràng buộc dựa trên phân bố liều và mặt nạ cấu trúc.
        
        Phương thức này cần được ghi đè trong các lớp con.
        """
        raise NotImplementedError("Phương thức này phải được triển khai trong lớp con")
    
    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin mô tả về ràng buộc."""
        return {
            "structure_name": self.structure_name,
            "type": self.constraint_type,
            "priority": self.priority,
            "is_enabled": self.is_enabled,
            "is_hard_constraint": self.is_hard_constraint
        }
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Constraint cho {self.structure_name}"

@dataclass
class MaxDoseConstraint(ConstraintBase):
    """Ràng buộc liều tối đa cho cấu trúc."""
    dose_limit: float  # Giới hạn liều tối đa, đơn vị Gy
    constraint_type: str = "MaxDose"
    
    def _evaluate_constraint(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Đánh giá xem liều tối đa trong cấu trúc có vượt quá giới hạn không.
        
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        if len(structure_dose) == 0:
            return True, 0.0
        
        # Tính liều tối đa
        max_dose = np.max(structure_dose)
        
        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = max_dose <= self.dose_limit
        
        # Tính mức độ vi phạm
        violation = max(0, max_dose - self.dose_limit)
        
        return is_satisfied, violation
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Dmax ≤ {self.dose_limit:.1f} Gy cho {self.structure_name}"

@dataclass
class MeanDoseConstraint(ConstraintBase):
    """Ràng buộc liều trung bình cho cấu trúc."""
    dose_limit: float  # Giới hạn liều trung bình, đơn vị Gy
    constraint_type: str = "MeanDose"
    
    def _evaluate_constraint(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Đánh giá xem liều trung bình trong cấu trúc có vượt quá giới hạn không.
        
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        if len(structure_dose) == 0:
            return True, 0.0
        
        # Tính liều trung bình
        mean_dose = np.mean(structure_dose)
        
        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = mean_dose <= self.dose_limit
        
        # Tính mức độ vi phạm
        violation = max(0, mean_dose - self.dose_limit)
        
        return is_satisfied, violation
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Dmean ≤ {self.dose_limit:.1f} Gy cho {self.structure_name}"

@dataclass
class MinDoseConstraint(ConstraintBase):
    """Ràng buộc liều tối thiểu cho cấu trúc."""
    dose_limit: float  # Giới hạn liều tối thiểu, đơn vị Gy
    constraint_type: str = "MinDose"
    
    def _evaluate_constraint(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Đánh giá xem liều tối thiểu trong cấu trúc có vượt quá giới hạn không.
        
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
        
        if len(structure_dose) == 0:
            return True, 0.0
        
        # Tính liều tối thiểu
        min_dose = np.min(structure_dose)
        
        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = min_dose >= self.dose_limit
        
        # Tính mức độ vi phạm
        violation = max(0, self.dose_limit - min_dose)
        
        return is_satisfied, violation
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Dmin ≥ {self.dose_limit:.1f} Gy cho {self.structure_name}"

@dataclass
class DoseVolumeConstraint(ConstraintBase):
    """Ràng buộc liều-thể tích (DVH constraint)."""
    dose: float  # Mức liều, đơn vị Gy
    volume_percent: float  # Phần trăm thể tích
    direction: str = "upper"  # "upper" hoặc "lower"
    constraint_type: str = "DoseVolume"
    
    def __post_init__(self):
        """Xác thực các tham số sau khi khởi tạo."""
        super().__post_init__()
        if self.volume_percent < 0 or self.volume_percent > 100:
            raise ValueError(f"volume_percent phải nằm trong khoảng [0, 100], nhận được: {self.volume_percent}")
        
        if self.direction not in ["upper", "lower"]:
            raise ValueError(f"direction phải là 'upper' hoặc 'lower', nhận được: {self.direction}")
    
    def _evaluate_constraint(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Đánh giá xem ràng buộc liều-thể tích có được thỏa mãn không.
        
        Upper: không quá {volume_percent}% thể tích nhận liều >= {dose}
        Lower: ít nhất {volume_percent}% thể tích nhận liều >= {dose}
        
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
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
            100 - dvh['cumulative']  # Chuyển từ DVH tích lũy sang % thể tích
        )
        
        if self.direction == "upper":
            # Kiểm tra xem có vượt quá thể tích cho phép không
            is_satisfied = volume_at_dose <= self.volume_percent
            violation = max(0, volume_at_dose - self.volume_percent)
        else:  # "lower"
            # Kiểm tra xem có đạt đủ thể tích yêu cầu không
            is_satisfied = volume_at_dose >= self.volume_percent
            violation = max(0, self.volume_percent - volume_at_dose)
        
        return is_satisfied, violation
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        if self.direction == "upper":
            return f"V{self.dose:.1f}Gy ≤ {self.volume_percent:.1f}% cho {self.structure_name}"
        else:
            return f"V{self.dose:.1f}Gy ≥ {self.volume_percent:.1f}% cho {self.structure_name}"

@dataclass
class HomogeneityConstraint(ConstraintBase):
    """Ràng buộc về tính đồng nhất của liều trong cấu trúc."""
    prescription_dose: float  # Liều chỉ định, đơn vị Gy
    max_hi: float = 0.15  # Chỉ số đồng nhất tối đa (ICRU83)
    constraint_type: str = "Homogeneity"
    
    def _evaluate_constraint(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Đánh giá xem chỉ số đồng nhất có thỏa mãn giới hạn không.
        
        Sử dụng định nghĩa ICRU83: HI = (D2% - D98%) / D50%
        
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Tính DVH
        dvh = calculate_dvh(
            dose_array=dose_grid.dose_array,
            structure_mask=structure_mask,
            volume_type='relative'
        )
        
        # Tính các giá trị liều cần thiết
        metrics = ['D2', 'D50', 'D98']
        dvh_metrics = calculate_dvh_metrics(dvh, metrics, self.prescription_dose)
        
        # Tránh chia cho 0
        if dvh_metrics['D50'] == 0:
            return True, 0.0
        
        # Tính chỉ số đồng nhất theo định nghĩa ICRU83
        hi = (dvh_metrics['D2'] - dvh_metrics['D98']) / dvh_metrics['D50']
        
        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = hi <= self.max_hi
        
        # Tính mức độ vi phạm
        violation = max(0, hi - self.max_hi)
        
        return is_satisfied, violation
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"HI ≤ {self.max_hi:.2f} cho {self.structure_name}"

@dataclass
class ConformityConstraint(ConstraintBase):
    """Ràng buộc về tính đồng dạng của liều so với cấu trúc đích."""
    reference_dose: float  # Liều tham chiếu, thường là liều chỉ định, đơn vị Gy
    min_ci: float = 0.8  # Chỉ số đồng dạng tối thiểu (Paddick CI)
    constraint_type: str = "Conformity"
    
    def _evaluate_constraint(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Đánh giá xem chỉ số đồng dạng có thỏa mãn giới hạn không.
        
        Sử dụng định nghĩa Paddick: CI = (V_ref_target)^2 / (V_target * V_ref_total)
        
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Tính thể tích PTV
        v_target = np.sum(structure_mask)
        
        # Tạo mặt nạ cho vùng nhận liều >= reference_dose
        ref_dose_mask = dose_grid.dose_array >= self.reference_dose
        
        # Tính V_ref_total - tổng thể tích nhận liều >= reference_dose
        v_ref_total = np.sum(ref_dose_mask)
        
        # Tính V_ref_target - thể tích PTV nhận liều >= reference_dose
        v_ref_target = np.sum(ref_dose_mask & structure_mask)
        
        # Tránh chia cho 0
        if v_target == 0 or v_ref_total == 0:
            return True, 0.0
        
        # Tính chỉ số đồng dạng (Paddick CI)
        ci = (v_ref_target * v_ref_target) / (v_target * v_ref_total)
        
        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = ci >= self.min_ci
        
        # Tính mức độ vi phạm
        violation = max(0, self.min_ci - ci)
        
        return is_satisfied, violation
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"CI ≥ {self.min_ci:.2f} cho {self.structure_name}"

# Dictionary chứa tất cả các loại constraint có sẵn để dễ dàng tạo mới
CONSTRAINT_TYPES = {
    "MaxDose": MaxDoseConstraint,
    "MinDose": MinDoseConstraint,
    "MeanDose": MeanDoseConstraint,
    "DoseVolume": DoseVolumeConstraint,
    "Homogeneity": HomogeneityConstraint,
    "Conformity": ConformityConstraint
}

def create_constraint(constraint_type: str, **kwargs) -> ConstraintBase:
    """
    Tạo đối tượng constraint từ loại và tham số.
    
    Args:
        constraint_type: Loại constraint ('MaxDose', 'DoseVolume', etc.)
        **kwargs: Các tham số cần thiết cho loại constraint đó
        
    Returns:
        Đối tượng constraint đã được tạo
        
    Raises:
        ValueError: Nếu constraint_type không được hỗ trợ
    """
    if constraint_type not in CONSTRAINT_TYPES:
        raise ValueError(f"Loại constraint không hợp lệ: {constraint_type}. "
                         f"Các loại được hỗ trợ: {list(CONSTRAINT_TYPES.keys())}")
    
    return CONSTRAINT_TYPES[constraint_type](**kwargs)

class ConstraintCollection:
    """Tập hợp nhiều ràng buộc để đánh giá tổng thể kế hoạch."""
    
    def __init__(self):
        """Khởi tạo danh sách ràng buộc trống."""
        self.constraints: List[ConstraintBase] = []
    
    def add_constraint(self, constraint: ConstraintBase) -> None:
        """Thêm một ràng buộc vào danh sách."""
        self.constraints.append(constraint)
    
    def remove_constraint(self, index: int) -> None:
        """Xóa một ràng buộc từ danh sách theo chỉ số."""
        if 0 <= index < len(self.constraints):
            del self.constraints[index]
        else:
            raise IndexError(f"Chỉ số không hợp lệ: {index}")
    
    def enable_constraint(self, index: int, enabled: bool = True) -> None:
        """Bật/tắt một ràng buộc theo chỉ số."""
        if 0 <= index < len(self.constraints):
            self.constraints[index].is_enabled = enabled
        else:
            raise IndexError(f"Chỉ số không hợp lệ: {index}")
    
    def check_all(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Kiểm tra tất cả các ràng buộc với phân bố liều và cấu trúc hiện tại.
        
        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            Dictionary chứa kết quả kiểm tra của từng ràng buộc và số ràng buộc vi phạm
        """
        results = {}
        violations_count = 0
        hard_violations_count = 0
        
        for i, constraint in enumerate(self.constraints):
            if not constraint.is_enabled:
                results[f"constraint_{i}"] = {
                    "is_satisfied": True,
                    "violation": 0.0,
                    "description": constraint.get_description(),
                    "is_enabled": False,
                    "is_hard_constraint": constraint.is_hard_constraint
                }
                continue
                
            is_satisfied, violation = constraint.check(dose_grid, structures)
            
            results[f"constraint_{i}"] = {
                "is_satisfied": is_satisfied,
                "violation": violation,
                "description": constraint.get_description(),
                "is_enabled": True,
                "is_hard_constraint": constraint.is_hard_constraint
            }
            
            if not is_satisfied:
                violations_count += 1
                if constraint.is_hard_constraint:
                    hard_violations_count += 1
        
        results["summary"] = {
            "total_constraints": len(self.constraints),
            "enabled_constraints": sum(1 for c in self.constraints if c.is_enabled),
            "violations_count": violations_count,
            "hard_violations_count": hard_violations_count
        }
        
        return results
    
    def get_constraints_info(self) -> List[Dict[str, Any]]:
        """Trả về thông tin mô tả về tất cả các ràng buộc."""
        return [constraint.get_info() for constraint in self.constraints]
    
    def __len__(self) -> int:
        """Trả về số lượng ràng buộc trong danh sách."""
        return len(self.constraints)

def get_default_constraints_for_structure(structure_name: str, structure_type: str, prescription_dose: Optional[float] = None) -> List[ConstraintBase]:
    """
    Tạo các ràng buộc mặc định cho một cấu trúc dựa trên loại cấu trúc.
    
    Args:
        structure_name: Tên cấu trúc
        structure_type: Loại cấu trúc ('ptv', 'oar', 'normal')
        prescription_dose: Liều chỉ định (chỉ cần với PTV)
        
    Returns:
        Danh sách các ràng buộc mặc định
    """
    constraints = []
    
    if structure_type.lower() == 'ptv' and prescription_dose is not None:
        # Ràng buộc mặc định cho PTV
        constraints.append(MinDoseConstraint(
            structure_name=structure_name,
            dose_limit=prescription_dose * 0.95,
            priority=1,
            is_hard_constraint=True
        ))
        
        constraints.append(MaxDoseConstraint(
            structure_name=structure_name,
            dose_limit=prescription_dose * 1.07,
            priority=1,
            is_hard_constraint=True
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=prescription_dose,
            volume_percent=95,
            direction="lower",
            priority=1,
            is_hard_constraint=True
        ))
        
        constraints.append(HomogeneityConstraint(
            structure_name=structure_name,
            prescription_dose=prescription_dose,
            max_hi=0.15,
            priority=2
        ))
        
        constraints.append(ConformityConstraint(
            structure_name=structure_name,
            reference_dose=prescription_dose * 0.95,
            min_ci=0.8,
            priority=2
        ))
    
    elif structure_type.lower() == 'oar':
        # Ràng buộc mặc định cho OAR - sẽ cần điều chỉnh theo từng loại OAR cụ thể
        constraints.append(MaxDoseConstraint(
            structure_name=structure_name,
            dose_limit=45.0,  # Giá trị mặc định, cần điều chỉnh
            priority=2
        ))
        
        constraints.append(MeanDoseConstraint(
            structure_name=structure_name,
            dose_limit=30.0,  # Giá trị mặc định, cần điều chỉnh
            priority=3
        ))
    
    return constraints

def get_organ_specific_constraints(structure_name: str, prescription_dose: float) -> List[ConstraintBase]:
    """
    Tạo các ràng buộc đặc thù cho từng cơ quan dựa trên dữ liệu lâm sàng.
    
    Args:
        structure_name: Tên cấu trúc
        prescription_dose: Liều chỉ định
        
    Returns:
        Danh sách các ràng buộc đặc thù
    """
    constraints = []
    lower_case_name = structure_name.lower()
    
    # Não
    if any(term in lower_case_name for term in ['brain', 'não']):
        constraints.append(MaxDoseConstraint(
            structure_name=structure_name,
            dose_limit=60.0,
            priority=1,
            is_hard_constraint=True
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=45.0,
            volume_percent=33.0,
            direction="upper",
            priority=2
        ))
    
    # Thân não
    elif any(term in lower_case_name for term in ['brainstem', 'thân não']):
        constraints.append(MaxDoseConstraint(
            structure_name=structure_name,
            dose_limit=54.0,
            priority=1,
            is_hard_constraint=True
        ))
    
    # Dây thần kinh thị giác
    elif any(term in lower_case_name for term in ['optic', 'thị giác']):
        constraints.append(MaxDoseConstraint(
            structure_name=structure_name,
            dose_limit=55.0,
            priority=1,
            is_hard_constraint=True
        ))
    
    # Tim
    elif any(term in lower_case_name for term in ['heart', 'tim']):
        constraints.append(MeanDoseConstraint(
            structure_name=structure_name,
            dose_limit=26.0,
            priority=2
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=30.0,
            volume_percent=46.0,
            direction="upper",
            priority=2
        ))
    
    # Phổi
    elif any(term in lower_case_name for term in ['lung', 'phổi']):
        constraints.append(MeanDoseConstraint(
            structure_name=structure_name,
            dose_limit=20.0,
            priority=2
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=20.0,
            volume_percent=30.0,
            direction="upper",
            priority=2
        ))
    
    # Tủy sống
    elif any(term in lower_case_name for term in ['spinal', 'cord', 'tủy']):
        constraints.append(MaxDoseConstraint(
            structure_name=structure_name,
            dose_limit=50.0,
            priority=1,
            is_hard_constraint=True
        ))
    
    # Thực quản
    elif any(term in lower_case_name for term in ['esophag', 'thực quản']):
        constraints.append(MeanDoseConstraint(
            structure_name=structure_name,
            dose_limit=34.0,
            priority=2
        ))
        
        constraints.append(MaxDoseConstraint(
            structure_name=structure_name,
            dose_limit=105.0,
            priority=2
        ))
    
    # Gan
    elif any(term in lower_case_name for term in ['liver', 'gan']):
        constraints.append(MeanDoseConstraint(
            structure_name=structure_name,
            dose_limit=30.0,
            priority=2
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=30.0,
            volume_percent=50.0,
            direction="upper",
            priority=2
        ))
    
    # Thận
    elif any(term in lower_case_name for term in ['kidney', 'thận']):
        constraints.append(MeanDoseConstraint(
            structure_name=structure_name,
            dose_limit=18.0,
            priority=2
        ))
    
    # Bàng quang
    elif any(term in lower_case_name for term in ['bladder', 'bàng quang']):
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=65.0,
            volume_percent=50.0,
            direction="upper",
            priority=2
        ))
    
    # Trực tràng
    elif any(term in lower_case_name for term in ['rectum', 'trực tràng']):
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=75.0,
            volume_percent=15.0,
            direction="upper",
            priority=2
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=70.0,
            volume_percent=20.0,
            direction="upper",
            priority=2
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=65.0,
            volume_percent=25.0,
            direction="upper",
            priority=2
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=60.0,
            volume_percent=35.0,
            direction="upper",
            priority=2
        ))
        
        constraints.append(DoseVolumeConstraint(
            structure_name=structure_name,
            dose=50.0,
            volume_percent=50.0,
            direction="upper",
            priority=2
        ))
    
    # Nếu là PTV, thêm các ràng buộc mặc định
    elif any(term in lower_case_name for term in ['ptv', 'target']):
        constraints.extend(get_default_constraints_for_structure(
            structure_name=structure_name,
            structure_type='ptv',
            prescription_dose=prescription_dose
        ))
    
    return constraints
