"""
Module định nghĩa các phương pháp tối ưu hóa dựa trên mô hình sinh học.

Module này cung cấp các thuật toán và chiến lược tối ưu hóa kế hoạch xạ trị
sử dụng các mô hình sinh học như TCP (Tumor Control Probability) và NTCP 
(Normal Tissue Complication Probability) để đánh giá và tối ưu hóa kế hoạch.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field

from quangtps.dose.dose_grid import DoseGrid
from quangtps.optimization.objectives import ObjectiveCollection, ObjectiveBase
from quangtps.optimization.constraints import ConstraintCollection
from quangtps.optimization.optimization_engine import OptimizationParameters, OptimizationEngine, OptimizationResults
from quangtps.optimization.methods.objective_based import ObjectiveBasedMethod

logger = logging.getLogger(__name__)

class BiologicalMethod(ObjectiveBasedMethod):
    """
    Lớp cơ sở cho các phương pháp tối ưu hóa dựa trên mô hình sinh học.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None
    ):
        """
        Khởi tạo phương pháp tối ưu hóa sinh học.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc (nếu có)
            parameters: Các tham số tối ưu hóa
        """
        super().__init__(objectives, constraints, parameters)

class TCPNTCPMethod(BiologicalMethod):
    """
    Phương pháp tối ưu hóa dựa trên mô hình TCP/NTCP.
    
    Phương pháp này tối ưu hóa kế hoạch xạ trị bằng cách tối đa hóa xác suất kiểm soát khối u (TCP)
    trong khi giảm thiểu xác suất biến chứng mô lành (NTCP).
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None,
        tcp_weight: float = 1.0,
        ntcp_weight: float = 1.0,
        tcp_parameters: Optional[Dict[str, Dict[str, float]]] = None,
        ntcp_parameters: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Khởi tạo phương pháp tối ưu hóa TCP/NTCP.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc (nếu có)
            parameters: Các tham số tối ưu hóa
            tcp_weight: Trọng số cho TCP trong hàm mục tiêu
            ntcp_weight: Trọng số cho NTCP trong hàm mục tiêu
            tcp_parameters: Tham số TCP cho từng cấu trúc PTV
            ntcp_parameters: Tham số NTCP cho từng cấu trúc OAR
        """
        super().__init__(objectives, constraints, parameters)
        self.tcp_weight = tcp_weight
        self.ntcp_weight = ntcp_weight
        self.tcp_parameters = tcp_parameters or {}
        self.ntcp_parameters = ntcp_parameters or {}
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa dựa trên TCP/NTCP.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Tuple[DoseGrid, OptimizationResults]: Phân bố liều tối ưu và kết quả tối ưu hóa
        """
        # Lưu trữ các mục tiêu ban đầu
        original_objectives = self.objectives
        
        try:
            # Tạo các mục tiêu TCP/NTCP
            biological_objectives = ObjectiveCollection()
            
            # Thêm mục tiêu TCP cho các cấu trúc PTV
            for structure_name, params in self.tcp_parameters.items():
                if structure_name in structures:
                    tcp_objective = TCPObjective(
                        structure_name=structure_name,
                        weight=self.tcp_weight,
                        **params
                    )
                    biological_objectives.add_objective(tcp_objective)
                else:
                    logger.warning(f"Cấu trúc '{structure_name}' không tồn tại trong structures, bỏ qua mục tiêu TCP")
            
            # Thêm mục tiêu NTCP cho các cấu trúc OAR
            for structure_name, params in self.ntcp_parameters.items():
                if structure_name in structures:
                    ntcp_objective = NTCPObjective(
                        structure_name=structure_name,
                        weight=self.ntcp_weight,
                        **params
                    )
                    biological_objectives.add_objective(ntcp_objective)
                else:
                    logger.warning(f"Cấu trúc '{structure_name}' không tồn tại trong structures, bỏ qua mục tiêu NTCP")
            
            # Thêm các mục tiêu gốc nếu cần
            for obj in original_objectives:
                biological_objectives.add_objective(obj)
            
            # Thay thế tạm thời các mục tiêu
            self.objectives = biological_objectives
            
            # Thực hiện tối ưu hóa
            final_dose_grid, results = super().optimize(dose_grid, structures, solver_name)
            
            # Tính toán và lưu trữ các giá trị TCP/NTCP cuối cùng
            tcp_values = {}
            ntcp_values = {}
            
            for structure_name, params in self.tcp_parameters.items():
                if structure_name in structures:
                    tcp_values[structure_name] = calculate_tcp(
                        dose_grid=final_dose_grid,
                        structure_mask=structures[structure_name],
                        **params
                    )
            
            for structure_name, params in self.ntcp_parameters.items():
                if structure_name in structures:
                    ntcp_values[structure_name] = calculate_ntcp(
                        dose_grid=final_dose_grid,
                        structure_mask=structures[structure_name],
                        **params
                    )
            
            # Lưu trữ kết quả
            results.tcp_values = tcp_values
            results.ntcp_values = ntcp_values
            
            return final_dose_grid, results
            
        finally:
            # Khôi phục các mục tiêu ban đầu
            self.objectives = original_objectives
    
    def set_tcp_parameters(self, structure_name: str, **params) -> None:
        """
        Thiết lập tham số TCP cho một cấu trúc.
        
        Args:
            structure_name: Tên của cấu trúc
            **params: Các tham số TCP (alpha, beta, rho, ...)
        """
        self.tcp_parameters[structure_name] = params
    
    def set_ntcp_parameters(self, structure_name: str, **params) -> None:
        """
        Thiết lập tham số NTCP cho một cấu trúc.
        
        Args:
            structure_name: Tên của cấu trúc
            **params: Các tham số NTCP (n, m, TD50, ...)
        """
        self.ntcp_parameters[structure_name] = params

class EUDMethod(BiologicalMethod):
    """
    Phương pháp tối ưu hóa dựa trên liều tương đương đồng nhất (EUD).
    
    Phương pháp này tối ưu hóa kế hoạch xạ trị bằng cách tối ưu hóa EUD cho các cấu trúc,
    thay vì tối ưu hóa trực tiếp phân bố liều.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None,
        eud_parameters: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Khởi tạo phương pháp tối ưu hóa EUD.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc (nếu có)
            parameters: Các tham số tối ưu hóa
            eud_parameters: Tham số EUD cho từng cấu trúc
        """
        super().__init__(objectives, constraints, parameters)
        self.eud_parameters = eud_parameters or {}
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa dựa trên EUD.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Tuple[DoseGrid, OptimizationResults]: Phân bố liều tối ưu và kết quả tối ưu hóa
        """
        # Lưu trữ các mục tiêu ban đầu
        original_objectives = self.objectives
        
        try:
            # Tạo các mục tiêu EUD
            eud_objectives = ObjectiveCollection()
            
            # Thêm mục tiêu EUD cho các cấu trúc
            for structure_name, params in self.eud_parameters.items():
                if structure_name in structures:
                    # Tạo mục tiêu EUD với tham số a phù hợp
                    from quangtps.optimization.objectives import EUDObjective
                    
                    # Xác định hướng dựa trên loại cấu trúc (PTV hoặc OAR)
                    direction = params.get("direction", "lower" if params.get("a", 1) > 0 else "upper")
                    
                    eud_obj = EUDObjective(
                        structure_name=structure_name,
                        target_eud=params.get("target_eud", 0.0),
                        parameter_a=params.get("a", 1.0),
                        direction=direction,
                        weight=params.get("weight", 1.0)
                    )
                    eud_objectives.add_objective(eud_obj)
                else:
                    logger.warning(f"Cấu trúc '{structure_name}' không tồn tại trong structures, bỏ qua mục tiêu EUD")
            
            # Thêm các mục tiêu gốc nếu cần
            for obj in original_objectives:
                if not isinstance(obj, EUDObjective):  # Tránh trùng lặp
                    eud_objectives.add_objective(obj)
            
            # Thay thế tạm thời các mục tiêu
            self.objectives = eud_objectives
            
            # Thực hiện tối ưu hóa
            final_dose_grid, results = super().optimize(dose_grid, structures, solver_name)
            
            # Tính toán và lưu trữ các giá trị EUD cuối cùng
            eud_values = {}
            
            for structure_name, params in self.eud_parameters.items():
                if structure_name in structures:
                    eud_values[structure_name] = calculate_eud(
                        dose_grid=final_dose_grid,
                        structure_mask=structures[structure_name],
                        a=params.get("a", 1.0)
                    )
            
            # Lưu trữ kết quả
            results.eud_values = eud_values
            
            return final_dose_grid, results
            
        finally:
            # Khôi phục các mục tiêu ban đầu
            self.objectives = original_objectives
    
    def set_eud_parameters(self, structure_name: str, **params) -> None:
        """
        Thiết lập tham số EUD cho một cấu trúc.
        
        Args:
            structure_name: Tên của cấu trúc
            **params: Các tham số EUD (a, target_eud, weight, direction)
        """
        self.eud_parameters[structure_name] = params

class TCPObjective(ObjectiveBase):
    """Hàm mục tiêu dựa trên xác suất kiểm soát khối u (TCP)."""
    def __init__(
        self,
        structure_name: str,
        alpha: float = 0.3,  # Gy^-1
        beta: float = 0.03,  # Gy^-2
        rho: float = 1e7,    # Mật độ tế bào khối u (cells/cm^3)
        weight: float = 1.0,
        maximize: bool = True
    ):
        """
        Khởi tạo hàm mục tiêu TCP.
        
        Args:
            structure_name: Tên của cấu trúc (PTV)
            alpha: Tham số alpha trong mô hình tuyến tính-bậc hai (LQ)
            beta: Tham số beta trong mô hình LQ
            rho: Mật độ tế bào khối u
            weight: Trọng số cho hàm mục tiêu
            maximize: True để tối đa hóa TCP, False để tối thiểu hóa
        """
        super().__init__(structure_name=structure_name, weight=weight, objective_type="TCP")
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.maximize = maximize
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính toán giá trị cost dựa trên TCP.
        
        Args:
            dose_grid: Phân bố liều hiện tại
            structure_mask: Mặt nạ cấu trúc
            
        Returns:
            Giá trị cost
        """
        # Tính TCP
        tcp = calculate_tcp(
            dose_grid=dose_grid,
            structure_mask=structure_mask,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho
        )
        
        # Tính cost (tối đa hóa TCP tương đương với tối thiểu hóa -TCP)
        if self.maximize:
            return -tcp * self.weight
        else:
            return tcp * self.weight

class NTCPObjective(ObjectiveBase):
    """Hàm mục tiêu dựa trên xác suất biến chứng mô lành (NTCP)."""
    def __init__(
        self,
        structure_name: str,
        n: float = 0.1,      # Tham số thể tích
        m: float = 0.1,      # Độ dốc của đường cong liều-đáp ứng
        TD50: float = 50.0,  # Liều gây ra 50% biến chứng
        weight: float = 1.0,
        minimize: bool = True
    ):
        """
        Khởi tạo hàm mục tiêu NTCP.
        
        Args:
            structure_name: Tên của cấu trúc (OAR)
            n: Tham số thể tích
            m: Độ dốc của đường cong liều-đáp ứng
            TD50: Liều gây ra 50% biến chứng
            weight: Trọng số cho hàm mục tiêu
            minimize: True để tối thiểu hóa NTCP, False để tối đa hóa
        """
        super().__init__(structure_name=structure_name, weight=weight, objective_type="NTCP")
        self.n = n
        self.m = m
        self.TD50 = TD50
        self.minimize = minimize
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính toán giá trị cost dựa trên NTCP.
        
        Args:
            dose_grid: Phân bố liều hiện tại
            structure_mask: Mặt nạ cấu trúc
            
        Returns:
            Giá trị cost
        """
        # Tính NTCP
        ntcp = calculate_ntcp(
            dose_grid=dose_grid,
            structure_mask=structure_mask,
            n=self.n,
            m=self.m,
            TD50=self.TD50
        )
        
        # Tính cost
        if self.minimize:
            return ntcp * self.weight
        else:
            return -ntcp * self.weight

def calculate_tcp(
    dose_grid: DoseGrid,
    structure_mask: np.ndarray,
    alpha: float = 0.3,  # Gy^-1
    beta: float = 0.03,  # Gy^-2
    rho: float = 1e7,    # cells/cm^3
    num_fractions: int = 1
) -> float:
    """
    Tính xác suất kiểm soát khối u (TCP) sử dụng mô hình Poisson.
    
    Args:
        dose_grid: Phân bố liều
        structure_mask: Mặt nạ cấu trúc
        alpha: Tham số alpha trong mô hình tuyến tính-bậc hai (LQ)
        beta: Tham số beta trong mô hình LQ
        rho: Mật độ tế bào khối u
        num_fractions: Số phân đoạn
        
    Returns:
        Giá trị TCP
    """
    # Lấy phân bố liều trên cấu trúc
    structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
    
    if len(structure_dose) == 0:
        return 0.0
    
    # Tính thể tích voxel (cm^3)
    voxel_size = dose_grid.voxel_size  # mm
    voxel_volume = np.prod(voxel_size) / 1000.0  # cm^3
    
    # Tính số tế bào trong mỗi voxel
    num_cells = rho * voxel_volume
    
    # Tính hiệu quả sinh học (BED) cho mỗi voxel
    if num_fractions > 1:
        # Liều mỗi phân đoạn
        dose_per_fraction = structure_dose / num_fractions
        # BED = n*d*(1 + d/(alpha/beta))
        bed = num_fractions * dose_per_fraction * (1 + dose_per_fraction / (alpha / beta))
    else:
        bed = structure_dose
    
    # Tính xác suất sống sót của tế bào trong mỗi voxel
    survival = np.exp(-alpha * bed)
    
    # Tính số tế bào sống sót trong mỗi voxel
    surviving_cells = num_cells * survival
    
    # Tính tổng số tế bào sống sót
    total_surviving_cells = np.sum(surviving_cells)
    
    # Tính TCP theo mô hình Poisson
    tcp = np.exp(-total_surviving_cells)
    
    return tcp

def calculate_ntcp(
    dose_grid: DoseGrid,
    structure_mask: np.ndarray,
    n: float = 0.1,      # Tham số thể tích
    m: float = 0.1,      # Độ dốc của đường cong liều-đáp ứng
    TD50: float = 50.0,  # Liều gây ra 50% biến chứng
    use_lkb: bool = True  # Sử dụng mô hình LKB (True) hoặc mô hình tích phân (False)
) -> float:
    """
    Tính xác suất biến chứng mô lành (NTCP) sử dụng mô hình LKB hoặc mô hình tích phân.
    
    Args:
        dose_grid: Phân bố liều
        structure_mask: Mặt nạ cấu trúc
        n: Tham số thể tích
        m: Độ dốc của đường cong liều-đáp ứng
        TD50: Liều gây ra 50% biến chứng
        use_lkb: Sử dụng mô hình LKB (True) hoặc mô hình tích phân (False)
        
    Returns:
        Giá trị NTCP
    """
    # Lấy phân bố liều trên cấu trúc
    structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
    
    if len(structure_dose) == 0:
        return 0.0
    
    if use_lkb:
        # Mô hình LKB (Lyman-Kutcher-Burman)
        # Tính gEUD
        gEUD = calculate_eud(dose_grid, structure_mask, a=(1/n))
        
        # Tính tham số t
        t = (gEUD - TD50) / (m * TD50)
        
        # Tính NTCP
        from scipy.special import erf
        ntcp = 0.5 * (1 + erf(t / np.sqrt(2)))
    else:
        # Mô hình tích phân
        # Tính xác suất biến chứng cho mỗi voxel
        p_i = 1 / (1 + (TD50 / structure_dose)**(4 * m))
        
        # Tính NTCP
        ntcp = 1 - np.prod(1 - p_i)
    
    return ntcp

def calculate_eud(
    dose_grid: DoseGrid,
    structure_mask: np.ndarray,
    a: float = 1.0
) -> float:
    """
    Tính liều tương đương đồng nhất (EUD).
    
    Args:
        dose_grid: Phân bố liều
        structure_mask: Mặt nạ cấu trúc
        a: Tham số a (a > 0 cho PTV, a < 0 cho OAR)
        
    Returns:
        Giá trị EUD
    """
    # Lấy phân bố liều trên cấu trúc
    structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)
    
    if len(structure_dose) == 0:
        return 0.0
    
    # Tính EUD
    # EUD = (sum(D_i^a) / N)^(1/a)
    N = len(structure_dose)
    
    if abs(a) < 1e-6:  # a gần bằng 0, sử dụng trung bình nhân
        # lim(a->0) EUD = geometric mean
        log_doses = np.log(structure_dose + 1e-10)  # Tránh log(0)
        eud = np.exp(np.mean(log_doses))
    else:
        eud = np.power(np.mean(np.power(structure_dose, a)), 1/a)
    
    return eud

def create_biological_method(
    method_type: str,
    objectives: ObjectiveCollection,
    constraints: Optional[ConstraintCollection] = None,
    parameters: Optional[OptimizationParameters] = None,
    **kwargs
) -> BiologicalMethod:
    """
    Tạo đối tượng phương pháp tối ưu hóa sinh học dựa trên loại phương pháp.
    
    Args:
        method_type: Loại phương pháp tối ưu hóa ("tcp_ntcp", "eud")
        objectives: Collection chứa các hàm mục tiêu
        constraints: Collection chứa các ràng buộc (nếu có)
        parameters: Các tham số tối ưu hóa
        **kwargs: Các tham số bổ sung cho từng loại phương pháp
        
    Returns:
        Đối tượng phương pháp tối ưu hóa sinh học
    """
    if method_type == "tcp_ntcp":
        tcp_weight = kwargs.get("tcp_weight", 1.0)
        ntcp_weight = kwargs.get("ntcp_weight", 1.0)
        tcp_parameters = kwargs.get("tcp_parameters", {})
        ntcp_parameters = kwargs.get("ntcp_parameters", {})
        
        return TCPNTCPMethod(
            objectives=objectives,
            constraints=constraints,
            parameters=parameters,
            tcp_weight=tcp_weight,
            ntcp_weight=ntcp_weight,
            tcp_parameters=tcp_parameters,
            ntcp_parameters=ntcp_parameters
        )
    elif method_type == "eud":
        eud_parameters = kwargs.get("eud_parameters", {})
        
        return EUDMethod(
            objectives=objectives,
            constraints=constraints,
            parameters=parameters,
            eud_parameters=eud_parameters
        )
    else:
        raise ValueError(f"Không hỗ trợ loại phương pháp: {method_type}")
