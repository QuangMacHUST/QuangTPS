#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích độ vững (robustness) của kế hoạch xạ trị.

Module này cung cấp các công cụ để đánh giá độ vững của kế hoạch xạ trị 
khi có các sai số trong thiết lập và thực hiện điều trị. Các tính năng này 
giúp đánh giá khả năng chịu đựng của kế hoạch đối với các biến thiên 
không chắc chắn.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from enum import Enum, auto
import time
import concurrent.futures

from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.exceptions import RobustnessError
from quangtps.evaluation.dvh import calculate_dvh, DVHCalculator

logger = logging.getLogger(__name__)

class UncertaintyType(Enum):
    """Loại không chắc chắn trong điều trị."""
    SETUP = auto()  # Sai số thiết lập
    RANGE = auto()  # Sai số phạm vi (cho proton)
    BREATHING = auto()  # Sai số do hô hấp
    DEFORMATION = auto()  # Sai số do biến dạng
    DENSITY = auto()  # Sai số về mật độ vật liệu

@dataclass
class UncertaintyParameter:
    """Tham số cho một loại không chắc chắn."""
    type: UncertaintyType
    description: str
    magnitude: Union[float, Tuple[float, float, float]]  # Đơn giá trị hoặc vectơ (x, y, z)
    units: str
    enabled: bool = True
    
    def __post_init__(self):
        """Xác thực tham số sau khi khởi tạo."""
        if isinstance(self.magnitude, tuple) and len(self.magnitude) != 3:
            raise ValueError("Vectơ không chắc chắn phải có 3 thành phần (x, y, z)")

@dataclass
class ScenarioResult:
    """Kết quả của một kịch bản phân tích độ vững."""
    scenario_name: str
    uncertainty_parameters: Dict[str, UncertaintyParameter]
    dose_grid: Optional[DoseGrid] = None
    dvh_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)

@dataclass
class RobustnessResult:
    """Kết quả của phân tích độ vững."""
    nominal_scenario: ScenarioResult
    scenarios: List[ScenarioResult]
    target_coverage_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    oar_dose_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def get_worst_case_dvh(self, structure_name: str) -> Dict[str, np.ndarray]:
        """
        Lấy DVH xấu nhất cho một cấu trúc.
        
        Với PTV, đây là điểm có độ bao phủ thấp nhất.
        Với OAR, đây là điểm có liều cao nhất.
        
        Args:
            structure_name: Tên cấu trúc
            
        Returns:
            Dict: Dữ liệu DVH
        """
        if structure_name not in self.nominal_scenario.dvh_data:
            raise ValueError(f"Cấu trúc '{structure_name}' không tồn tại")
        
        # Kiểm tra xem đây là PTV hay OAR
        is_ptv = structure_name.lower().startswith("ptv")
        
        worst_dvh = None
        worst_score = float('-inf') if is_ptv else float('inf')
        
        # Kiểm tra tất cả kịch bản
        for scenario in self.scenarios:
            if structure_name not in scenario.dvh_data:
                continue
                
            dvh = scenario.dvh_data[structure_name]
            
            # Tính điểm dựa trên D95 cho PTV hoặc D1cc cho OAR
            if is_ptv:
                # D95 càng cao càng tốt cho PTV
                d95 = np.interp(95, dvh['volume_percent'][::-1], dvh['dose'][::-1])
                score = d95
                if score < worst_score:
                    worst_dvh = dvh
                    worst_score = score
            else:
                # D1cc càng thấp càng tốt cho OAR
                vol_1cc = min(1.0, 1.0 / dvh.get('volume_cc', 1.0) * 100)
                d1cc = np.interp(vol_1cc, dvh['volume_percent'][::-1], dvh['dose'][::-1])
                score = d1cc
                if score > worst_score:
                    worst_dvh = dvh
                    worst_score = score
        
        return worst_dvh or self.nominal_scenario.dvh_data[structure_name]
    
    def get_band_dvh(self, structure_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Lấy biên trên và dưới của DVH cho tất cả các kịch bản.
        
        Args:
            structure_name: Tên cấu trúc
            
        Returns:
            Tuple[Dict, Dict]: (Biên dưới DVH, Biên trên DVH)
        """
        if structure_name not in self.nominal_scenario.dvh_data:
            raise ValueError(f"Cấu trúc '{structure_name}' không tồn tại")
        
        # Kết hợp tất cả DVH
        all_doses = []
        all_volumes = []
        
        nominal_dvh = self.nominal_scenario.dvh_data[structure_name]
        all_doses.append(nominal_dvh['dose'])
        all_volumes.append(nominal_dvh['volume_percent'])
        
        for scenario in self.scenarios:
            if structure_name in scenario.dvh_data:
                dvh = scenario.dvh_data[structure_name]
                all_doses.append(dvh['dose'])
                all_volumes.append(dvh['volume_percent'])
        
        # Tạo lưới chung cho liều
        min_dose = min(np.min(d) for d in all_doses)
        max_dose = max(np.max(d) for d in all_doses)
        common_dose = np.linspace(min_dose, max_dose, 100)
        
        # Nội suy thể tích ở mỗi điểm liều
        interp_volumes = []
        for i, dose in enumerate(all_doses):
            volume = all_volumes[i]
            interp_vol = np.interp(common_dose, dose, volume, left=100, right=0)
            interp_volumes.append(interp_vol)
        
        # Tính biên trên và dưới
        lower_bound = np.min(interp_volumes, axis=0)
        upper_bound = np.max(interp_volumes, axis=0)
        
        # Trả về định dạng DVH
        lower_dvh = {
            'dose': common_dose,
            'volume_percent': lower_bound
        }
        
        upper_dvh = {
            'dose': common_dose,
            'volume_percent': upper_bound
        }
        
        return lower_dvh, upper_dvh


class RobustnessAnalyzer:
    """
    Phân tích độ vững của kế hoạch xạ trị.
    
    Lớp này phân tích độ vững của kế hoạch xạ trị đối với các sai số 
    và không chắc chắn trong quá trình điều trị.
    """
    
    def __init__(self):
        """Khởi tạo phân tích độ vững."""
        self.uncertainty_parameters = []
        self.nominal_dose_grid = None
        self.structures = {}
        self.dvh_calculator = None
    
    def add_setup_uncertainty(
        self, 
        magnitude: Union[float, Tuple[float, float, float]], 
        units: str = 'mm',
        description: str = 'Setup Uncertainty'
    ) -> None:
        """
        Thêm sai số thiết lập.
        
        Args:
            magnitude: Độ lớn của sai số (đơn giá trị hoặc vectơ x, y, z)
            units: Đơn vị ('mm' hoặc 'cm')
            description: Mô tả
        """
        param = UncertaintyParameter(
            type=UncertaintyType.SETUP,
            description=description,
            magnitude=magnitude,
            units=units
        )
        self.uncertainty_parameters.append(param)
    
    def add_range_uncertainty(
        self, 
        magnitude: float, 
        units: str = '%',
        description: str = 'Range Uncertainty'
    ) -> None:
        """
        Thêm sai số phạm vi (cho proton).
        
        Args:
            magnitude: Độ lớn của sai số
            units: Đơn vị ('%' hoặc 'mm')
            description: Mô tả
        """
        param = UncertaintyParameter(
            type=UncertaintyType.RANGE,
            description=description,
            magnitude=magnitude,
            units=units
        )
        self.uncertainty_parameters.append(param)
    
    def add_breathing_uncertainty(
        self, 
        magnitude: Union[float, Tuple[float, float, float]], 
        units: str = 'mm',
        description: str = 'Breathing Motion'
    ) -> None:
        """
        Thêm sai số do hô hấp.
        
        Args:
            magnitude: Độ lớn của sai số (đơn giá trị hoặc vectơ x, y, z)
            units: Đơn vị ('mm' hoặc 'cm')
            description: Mô tả
        """
        param = UncertaintyParameter(
            type=UncertaintyType.BREATHING,
            description=description,
            magnitude=magnitude,
            units=units
        )
        self.uncertainty_parameters.append(param)
    
    def add_density_uncertainty(
        self, 
        magnitude: float, 
        units: str = '%',
        description: str = 'Density Uncertainty'
    ) -> None:
        """
        Thêm sai số về mật độ vật liệu.
        
        Args:
            magnitude: Độ lớn của sai số
            units: Đơn vị ('%')
            description: Mô tả
        """
        param = UncertaintyParameter(
            type=UncertaintyType.DENSITY,
            description=description,
            magnitude=magnitude,
            units=units
        )
        self.uncertainty_parameters.append(param)
    
    def set_nominal_state(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray]
    ) -> None:
        """
        Thiết lập trạng thái danh nghĩa.
        
        Args:
            dose_grid: Lưới liều danh nghĩa
            structures: Dictionary các cấu trúc (mask)
        """
        self.nominal_dose_grid = dose_grid.copy()
        self.structures = structures.copy()
        self.dvh_calculator = DVHCalculator(structures)
    
    def generate_scenarios(self) -> List[Dict[str, UncertaintyParameter]]:
        """
        Tạo các kịch bản phân tích.
        
        Returns:
            List[Dict]: Danh sách các kịch bản
        """
        scenarios = []
        
        # Tạo kịch bản danh nghĩa
        scenarios.append({})
        
        # Tạo các kịch bản theo sai số thiết lập
        setup_params = [p for p in self.uncertainty_parameters if p.type == UncertaintyType.SETUP]
        
        for param in setup_params:
            if isinstance(param.magnitude, tuple):
                # Tạo 6 kịch bản: +/- x, +/- y, +/- z
                for axis, value in zip(['x', 'y', 'z'], param.magnitude):
                    for sign in [1, -1]:
                        shift = [0, 0, 0]
                        if axis == 'x':
                            shift[0] = sign * value
                        elif axis == 'y':
                            shift[1] = sign * value
                        else:  # z
                            shift[2] = sign * value
                        
                        scenario = {
                            f"{param.description} {'+' if sign > 0 else '-'}{axis}": UncertaintyParameter(
                                type=UncertaintyType.SETUP,
                                description=f"{param.description} {'+' if sign > 0 else '-'}{axis}",
                                magnitude=tuple(shift),
                                units=param.units
                            )
                        }
                        scenarios.append(scenario)
            else:
                # Isotropic: tạo 6 kịch bản theo các hướng chính
                for axis in ['x', 'y', 'z']:
                    for sign in [1, -1]:
                        shift = [0, 0, 0]
                        if axis == 'x':
                            shift[0] = sign * param.magnitude
                        elif axis == 'y':
                            shift[1] = sign * param.magnitude
                        else:  # z
                            shift[2] = sign * param.magnitude
                        
                        scenario = {
                            f"{param.description} {'+' if sign > 0 else '-'}{axis}": UncertaintyParameter(
                                type=UncertaintyType.SETUP,
                                description=f"{param.description} {'+' if sign > 0 else '-'}{axis}",
                                magnitude=tuple(shift),
                                units=param.units
                            )
                        }
                        scenarios.append(scenario)
        
        # Tạo các kịch bản cho sai số phạm vi (chỉ cho proton)
        range_params = [p for p in self.uncertainty_parameters if p.type == UncertaintyType.RANGE]
        
        for param in range_params:
            for sign in [1, -1]:
                scenario = {
                    f"{param.description} {'+' if sign > 0 else '-'}": UncertaintyParameter(
                        type=UncertaintyType.RANGE,
                        description=f"{param.description} {'+' if sign > 0 else '-'}",
                        magnitude=sign * param.magnitude,
                        units=param.units
                    )
                }
                scenarios.append(scenario)
        
        # Tạo các kịch bản cho sai số về mật độ
        density_params = [p for p in self.uncertainty_parameters if p.type == UncertaintyType.DENSITY]
        
        for param in density_params:
            for sign in [1, -1]:
                scenario = {
                    f"{param.description} {'+' if sign > 0 else '-'}": UncertaintyParameter(
                        type=UncertaintyType.DENSITY,
                        description=f"{param.description} {'+' if sign > 0 else '-'}",
                        magnitude=sign * param.magnitude,
                        units=param.units
                    )
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _simulate_scenario(
        self, 
        scenario: Dict[str, UncertaintyParameter]
    ) -> Tuple[Dict[str, UncertaintyParameter], DoseGrid, Dict[str, Any]]:
        """
        Mô phỏng một kịch bản phân tích.
        
        Args:
            scenario: Dictionary các tham số không chắc chắn
            
        Returns:
            Tuple: (Tham số, Lưới liều mô phỏng, Dữ liệu DVH)
        """
        # Bắt đầu với lưới liều danh nghĩa
        simulated_dose = self.nominal_dose_grid.copy()
        
        # Áp dụng các sai số
        for param_name, param in scenario.items():
            if param.type == UncertaintyType.SETUP:
                # Dịch chuyển phân bố liều
                shift = param.magnitude
                if isinstance(shift, tuple):
                    simulated_dose = self._shift_dose_grid(simulated_dose, shift)
                else:
                    # Nếu là đơn giá trị, áp dụng cùng giá trị cho cả 3 chiều
                    simulated_dose = self._shift_dose_grid(simulated_dose, (shift, shift, shift))
            
            elif param.type == UncertaintyType.RANGE:
                # Mô phỏng sai số phạm vi bằng cách scale phân bố liều
                # Đây là mô phỏng đơn giản, thực tế cần tính toán phức tạp hơn
                scale = 1.0 + param.magnitude / 100.0 if param.units == '%' else 1.0 + param.magnitude / 10.0
                simulated_dose.dose_array = simulated_dose.dose_array * scale
            
            elif param.type == UncertaintyType.DENSITY:
                # Mô phỏng sai số mật độ bằng cách scale phân bố liều
                scale = 1.0 + param.magnitude / 100.0 if param.units == '%' else 1.0 + param.magnitude / 10.0
                simulated_dose.dose_array = simulated_dose.dose_array * scale
        
        # Tính DVH cho kịch bản này
        dvh_data = {}
        for struct_name, struct_mask in self.structures.items():
            dvh = calculate_dvh(simulated_dose.dose_array, struct_mask)
            dvh_data[struct_name] = dvh
        
        return scenario, simulated_dose, dvh_data
    
    def _shift_dose_grid(self, dose_grid: DoseGrid, shift: Tuple[float, float, float]) -> DoseGrid:
        """
        Dịch chuyển lưới liều.
        
        Args:
            dose_grid: Lưới liều gốc
            shift: Vectơ dịch chuyển (x, y, z) theo mm
            
        Returns:
            DoseGrid: Lưới liều đã dịch chuyển
        """
        # Chuyển đổi shift từ mm sang voxel
        voxel_size = dose_grid.voxel_size  # (x, y, z) in mm
        shift_voxels = (
            shift[0] / voxel_size[0],
            shift[1] / voxel_size[1],
            shift[2] / voxel_size[2]
        )
        
        # Làm tròn về số nguyên
        shift_voxels_int = (int(round(shift_voxels[0])), 
                           int(round(shift_voxels[1])), 
                           int(round(shift_voxels[2])))
        
        # Tạo lưới liều mới
        shifted_dose = dose_grid.copy()
        
        # Thực hiện dịch chuyển
        array = dose_grid.dose_array
        padded = np.pad(array, ((abs(shift_voxels_int[0]), abs(shift_voxels_int[0])), 
                                (abs(shift_voxels_int[1]), abs(shift_voxels_int[1])), 
                                (abs(shift_voxels_int[2]), abs(shift_voxels_int[2]))), 
                        mode='constant')
        
        # Lấy phần tử với offset
        x_start = abs(shift_voxels_int[0]) - shift_voxels_int[0]
        y_start = abs(shift_voxels_int[1]) - shift_voxels_int[1]
        z_start = abs(shift_voxels_int[2]) - shift_voxels_int[2]
        
        x_end = x_start + array.shape[0]
        y_end = y_start + array.shape[1]
        z_end = z_start + array.shape[2]
        
        shifted_array = padded[x_start:x_end, y_start:y_end, z_start:z_end]
        
        shifted_dose.dose_array = shifted_array
        
        return shifted_dose
    
    def analyze(self, max_workers: int = 4) -> RobustnessResult:
        """
        Thực hiện phân tích độ vững.
        
        Args:
            max_workers: Số luồng tối đa cho xử lý song song
            
        Returns:
            RobustnessResult: Kết quả phân tích độ vững
        """
        if not self.nominal_dose_grid:
            raise RobustnessError("Chưa thiết lập trạng thái danh nghĩa")
        
        # Tạo các kịch bản
        scenarios = self.generate_scenarios()
        logger.info(f"Đã tạo {len(scenarios)} kịch bản phân tích độ vững")
        
        # Tính DVH danh nghĩa
        nominal_dvh_data = {}
        for struct_name, struct_mask in self.structures.items():
            dvh = calculate_dvh(self.nominal_dose_grid.dose_array, struct_mask)
            nominal_dvh_data[struct_name] = dvh
        
        # Tạo kết quả danh nghĩa
        nominal_result = ScenarioResult(
            scenario_name="Nominal",
            uncertainty_parameters={},
            dose_grid=self.nominal_dose_grid,
            dvh_data=nominal_dvh_data
        )
        
        # Mô phỏng các kịch bản
        scenario_results = []
        
        # Thêm kịch bản danh nghĩa
        scenario_results.append(nominal_result)
        
        # Xử lý song song các kịch bản khác
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Bỏ qua kịch bản đầu tiên (nominal)
            futures = [executor.submit(self._simulate_scenario, scenario) 
                      for scenario in scenarios[1:]]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    params, dose, dvh = future.result()
                    
                    # Tạo tên kịch bản từ tham số
                    scenario_name = " & ".join(params.keys())
                    
                    # Tạo kết quả kịch bản
                    result = ScenarioResult(
                        scenario_name=scenario_name,
                        uncertainty_parameters=params,
                        dose_grid=dose,
                        dvh_data=dvh
                    )
                    
                    scenario_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi mô phỏng kịch bản: {str(e)}")
        
        # Tính phạm vi độ bao phủ cho các PTV
        target_coverage_range = {}
        for struct_name in self.structures:
            if struct_name.lower().startswith("ptv"):
                min_d95 = float('inf')
                max_d95 = float('-inf')
                
                for result in scenario_results:
                    if struct_name in result.dvh_data:
                        dvh = result.dvh_data[struct_name]
                        d95 = np.interp(95, dvh['volume_percent'][::-1], dvh['dose'][::-1])
                        
                        min_d95 = min(min_d95, d95)
                        max_d95 = max(max_d95, d95)
                
                target_coverage_range[struct_name] = (min_d95, max_d95)
        
        # Tính phạm vi liều cho các OAR
        oar_dose_range = {}
        for struct_name in self.structures:
            if not struct_name.lower().startswith("ptv"):
                min_dmean = float('inf')
                max_dmean = float('-inf')
                
                for result in scenario_results:
                    if struct_name in result.dvh_data:
                        dvh = result.dvh_data[struct_name]
                        dmean = np.mean(dvh['dose'])
                        
                        min_dmean = min(min_dmean, dmean)
                        max_dmean = max(max_dmean, dmean)
                
                oar_dose_range[struct_name] = (min_dmean, max_dmean)
        
        # Tạo kết quả tổng hợp
        robustness_result = RobustnessResult(
            nominal_scenario=nominal_result,
            scenarios=scenario_results,
            target_coverage_range=target_coverage_range,
            oar_dose_range=oar_dose_range
        )
        
        return robustness_result


def plot_robustness_dvh(
    result: RobustnessResult,
    structure_names: List[str],
    show_band: bool = True,
    show_worst_case: bool = True,
    title: str = "Robustness Analysis DVH",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Vẽ biểu đồ DVH cho phân tích độ vững.
    
    Args:
        result: Kết quả phân tích độ vững
        structure_names: Danh sách tên cấu trúc cần vẽ
        show_band: Có hiển thị dải DVH không
        show_worst_case: Có hiển thị trường hợp xấu nhất không
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu biểu đồ
        
    Returns:
        Figure: Đối tượng biểu đồ matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10.colors
    color_idx = 0
    
    for struct_name in structure_names:
        if struct_name not in result.nominal_scenario.dvh_data:
            logger.warning(f"Cấu trúc '{struct_name}' không tồn tại trong dữ liệu")
            continue
        
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        # Vẽ DVH danh nghĩa
        nominal_dvh = result.nominal_scenario.dvh_data[struct_name]
        ax.plot(nominal_dvh['dose'], nominal_dvh['volume_percent'], 
                color=color, linestyle='-', linewidth=2, label=f"{struct_name} (Nominal)")
        
        if show_band:
            # Vẽ dải DVH
            lower_dvh, upper_dvh = result.get_band_dvh(struct_name)
            ax.fill_between(lower_dvh['dose'], lower_dvh['volume_percent'], upper_dvh['volume_percent'],
                           color=color, alpha=0.2)
        
        if show_worst_case:
            # Vẽ trường hợp xấu nhất
            worst_dvh = result.get_worst_case_dvh(struct_name)
            ax.plot(worst_dvh['dose'], worst_dvh['volume_percent'], 
                    color=color, linestyle='--', linewidth=1.5, label=f"{struct_name} (Worst Case)")
    
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    # Đặt giới hạn trục
    ax.set_xlim(0, None)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 