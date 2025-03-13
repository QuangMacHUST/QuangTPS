#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đánh giá kế hoạch xạ trị trong QuangTPS.

Module này cung cấp các lớp và phương thức để đánh giá chất lượng kế hoạch xạ trị,
bao gồm phân tích DVH và tính toán các chỉ số đánh giá kế hoạch.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import matplotlib.pyplot as plt
from enum import Enum

logger = logging.getLogger(__name__)


class DVHType(Enum):
    """Enum cho các loại biểu đồ DVH."""
    CUMULATIVE = "Cumulative"
    DIFFERENTIAL = "Differential"


class PlanEvaluation:
    """
    Lớp đánh giá kế hoạch xạ trị.
    
    Lớp này cung cấp các phương thức để đánh giá toàn diện một kế hoạch xạ trị,
    bao gồm phân tích DVH, tính toán các chỉ số đánh giá và so sánh các kế hoạch.
    """
    
    def __init__(self, plan_id: str, dose_grid: Optional[np.ndarray] = None):
        """
        Khởi tạo đối tượng đánh giá kế hoạch.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch cần đánh giá
        dose_grid : np.ndarray, optional
            Lưới liều 3D của kế hoạch
        """
        self.plan_id = plan_id
        self.dose_grid = dose_grid
        self.structure_masks = {}  # Dict[str, np.ndarray]
        self.dvh_data = {}  # Dict[str, DVHAnalysis]
        self.metrics = {}  # Dict[str, PlanQualityMetrics]
        
    def set_dose_grid(self, dose_grid: np.ndarray):
        """
        Đặt lưới liều cho đánh giá.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            Lưới liều 3D của kế hoạch
        """
        self.dose_grid = dose_grid
        
    def add_structure(self, structure_id: str, structure_name: str, structure_mask: np.ndarray, structure_type: str = ""):
        """
        Thêm cấu trúc để đánh giá.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        structure_name : str
            Tên hiển thị của cấu trúc
        structure_mask : np.ndarray
            Mặt nạ nhị phân 3D của cấu trúc
        structure_type : str, optional
            Loại cấu trúc (TARGET, OAR, ...)
        """
        self.structure_masks[structure_id] = {
            'id': structure_id,
            'name': structure_name,
            'mask': structure_mask,
            'type': structure_type
        }
        
    def calculate_dvh(self, structure_ids: Optional[List[str]] = None, bins: int = 100, 
                      max_dose: Optional[float] = None) -> Dict[str, Any]:
        """
        Tính toán DVH cho các cấu trúc.
        
        Parameters
        ----------
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần tính DVH, None = tất cả
        bins : int
            Số lượng bin trong histogram
        max_dose : float, optional
            Liều tối đa để chuẩn hóa, None = tự động
            
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa dữ liệu DVH cho mỗi cấu trúc
        """
        if self.dose_grid is None:
            logger.error("Không thể tính DVH - chưa có lưới liều")
            return {}
            
        # Xác định danh sách cấu trúc cần tính DVH
        structures_to_process = structure_ids if structure_ids else list(self.structure_masks.keys())
        
        # Xác định liều tối đa nếu không được cung cấp
        if max_dose is None:
            max_dose = np.max(self.dose_grid) * 1.1  # Thêm 10% margint
            
        # Tính toán DVH cho mỗi cấu trúc
        for struct_id in structures_to_process:
            if struct_id not in self.structure_masks:
                logger.warning(f"Bỏ qua cấu trúc không tồn tại: {struct_id}")
                continue
                
            struct_data = self.structure_masks[struct_id]
            struct_mask = struct_data['mask']
            
            # Trích xuất liều trong cấu trúc
            structure_dose = self.dose_grid[struct_mask > 0]
            
            if len(structure_dose) == 0:
                logger.warning(f"Cấu trúc {struct_id} không có voxel")
                continue
                
            # Tính histogram
            hist, bin_edges = np.histogram(structure_dose, bins=bins, range=(0, max_dose))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Tính DVH tích lũy
            cumulative_dvh = np.cumsum(hist[::-1])[::-1]
            cumulative_dvh = cumulative_dvh / cumulative_dvh[0] if cumulative_dvh[0] > 0 else cumulative_dvh
            
            # Tính DVH vi phân
            differential_dvh = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            
            # Lưu kết quả
            self.dvh_data[struct_id] = {
                'structure_id': struct_id,
                'structure_name': struct_data['name'],
                'structure_type': struct_data['type'],
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'differential': differential_dvh,
                'cumulative': cumulative_dvh,
                'volume_cc': np.sum(struct_mask) * (self.voxel_size()[0] * self.voxel_size()[1] * self.voxel_size()[2]) / 1000.0,
                'min_dose': np.min(structure_dose),
                'max_dose': np.max(structure_dose),
                'mean_dose': np.mean(structure_dose),
                'median_dose': np.median(structure_dose)
            }
            
        return self.dvh_data
    
    def voxel_size(self) -> Tuple[float, float, float]:
        """
        Lấy kích thước voxel.
        
        Returns
        -------
        Tuple[float, float, float]
            Kích thước voxel (mm) theo (x, y, z)
        """
        # Giả định kích thước voxel, trong thực tế lấy từ thông tin DICOM
        return (3.0, 3.0, 3.0)
        
    def plot_dvh(self, structure_ids: Optional[List[str]] = None, dvh_type: DVHType = DVHType.CUMULATIVE,
                figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None):
        """
        Vẽ biểu đồ DVH.
        
        Parameters
        ----------
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần vẽ, None = tất cả
        dvh_type : DVHType
            Loại DVH cần vẽ
        figsize : Tuple[int, int]
            Kích thước hình (inch)
        save_path : str, optional
            Đường dẫn để lưu hình, None = không lưu
        """
        if not self.dvh_data:
            logger.error("Không thể vẽ DVH - chưa tính toán dữ liệu DVH")
            return
            
        # Xác định danh sách cấu trúc cần vẽ
        structures_to_plot = structure_ids if structure_ids else list(self.dvh_data.keys())
        
        # Tạo figure
        plt.figure(figsize=figsize)
        
        # Lặp qua các cấu trúc và vẽ DVH
        for struct_id in structures_to_plot:
            if struct_id not in self.dvh_data:
                continue
                
            dvh = self.dvh_data[struct_id]
            struct_name = dvh['structure_name']
            
            # Chọn loại DVH để vẽ
            if dvh_type == DVHType.CUMULATIVE:
                y_data = dvh['cumulative']
                ylabel = 'Volume (%)'
                title = 'Cumulative Dose Volume Histogram'
            else:
                y_data = dvh['differential']
                ylabel = 'Volume Differential (%)'
                title = 'Differential Dose Volume Histogram'
                
            # Vẽ đường DVH
            plt.plot(dvh['bin_centers'], y_data * 100.0, label=struct_name)
            
        # Thiết lập các thuộc tính đồ thị
        plt.xlabel('Dose (Gy)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        # Lưu hình nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close()
        
    def calculate_quality_metrics(self, structure_ids: Optional[List[str]] = None, 
                                 prescription_dose: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Tính toán các chỉ số đánh giá chất lượng kế hoạch.
        
        Parameters
        ----------
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần tính, None = tất cả
        prescription_dose : Dict[str, float], optional
            Dictionary chứa liều kê đơn cho mỗi cấu trúc
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary chứa các chỉ số chất lượng cho mỗi cấu trúc
        """
        if not self.dvh_data:
            logger.error("Không thể tính chỉ số - chưa tính toán dữ liệu DVH")
            return {}
            
        # Xác định danh sách cấu trúc cần tính
        structures_to_analyze = structure_ids if structure_ids else list(self.dvh_data.keys())
        
        for struct_id in structures_to_analyze:
            if struct_id not in self.dvh_data:
                continue
                
            dvh = self.dvh_data[struct_id]
            struct_type = dvh['structure_type']
            
            # Khởi tạo dictionary cho các metric
            metrics = {
                'min_dose': dvh['min_dose'],
                'max_dose': dvh['max_dose'],
                'mean_dose': dvh['mean_dose'],
                'median_dose': dvh['median_dose'],
                'volume_cc': dvh['volume_cc']
            }
            
            # Tính các chỉ số đặc biệt cho cấu trúc mục tiêu
            if struct_type == 'TARGET' and prescription_dose and struct_id in prescription_dose:
                rx_dose = prescription_dose[struct_id]
                metrics.update(self._calculate_target_metrics(dvh, rx_dose))
                
            # Tính các chỉ số đặc biệt cho cơ quan nguy cấp
            elif struct_type == 'OAR':
                metrics.update(self._calculate_oar_metrics(dvh))
                
            # Lưu kết quả
            self.metrics[struct_id] = metrics
            
        return self.metrics
        
    def _calculate_target_metrics(self, dvh: Dict[str, Any], rx_dose: float) -> Dict[str, float]:
        """
        Tính toán các chỉ số chất lượng cho cấu trúc mục tiêu.
        
        Parameters
        ----------
        dvh : Dict[str, Any]
            Dữ liệu DVH của cấu trúc
        rx_dose : float
            Liều kê đơn (Gy)
            
        Returns
        -------
        Dict[str, float]
            Dictionary chứa các chỉ số chất lượng
        """
        bin_centers = dvh['bin_centers']
        cumulative = dvh['cumulative']
        
        # Nội suy để tìm các giá trị Dx và Vx
        metrics = {}
        
        # D95: liều nhận bởi ít nhất 95% thể tích
        d95_index = np.argmin(np.abs(cumulative - 0.95))
        metrics['D95'] = bin_centers[d95_index]
        
        # D98: liều nhận bởi ít nhất 98% thể tích
        d98_index = np.argmin(np.abs(cumulative - 0.98))
        metrics['D98'] = bin_centers[d98_index]
        
        # D50: liều nhận bởi ít nhất 50% thể tích
        d50_index = np.argmin(np.abs(cumulative - 0.5))
        metrics['D50'] = bin_centers[d50_index]
        
        # D2: liều nhận bởi ít nhất 2% thể tích
        d2_index = np.argmin(np.abs(cumulative - 0.02))
        metrics['D2'] = bin_centers[d2_index]
        
        # V95: % thể tích nhận ít nhất 95% liều kê đơn
        v95_dose = 0.95 * rx_dose
        v95_index = np.argmin(np.abs(bin_centers - v95_dose))
        metrics['V95'] = cumulative[v95_index] * 100.0  # %
        
        # Conformity Index (CI): Tỷ lệ thể tích nhận liều kê đơn và thể tích mục tiêu
        # Đây là công thức đơn giản, trong thực tế cần tính toán phức tạp hơn
        rx_index = np.argmin(np.abs(bin_centers - rx_dose))
        v100 = cumulative[rx_index]
        metrics['CI'] = v100
        
        # Homogeneity Index (HI): (D2 - D98) / D50
        hi = (metrics['D2'] - metrics['D98']) / metrics['D50'] if metrics['D50'] > 0 else float('inf')
        metrics['HI'] = hi
        
        return metrics
        
    def _calculate_oar_metrics(self, dvh: Dict[str, Any]) -> Dict[str, float]:
        """
        Tính toán các chỉ số chất lượng cho cơ quan nguy cấp.
        
        Parameters
        ----------
        dvh : Dict[str, Any]
            Dữ liệu DVH của cấu trúc
            
        Returns
        -------
        Dict[str, float]
            Dictionary chứa các chỉ số chất lượng
        """
        bin_centers = dvh['bin_centers']
        cumulative = dvh['cumulative']
        
        # Nội suy để tìm các giá trị Dx và Vx cho OAR
        metrics = {}
        
        # Các ngưỡng liều phổ biến cho OAR (Gy)
        dose_thresholds = [5, 10, 15, 20, 30, 40, 50]
        
        # Tính % thể tích nhận liều cao hơn ngưỡng
        for dose in dose_thresholds:
            if dose > np.max(bin_centers):
                continue
                
            dose_index = np.argmin(np.abs(bin_centers - dose))
            metrics[f'V{dose}'] = cumulative[dose_index] * 100.0  # %
            
        # D1cc: liều nhận bởi 1cc thể tích
        # Cần biết tổng thể tích và kích thước bin
        total_volume_cc = dvh['volume_cc']
        if total_volume_cc > 1.0:
            v1cc_ratio = 1.0 / total_volume_cc
            v1cc_index = np.argmin(np.abs(cumulative - v1cc_ratio))
            metrics['D1cc'] = bin_centers[v1cc_index]
            
        return metrics
    
    def compare_plans(self, other_evaluation: 'PlanEvaluation',
                    structure_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        So sánh kế hoạch hiện tại với kế hoạch khác.
        
        Parameters
        ----------
        other_evaluation : PlanEvaluation
            Đối tượng đánh giá kế hoạch khác
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần so sánh, None = tất cả
            
        Returns
        -------
        Dict[str, Dict[str, Tuple[float, float]]]
            Dictionary chứa sự khác nhau giữa các chỉ số chất lượng
        """
        if not self.metrics or not other_evaluation.metrics:
            logger.error("Không thể so sánh - chưa tính toán chỉ số chất lượng")
            return {}
            
        # Xác định danh sách cấu trúc cần so sánh
        structures_to_compare = structure_ids if structure_ids else set(self.metrics.keys()) & set(other_evaluation.metrics.keys())
        
        comparison = {}
        
        for struct_id in structures_to_compare:
            if struct_id not in self.metrics or struct_id not in other_evaluation.metrics:
                continue
                
            this_metrics = self.metrics[struct_id]
            other_metrics = other_evaluation.metrics[struct_id]
            
            # Tìm các chỉ số có trong cả hai kế hoạch
            common_metrics = set(this_metrics.keys()) & set(other_metrics.keys())
            
            # So sánh từng chỉ số
            struct_comparison = {}
            for metric in common_metrics:
                this_value = this_metrics[metric]
                other_value = other_metrics[metric]
                difference = this_value - other_value
                percent_difference = 100.0 * difference / other_value if other_value != 0 else float('inf')
                
                struct_comparison[metric] = (difference, percent_difference)
                
            comparison[struct_id] = struct_comparison
            
        return comparison
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng đánh giá thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin đánh giá
        """
        return {
            'plan_id': self.plan_id,
            'dvh_data': self.dvh_data,
            'metrics': self.metrics
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanEvaluation':
        """
        Tạo đối tượng PlanEvaluation từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin đánh giá
            
        Returns
        -------
        PlanEvaluation
            Đối tượng đánh giá kế hoạch
        """
        evaluation = cls(plan_id=data.get('plan_id', ''))
        
        # Phục hồi dữ liệu DVH
        if 'dvh_data' in data:
            evaluation.dvh_data = data['dvh_data']
            
        # Phục hồi chỉ số chất lượng
        if 'metrics' in data:
            evaluation.metrics = data['metrics']
            
        return evaluation


class DVHAnalysis:
    """
    Lớp phân tích biểu đồ Liều-Thể tích (DVH).
    
    Lớp này cung cấp các phương thức để tính toán, phân tích và trực quan hóa DVH
    cho các cấu trúc trong kế hoạch xạ trị.
    """
    
    def __init__(self, structure_id: str, structure_name: str, structure_type: str = ""):
        """
        Khởi tạo đối tượng phân tích DVH.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        structure_name : str
            Tên hiển thị của cấu trúc
        structure_type : str, optional
            Loại cấu trúc (TARGET, OAR, ...)
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.structure_type = structure_type
        self.bin_centers = None
        self.bin_edges = None
        self.differential = None
        self.cumulative = None
        self.min_dose = None
        self.max_dose = None
        self.mean_dose = None
        self.median_dose = None
        self.volume_cc = None
        
    def set_dvh_data(self, bin_centers: np.ndarray, bin_edges: np.ndarray, 
                   differential: np.ndarray, cumulative: np.ndarray, 
                   min_dose: float, max_dose: float, mean_dose: float, 
                   median_dose: float, volume_cc: float):
        """
        Đặt dữ liệu DVH.
        
        Parameters
        ----------
        bin_centers : np.ndarray
            Mảng chứa giá trị trung tâm của các bin
        bin_edges : np.ndarray
            Mảng chứa các cạnh của bin
        differential : np.ndarray
            Mảng chứa giá trị DVH vi phân
        cumulative : np.ndarray
            Mảng chứa giá trị DVH tích lũy
        min_dose : float
            Liều tối thiểu trong cấu trúc
        max_dose : float
            Liều tối đa trong cấu trúc
        mean_dose : float
            Liều trung bình trong cấu trúc
        median_dose : float
            Liều trung vị trong cấu trúc
        volume_cc : float
            Thể tích của cấu trúc (cc)
        """
        self.bin_centers = bin_centers
        self.bin_edges = bin_edges
        self.differential = differential
        self.cumulative = cumulative
        self.min_dose = min_dose
        self.max_dose = max_dose
        self.mean_dose = mean_dose
        self.median_dose = median_dose
        self.volume_cc = volume_cc


class PlanQualityMetrics:
    """
    Lớp tính toán và quản lý các chỉ số đánh giá chất lượng kế hoạch.
    
    Lớp này cung cấp các phương thức để tính toán và phân tích các chỉ số đánh giá
    kế hoạch xạ trị, bao gồm chỉ số đồng dạng, đồng nhất, gradient và các chỉ số khác.
    """
    
    def __init__(self, structure_id: str, structure_name: str, structure_type: str = ""):
        """
        Khởi tạo đối tượng chỉ số chất lượng.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        structure_name : str
            Tên hiển thị của cấu trúc
        structure_type : str, optional
            Loại cấu trúc (TARGET, OAR, ...)
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.structure_type = structure_type
        self.metrics = {}
        
    def add_metric(self, metric_name: str, value: float, unit: str = ""):
        """
        Thêm một chỉ số chất lượng.
        
        Parameters
        ----------
        metric_name : str
            Tên chỉ số
        value : float
            Giá trị chỉ số
        unit : str, optional
            Đơn vị của chỉ số
        """
        self.metrics[metric_name] = {
            'value': value,
            'unit': unit
        }
