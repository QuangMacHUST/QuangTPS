#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích biểu đồ liều-thể tích (DVH) cho hệ thống QuangTPS.

Module này cung cấp các công cụ để tính toán, hiển thị và phân tích DVH
(Dose-Volume Histogram) cho các cấu trúc trong kế hoạch điều trị.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DVHAnalyzer:
    """
    Lớp phân tích biểu đồ liều-thể tích (DVH).
    
    Cung cấp các phương thức để tính toán, phân tích và hiển thị DVH
    cho các cấu trúc trong kế hoạch điều trị.
    """
    
    def __init__(self, dose_matrix: np.ndarray = None, structure_mask: np.ndarray = None):
        """
        Khởi tạo phân tích DVH với ma trận liều và mặt nạ cấu trúc.
        
        Args:
            dose_matrix: Ma trận 3D chứa dữ liệu liều
            structure_mask: Ma trận boolean 3D đánh dấu vị trí của cấu trúc
        """
        self.dose_matrix = dose_matrix
        self.structure_mask = structure_mask
        self.dvh_data = None
        self.cumulative = True  # Mặc định là DVH tích lũy
        self.normalized = False  # Mặc định không chuẩn hóa
        self.bin_width = 0.1  # Độ rộng bin mặc định là 0.1 Gy
        
    def set_data(self, dose_matrix: np.ndarray, structure_mask: np.ndarray):
        """
        Cập nhật dữ liệu liều và mặt nạ cấu trúc.
        
        Args:
            dose_matrix: Ma trận 3D chứa dữ liệu liều
            structure_mask: Ma trận boolean 3D đánh dấu vị trí của cấu trúc
        """
        self.dose_matrix = dose_matrix
        self.structure_mask = structure_mask
        self.dvh_data = None  # Reset dữ liệu DVH
    
    def calculate_dvh(self, bin_width: float = 0.1, cumulative: bool = True, 
                     normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính toán Biểu đồ Liều-Thể tích (DVH).
        
        Args:
            bin_width: Độ rộng bin liều (Gy)
            cumulative: Nếu True, tính DVH tích lũy, ngược lại tính DVH vi phân
            normalized: Nếu True, chuẩn hóa thể tích về phần trăm
            
        Returns:
            Tuple chứa trục liều và trục thể tích
        """
        if self.dose_matrix is None or self.structure_mask is None:
            logger.error("Không thể tính DVH: Thiếu dữ liệu liều hoặc mặt nạ cấu trúc")
            return np.array([]), np.array([])
        
        # Lấy giá trị liều trong cấu trúc
        structure_dose = self.dose_matrix[self.structure_mask]
        
        if len(structure_dose) == 0:
            logger.warning("Không có voxel nào trong cấu trúc để tính DVH")
            return np.array([]), np.array([])
        
        # Xác định khoảng giá trị liều
        max_dose = np.max(structure_dose)
        dose_bins = np.arange(0, max_dose + bin_width, bin_width)
        
        # Tính histogram
        hist, edges = np.histogram(structure_dose, bins=dose_bins)
        
        # Tính thể tích
        voxel_count = len(structure_dose)
        volume_ratio = hist / voxel_count
        
        # Chuẩn hóa nếu cần
        if normalized:
            volume = volume_ratio * 100  # Chuyển sang phần trăm
        else:
            volume = volume_ratio * voxel_count  # Giữ nguyên số voxel
        
        # Tính DVH tích lũy nếu cần
        if cumulative:
            volume = np.cumsum(volume[::-1])[::-1]
        
        # Lưu trữ thông số
        self.bin_width = bin_width
        self.cumulative = cumulative
        self.normalized = normalized
        
        # Lưu lại dữ liệu DVH
        dose_points = edges[:-1]  # Lấy điểm giữa của mỗi bin
        self.dvh_data = (dose_points, volume)
        
        return dose_points, volume
    
    def get_dose_constraint_metrics(self) -> Dict[str, float]:
        """
        Tính toán các chỉ số ràng buộc liều phổ biến.
        
        Returns:
            Dict chứa các chỉ số DVH (D95, D50, V20, v.v.)
        """
        if self.dvh_data is None:
            logger.warning("Không có dữ liệu DVH để tính các chỉ số")
            return {}
        
        dose_points, volume = self.dvh_data
        total_volume = volume[0] if self.cumulative else np.sum(volume)
        
        metrics = {}
        
        # Tính các chỉ số Dxx (liều nhận bởi xx% thể tích)
        for percent in [98, 95, 90, 50, 5, 2]:
            target_volume = total_volume * percent / 100.0
            if self.cumulative:
                # Tìm liều tương ứng với thể tích
                idx = np.argmin(np.abs(volume - target_volume))
                metrics[f'D{percent}'] = dose_points[idx]
            else:
                # Với DVH vi phân, tính tích phân tới ngưỡng
                cumulative_vol = np.cumsum(volume[::-1])[::-1]
                idx = np.argmin(np.abs(cumulative_vol - target_volume))
                metrics[f'D{percent}'] = dose_points[idx]
        
        # Tính Dmean
        if self.cumulative:
            # Đối với DVH tích lũy, cần chuyển về dạng vi phân
            diff_volume = np.diff(np.append(volume, 0))
            metrics['Dmean'] = np.sum(dose_points * diff_volume * (-1)) / total_volume
        else:
            metrics['Dmean'] = np.sum(dose_points * volume) / total_volume
        
        # Tính Dmin và Dmax
        if self.cumulative:
            metrics['Dmax'] = dose_points[0]  # Giá trị đầu tiên của DVH tích lũy
            # Tìm điểm mà thể tích gần như bằng 0
            near_zero_idx = np.where(volume < 0.01 * total_volume)[0]
            metrics['Dmin'] = dose_points[near_zero_idx[0]] if len(near_zero_idx) > 0 else dose_points[-1]
        else:
            non_zero_doses = dose_points[volume > 0]
            metrics['Dmax'] = np.max(non_zero_doses) if len(non_zero_doses) > 0 else 0
            metrics['Dmin'] = np.min(non_zero_doses) if len(non_zero_doses) > 0 else 0
        
        # Tính các chỉ số Vxx (thể tích nhận liều xx Gy)
        for dose_level in [5, 10, 20, 30, 40, 50, 60]:
            if dose_level <= np.max(dose_points):
                idx = np.argmin(np.abs(dose_points - dose_level))
                if self.normalized:
                    metrics[f'V{dose_level}'] = volume[idx]  # Đã là phần trăm
                else:
                    metrics[f'V{dose_level}'] = volume[idx] / total_volume * 100  # Chuyển sang phần trăm
            else:
                metrics[f'V{dose_level}'] = 0.0
        
        return metrics
    
    def plot_dvh(self, ax=None, label: str = None, color: str = None, 
                linestyle: str = '-', marker: str = None) -> plt.Axes:
        """
        Vẽ biểu đồ DVH.
        
        Args:
            ax: Trục để vẽ DVH. Nếu None, tạo trục mới
            label: Nhãn cho đường DVH
            color: Màu của đường DVH
            linestyle: Kiểu đường
            marker: Kiểu đánh dấu
            
        Returns:
            Trục matplotlib đã vẽ DVH
        """
        if self.dvh_data is None:
            logger.warning("Không có dữ liệu DVH để vẽ")
            return ax or plt.gca()
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        dose_points, volume = self.dvh_data
        
        ax.plot(dose_points, volume, label=label, color=color, 
               linestyle=linestyle, marker=marker)
        
        # Thiết lập nhãn trục
        volume_label = "Thể tích (%)" if self.normalized else "Thể tích (cc)"
        dvh_type = "Tích lũy" if self.cumulative else "Vi phân"
        
        ax.set_xlabel("Liều (Gy)")
        ax.set_ylabel(volume_label)
        ax.set_title(f"Biểu đồ Liều-Thể tích ({dvh_type})")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if label:
            ax.legend()
        
        return ax
    
    def compare_dvhs(self, other_dvhs: List['DVHAnalyzer'], labels: List[str] = None, 
                    colors: List[str] = None, title: str = "So sánh DVH") -> plt.Figure:
        """
        So sánh nhiều DVH trên cùng một biểu đồ.
        
        Args:
            other_dvhs: Danh sách các đối tượng DVHAnalyzer khác
            labels: Danh sách nhãn cho mỗi DVH
            colors: Danh sách màu cho mỗi DVH
            title: Tiêu đề biểu đồ
            
        Returns:
            Đối tượng Figure chứa biểu đồ so sánh
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        all_dvhs = [self] + other_dvhs
        
        if labels is None:
            labels = [f"DVH {i+1}" for i in range(len(all_dvhs))]
        
        if colors is None:
            # Tạo danh sách màu mặc định
            colors = plt.cm.tab10.colors
            colors = [colors[i % len(colors)] for i in range(len(all_dvhs))]
        
        for i, dvh in enumerate(all_dvhs):
            if dvh.dvh_data is not None:
                dvh.plot_dvh(ax=ax, label=labels[i], color=colors[i])
        
        ax.set_title(title)
        ax.legend()
        
        return fig
    
    def export_to_csv(self, filename: str) -> bool:
        """
        Xuất dữ liệu DVH ra file CSV.
        
        Args:
            filename: Đường dẫn đến file CSV
            
        Returns:
            True nếu xuất thành công, False nếu thất bại
        """
        if self.dvh_data is None:
            logger.error("Không có dữ liệu DVH để xuất")
            return False
        
        try:
            dose_points, volume = self.dvh_data
            
            # Tạo header
            dvh_type = "Cumulative" if self.cumulative else "Differential"
            volume_unit = "%" if self.normalized else "cc"
            header = f"Dose (Gy),Volume ({volume_unit}) - {dvh_type} DVH"
            
            # Lưu dữ liệu
            data = np.column_stack((dose_points, volume))
            np.savetxt(filename, data, delimiter=',', header=header, comments='')
            
            logger.info(f"Xuất DVH thành công vào file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất DVH: {e}")
            return False
    
    def calculate_conformity_index(self, prescription_dose: float, 
                                 reference_volume: float) -> float:
        """
        Tính chỉ số đồng dạng (Conformity Index).
        
        CI = (TV_PIV)^2 / (TV x PIV)
        TV_PIV: Thể tích khối u nhận liều chỉ định
        TV: Thể tích khối u
        PIV: Thể tích được bao bởi đường đẳng liều chỉ định
        
        Args:
            prescription_dose: Liều chỉ định (Gy)
            reference_volume: Thể tích khối u tham chiếu
            
        Returns:
            Chỉ số đồng dạng (0-1), 1 là tốt nhất
        """
        if self.dvh_data is None:
            logger.error("Không có dữ liệu DVH để tính chỉ số đồng dạng")
            return 0.0
        
        dose_points, volume = self.dvh_data
        
        # Tìm thể tích nhận liều chỉ định
        idx = np.argmin(np.abs(dose_points - prescription_dose))
        prescription_volume = volume[idx]
        
        # Trong trường hợp đơn giản này, chúng ta coi prescription_volume như là TV_PIV và PIV
        # Để tính chính xác cần có thêm thông tin về phân bố liều trên toàn bộ không gian
        
        if reference_volume > 0 and prescription_volume > 0:
            ci = (prescription_volume ** 2) / (reference_volume * prescription_volume)
            return min(ci, 1.0)  # CI không thể lớn hơn 1
        
        return 0.0
    
    def calculate_homogeneity_index(self) -> float:
        """
        Tính chỉ số đồng nhất (Homogeneity Index).
        
        HI = (D2% - D98%) / D50%
        
        Returns:
            Chỉ số đồng nhất, 0 là tốt nhất (hoàn toàn đồng nhất)
        """
        metrics = self.get_dose_constraint_metrics()
        
        d2 = metrics.get('D2', 0)
        d98 = metrics.get('D98', 0)
        d50 = metrics.get('D50', 0)
        
        if d50 > 0:
            hi = (d2 - d98) / d50
            return hi
        
        return float('inf')  # Đồng nhất rất kém
    
    def calculate_gradient_index(self, prescription_dose: float) -> float:
        """
        Tính chỉ số gradient (Gradient Index).
        
        GI = V50%prescription / Vprescription
        
        Args:
            prescription_dose: Liều chỉ định (Gy)
            
        Returns:
            Chỉ số gradient, giá trị thấp là tốt
        """
        if self.dvh_data is None:
            logger.error("Không có dữ liệu DVH để tính chỉ số gradient")
            return float('inf')
        
        dose_points, volume = self.dvh_data
        
        # Tìm thể tích nhận liều chỉ định
        idx_100 = np.argmin(np.abs(dose_points - prescription_dose))
        v_100 = volume[idx_100]
        
        # Tìm thể tích nhận 50% liều chỉ định
        idx_50 = np.argmin(np.abs(dose_points - prescription_dose * 0.5))
        v_50 = volume[idx_50]
        
        if v_100 > 0:
            return v_50 / v_100
        
        return float('inf') 