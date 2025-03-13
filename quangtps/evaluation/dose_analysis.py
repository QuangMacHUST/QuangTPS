"""
Module phân tích liều cho đánh giá kế hoạch xạ trị.

Module này cung cấp các công cụ phân tích liều, tính toán DVH (Dose Volume Histogram),
và các thông số thống kê liều như D95, V20, liều trung bình, liều cực đại, v.v.
"""

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

class DoseAnalysis:
    """
    Lớp phân tích liều cho kế hoạch xạ trị.
    
    Lớp này cung cấp các phương thức để phân tích phân bố liều, tính toán
    thống kê liều, và tạo các đồ thị biểu diễn dữ liệu liều.
    """
    
    def __init__(self, 
                dose_grid: DoseGrid, 
                structures: Dict[str, np.ndarray] = None):
        """
        Khởi tạo đối tượng phân tích liều.
        
        Parameters:
            dose_grid (DoseGrid): Lưới liều cần phân tích
            structures (dict, optional): Dict các cấu trúc (ROI), với key là tên cấu trúc
                                         và value là mảng mask 3D
        """
        self.dose_grid = dose_grid
        self.structures = structures if structures is not None else {}
        self.dvh_data = {}  # Lưu trữ dữ liệu DVH đã tính toán
        
        # Thông tin chung
        self.dose_array = dose_grid.get_grid_data()
        self.shape = dose_grid.get_shape()
        self.spacing = dose_grid.spacing
        self.origin = dose_grid.origin
        
        # Tính thống kê toàn cục
        self.min_dose = np.min(self.dose_array)
        self.max_dose = np.max(self.dose_array)
        self.mean_dose = np.mean(self.dose_array)
    
    def set_structures(self, structures: Dict[str, np.ndarray]):
        """
        Đặt cấu trúc (ROI) để phân tích.
        
        Parameters:
            structures (dict): Dict các cấu trúc, với key là tên cấu trúc
                              và value là mảng mask 3D
        """
        self.structures = structures
        # Xóa dữ liệu DVH đã tính toán trước đó
        self.dvh_data = {}
    
    def add_structure(self, name: str, mask: np.ndarray):
        """
        Thêm một cấu trúc (ROI) mới để phân tích.
        
        Parameters:
            name (str): Tên cấu trúc
            mask (np.ndarray): Mảng mask 3D của cấu trúc
        
        Raises:
            ValueError: Nếu kích thước mask không khớp với lưới liều
        """
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match dose grid shape {self.shape}")
        
        self.structures[name] = mask
        # Xóa dữ liệu DVH đã tính toán cho cấu trúc này
        if name in self.dvh_data:
            del self.dvh_data[name]
    
    def calculate_dvh(self, 
                     structure_name: str, 
                     bins: int = 100, 
                     dose_range: Optional[Tuple[float, float]] = None,
                     relative_volume: bool = True,
                     cumulative: bool = True) -> Dict[str, np.ndarray]:
        """
        Tính toán Dose Volume Histogram (DVH) cho một cấu trúc.
        
        Parameters:
            structure_name (str): Tên cấu trúc
            bins (int, optional): Số lượng bins trong histogram
            dose_range (tuple, optional): Khoảng liều (min, max) để tính DVH
            relative_volume (bool, optional): Sử dụng thể tích tương đối (%)
            cumulative (bool, optional): Tính DVH tích lũy (True) hoặc vi phân (False)
        
        Returns:
            dict: Dict chứa thông tin DVH với các key:
                - 'dose': Mảng giá trị liều
                - 'volume': Mảng giá trị thể tích
                - 'type': Loại DVH ('cumulative' hoặc 'differential')
                - 'structure_name': Tên cấu trúc
                - 'relative_volume': Sử dụng thể tích tương đối hay không
        
        Raises:
            ValueError: Nếu không tìm thấy cấu trúc
        """
        # Kiểm tra cấu trúc có tồn tại không
        if structure_name not in self.structures:
            raise ValueError(f"Structure '{structure_name}' not found")
        
        # Kiểm tra nếu DVH đã được tính toán trước đó
        cache_key = f"{structure_name}_{bins}_{dose_range}_{relative_volume}_{cumulative}"
        if cache_key in self.dvh_data:
            return self.dvh_data[cache_key]
        
        # Lấy mask của cấu trúc
        mask = self.structures[structure_name]
        
        # Lấy giá trị liều trong cấu trúc
        dose_in_structure = self.dose_array[mask > 0]
        
        # Nếu không có voxel nào trong cấu trúc
        if len(dose_in_structure) == 0:
            logger.warning(f"No voxels found in structure '{structure_name}'")
            return {
                'dose': np.array([0]),
                'volume': np.array([0]),
                'type': 'cumulative' if cumulative else 'differential',
                'structure_name': structure_name,
                'relative_volume': relative_volume
            }
        
        # Xác định khoảng liều
        if dose_range is None:
            min_dose = 0
            max_dose = np.max(dose_in_structure) * 1.05  # Thêm lề 5%
        else:
            min_dose, max_dose = dose_range
        
        # Tính histogram
        hist, bin_edges = np.histogram(dose_in_structure, bins=bins, range=(min_dose, max_dose))
        
        # Tính giá trị thể tích
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        volume = hist / len(dose_in_structure) * 100 if relative_volume else hist * np.prod(self.spacing)
        
        # Tính DVH tích lũy nếu yêu cầu
        if cumulative:
            volume = np.cumsum(volume[::-1])[::-1]
        
        # Lưu kết quả
        result = {
            'dose': bin_centers,
            'volume': volume,
            'type': 'cumulative' if cumulative else 'differential',
            'structure_name': structure_name,
            'relative_volume': relative_volume
        }
        
        # Lưu vào cache
        self.dvh_data[cache_key] = result
        
        return result
    
    def calculate_dose_statistics(self, structure_name: str) -> Dict[str, float]:
        """
        Tính toán các thống kê liều cho một cấu trúc.
        
        Parameters:
            structure_name (str): Tên cấu trúc
        
        Returns:
            dict: Dict các thống kê liều với các key:
                - 'min': Liều tối thiểu (Gy)
                - 'max': Liều tối đa (Gy)
                - 'mean': Liều trung bình (Gy)
                - 'median': Liều trung vị (Gy)
                - 'std': Độ lệch chuẩn liều (Gy)
        
        Raises:
            ValueError: Nếu không tìm thấy cấu trúc
        """
        # Kiểm tra cấu trúc có tồn tại không
        if structure_name not in self.structures:
            raise ValueError(f"Structure '{structure_name}' not found")
        
        # Lấy mask của cấu trúc
        mask = self.structures[structure_name]
        
        # Lấy giá trị liều trong cấu trúc
        dose_in_structure = self.dose_array[mask > 0]
        
        # Nếu không có voxel nào trong cấu trúc
        if len(dose_in_structure) == 0:
            logger.warning(f"No voxels found in structure '{structure_name}'")
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0
            }
        
        # Tính các thống kê
        return {
            'min': np.min(dose_in_structure),
            'max': np.max(dose_in_structure),
            'mean': np.mean(dose_in_structure),
            'median': np.median(dose_in_structure),
            'std': np.std(dose_in_structure)
        }
    
    def calculate_dx(self, structure_name: str, x: float) -> float:
        """
        Tính toán liều nhận bởi x% thể tích (Dx).
        
        Parameters:
            structure_name (str): Tên cấu trúc
            x (float): Phần trăm thể tích (0-100)
        
        Returns:
            float: Giá trị liều Dx (Gy)
        
        Raises:
            ValueError: Nếu không tìm thấy cấu trúc hoặc x không hợp lệ
        """
        if x < 0 or x > 100:
            raise ValueError(f"Percentage x must be between 0 and 100, got {x}")
        
        # Tính DVH tích lũy
        dvh = self.calculate_dvh(structure_name, bins=1000, cumulative=True, relative_volume=True)
        
        # Nội suy để tìm liều tại x% thể tích
        dose = np.interp(x, dvh['volume'][::-1], dvh['dose'][::-1])
        
        return dose
    
    def calculate_vx(self, structure_name: str, x: float, relative: bool = True) -> float:
        """
        Tính toán thể tích nhận liều ≥ x Gy (Vx).
        
        Parameters:
            structure_name (str): Tên cấu trúc
            x (float): Ngưỡng liều (Gy)
            relative (bool, optional): Trả về thể tích tương đối (%) nếu True, 
                                      ngược lại trả về thể tích tuyệt đối (cc)
        
        Returns:
            float: Giá trị thể tích Vx (% hoặc cc)
        
        Raises:
            ValueError: Nếu không tìm thấy cấu trúc
        """
        # Tính DVH tích lũy
        dvh = self.calculate_dvh(structure_name, bins=1000, cumulative=True, relative_volume=relative)
        
        # Nội suy để tìm thể tích tại liều x Gy
        volume = np.interp(x, dvh['dose'], dvh['volume'])
        
        return volume
    
    def calculate_conformity_index(self, 
                                  target_name: str, 
                                  reference_dose: float) -> float:
        """
        Tính toán chỉ số phù hợp (Conformity Index).
        
        CI = V_ref / V_target, với V_ref là thể tích nhận ít nhất liều tham chiếu
        và V_target là thể tích của target.
        
        Parameters:
            target_name (str): Tên cấu trúc target
            reference_dose (float): Liều tham chiếu (Gy)
        
        Returns:
            float: Chỉ số phù hợp (CI)
        
        Raises:
            ValueError: Nếu không tìm thấy cấu trúc target
        """
        # Kiểm tra cấu trúc có tồn tại không
        if target_name not in self.structures:
            raise ValueError(f"Target structure '{target_name}' not found")
        
        # Lấy mask của target
        target_mask = self.structures[target_name]
        
        # Tính thể tích target (cc)
        target_volume = np.sum(target_mask) * np.prod(self.spacing) / 1000.0
        
        # Tạo mask vùng nhận ít nhất liều tham chiếu
        reference_mask = self.dose_array >= reference_dose
        
        # Tính thể tích vùng nhận ít nhất liều tham chiếu (cc)
        reference_volume = np.sum(reference_mask) * np.prod(self.spacing) / 1000.0
        
        # Tính CI
        ci = reference_volume / target_volume if target_volume > 0 else float('inf')
        
        return ci
    
    def calculate_homogeneity_index(self, 
                                   target_name: str, 
                                   prescription_dose: float) -> float:
        """
        Tính toán chỉ số đồng nhất (Homogeneity Index).
        
        HI = (D2% - D98%) / D50%, với Dx% là liều nhận bởi x% thể tích.
        
        Parameters:
            target_name (str): Tên cấu trúc target
            prescription_dose (float): Liều kê đơn (Gy)
        
        Returns:
            float: Chỉ số đồng nhất (HI)
        
        Raises:
            ValueError: Nếu không tìm thấy cấu trúc target
        """
        # Tính D2%, D98%, và D50%
        d2 = self.calculate_dx(target_name, 2)
        d98 = self.calculate_dx(target_name, 98)
        d50 = self.calculate_dx(target_name, 50)
        
        # Tính HI
        hi = (d2 - d98) / d50 if d50 > 0 else float('inf')
        
        return hi
    
    def calculate_gradient_index(self, 
                               target_name: str, 
                               reference_dose: float,
                               half_reference_dose: Optional[float] = None) -> float:
        """
        Tính toán chỉ số gradient (Gradient Index).
        
        GI = V_(0.5*Rx) / V_Rx, với V_Rx là thể tích nhận ít nhất liều Rx
        và V_(0.5*Rx) là thể tích nhận ít nhất một nửa liều Rx.
        
        Parameters:
            target_name (str): Tên cấu trúc target
            reference_dose (float): Liều tham chiếu (Gy)
            half_reference_dose (float, optional): Nửa liều tham chiếu, nếu không 
                                                  cung cấp sẽ mặc định là reference_dose / 2
        
        Returns:
            float: Chỉ số gradient (GI)
        
        Raises:
            ValueError: Nếu không tìm thấy cấu trúc target
        """
        # Kiểm tra half_reference_dose
        if half_reference_dose is None:
            half_reference_dose = reference_dose / 2.0
        
        # Tạo mask vùng nhận ít nhất liều tham chiếu
        reference_mask = self.dose_array >= reference_dose
        
        # Tạo mask vùng nhận ít nhất nửa liều tham chiếu
        half_reference_mask = self.dose_array >= half_reference_dose
        
        # Tính thể tích (cc)
        reference_volume = np.sum(reference_mask) * np.prod(self.spacing) / 1000.0
        half_reference_volume = np.sum(half_reference_mask) * np.prod(self.spacing) / 1000.0
        
        # Tính GI
        gi = half_reference_volume / reference_volume if reference_volume > 0 else float('inf')
        
        return gi
    
    def plot_dvh(self, 
                structure_names: List[str], 
                title: str = "Dose Volume Histogram",
                figsize: Tuple[int, int] = (10, 6),
                colors: Optional[Dict[str, str]] = None,
                save_path: Optional[str] = None) -> Any:
        """
        Vẽ đồ thị DVH cho một hoặc nhiều cấu trúc.
        
        Parameters:
            structure_names (list): Danh sách tên các cấu trúc
            title (str, optional): Tiêu đề đồ thị
            figsize (tuple, optional): Kích thước đồ thị (inch)
            colors (dict, optional): Dict màu sắc cho mỗi cấu trúc (key: tên cấu trúc, value: mã màu)
            save_path (str, optional): Đường dẫn để lưu đồ thị, nếu không cung cấp sẽ hiển thị đồ thị
        
        Returns:
            matplotlib.figure.Figure: Đối tượng Figure
        
        Raises:
            ValueError: Nếu không tìm thấy một trong các cấu trúc
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Màu sắc mặc định
        default_colors = plt.cm.Set1.colors
        
        if colors is None:
            colors = {}
        
        for i, name in enumerate(structure_names):
            # Tính DVH tích lũy
            try:
                dvh = self.calculate_dvh(name, bins=100, cumulative=True, relative_volume=True)
            except ValueError as e:
                logger.error(f"Error calculating DVH for structure '{name}': {str(e)}")
                continue
            
            # Chọn màu
            color = colors.get(name, default_colors[i % len(default_colors)])
            
            # Vẽ đường DVH
            ax.plot(dvh['dose'], dvh['volume'], label=name, color=color, linewidth=2)
        
        # Thiết lập đồ thị
        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Volume (%)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 100.5)  # Đảm bảo biểu đồ bắt đầu từ 0 và kết thúc trên 100%
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Lưu hoặc hiển thị
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
