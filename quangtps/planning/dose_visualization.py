#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hiển thị phân bố liều trong QuangTPS.

Module này cung cấp các lớp và phương thức để hiển thị và phân tích phân bố liều
trong các kế hoạch xạ trị, bao gồm các chế độ hiển thị 2D và 3D.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
import os

logger = logging.getLogger(__name__)


class DoseColormap(str, Enum):
    """Các bảng màu cho hiển thị liều."""
    STANDARD = "Standard"  # Màu tiêu chuẩn (xanh -> đỏ)
    GRAYSCALE = "Grayscale"  # Thang độ xám
    HOT = "Hot"  # Nóng (đỏ -> vàng -> trắng)
    COOL = "Cool"  # Lạnh (xanh -> tím)
    JET = "Jet"  # Jet (xanh -> lục -> đỏ)
    CUSTOM = "Custom"  # Tùy chỉnh


class DoseDisplayMode(str, Enum):
    """Các chế độ hiển thị phân bố liều."""
    ABSOLUTE = "Absolute"  # Giá trị tuyệt đối (Gy)
    RELATIVE = "Relative"  # Giá trị tương đối (% của liều kê đơn)
    NORMALIZED = "Normalized"  # Chuẩn hóa (% của giá trị tối đa)


class DoseDisplay:
    """
    Lớp hiển thị phân bố liều.
    
    Lớp này quản lý việc hiển thị phân bố liều trong kế hoạch xạ trị,
    bao gồm các chế độ hiển thị, bảng màu và các tùy chọn khác.
    """
    
    def __init__(self, dose_data: np.ndarray, 
                spacing: Optional[Tuple[float, float, float]] = None, 
                origin: Optional[Tuple[float, float, float]] = None,
                prescription_dose: Optional[float] = None):
        """
        Khởi tạo đối tượng hiển thị liều.
        
        Parameters
        ----------
        dose_data : np.ndarray
            Mảng 3D chứa dữ liệu phân bố liều (Gy)
        spacing : Tuple[float, float, float], optional
            Khoảng cách giữa các điểm (mm) theo (x, y, z)
        origin : Tuple[float, float, float], optional
            Tọa độ gốc của phân bố liều (mm)
        prescription_dose : float, optional
            Liều kê đơn (Gy)
        """
        self.dose_data = dose_data
        self.spacing = spacing if spacing else (1.0, 1.0, 1.0)
        self.origin = origin if origin else (0.0, 0.0, 0.0)
        self.prescription_dose = prescription_dose
        
        self.display_mode = DoseDisplayMode.ABSOLUTE
        self.colormap = DoseColormap.STANDARD
        self.custom_colormap = None
        
        self.min_display_value = 0
        self.max_display_value = None
        self.isodose_levels = []
        
        # Khởi tạo các bảng màu có sẵn
        self._init_colormaps()
    
    def _init_colormaps(self):
        """Khởi tạo các bảng màu có sẵn."""
        self.colormaps = {
            DoseColormap.STANDARD: self._create_standard_colormap(),
            DoseColormap.GRAYSCALE: plt.cm.gray,
            DoseColormap.HOT: plt.cm.hot,
            DoseColormap.COOL: plt.cm.cool,
            DoseColormap.JET: plt.cm.jet
        }
    
    def _create_standard_colormap(self) -> LinearSegmentedColormap:
        """
        Tạo bảng màu tiêu chuẩn cho liều.
        
        Returns
        -------
        LinearSegmentedColormap
            Bảng màu tiêu chuẩn
        """
        # Màu tiêu chuẩn: từ xanh lam -> lục -> vàng -> đỏ
        cdict = {
            'red': [(0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0)],
                    
            'green': [(0.0, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 0.0, 0.0)],
                      
            'blue': [(0.0, 0.3, 0.3),
                     (0.25, 1.0, 1.0),
                     (0.5, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.0, 0.0, 0.0)]
        }
        
        return LinearSegmentedColormap('DoseStandard', cdict)
    
    def set_custom_colormap(self, cmap: Union[LinearSegmentedColormap, str]):
        """
        Đặt bảng màu tùy chỉnh.
        
        Parameters
        ----------
        cmap : Union[LinearSegmentedColormap, str]
            Đối tượng bảng màu hoặc tên bảng màu matplotlib
        """
        if isinstance(cmap, str):
            try:
                self.custom_colormap = plt.cm.get_cmap(cmap)
            except ValueError:
                logger.error(f"Không tìm thấy bảng màu {cmap}")
                return
        else:
            self.custom_colormap = cmap
            
        self.colormap = DoseColormap.CUSTOM
    
    def set_display_mode(self, mode: DoseDisplayMode):
        """
        Đặt chế độ hiển thị.
        
        Parameters
        ----------
        mode : DoseDisplayMode
            Chế độ hiển thị mới
        """
        self.display_mode = mode
    
    def set_display_range(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        """
        Đặt khoảng giá trị hiển thị.
        
        Parameters
        ----------
        min_value : float, optional
            Giá trị tối thiểu để hiển thị
        max_value : float, optional
            Giá trị tối đa để hiển thị
        """
        self.min_display_value = min_value if min_value is not None else 0
        self.max_display_value = max_value
    
    def set_isodose_levels(self, levels: List[float], mode: DoseDisplayMode = None):
        """
        Đặt các mức đường đồng liều.
        
        Parameters
        ----------
        levels : List[float]
            Danh sách các mức đường đồng liều theo đơn vị của chế độ hiển thị
        mode : DoseDisplayMode, optional
            Chế độ hiển thị cho các mức (nếu khác với chế độ hiện tại)
        """
        self.isodose_levels = levels
        if mode:
            self.display_mode = mode
    
    def _get_display_data(self, slice_data: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi dữ liệu liều theo chế độ hiển thị.
        
        Parameters
        ----------
        slice_data : np.ndarray
            Dữ liệu liều cần chuyển đổi
            
        Returns
        -------
        np.ndarray
            Dữ liệu đã chuyển đổi
        """
        if self.display_mode == DoseDisplayMode.ABSOLUTE:
            return slice_data
        elif self.display_mode == DoseDisplayMode.RELATIVE:
            if self.prescription_dose is None or self.prescription_dose == 0:
                logger.warning("Không có liều kê đơn, sử dụng chế độ chuẩn hóa thay thế")
                return slice_data / np.max(self.dose_data) * 100 if np.max(self.dose_data) > 0 else slice_data
            return slice_data / self.prescription_dose * 100
        elif self.display_mode == DoseDisplayMode.NORMALIZED:
            max_val = np.max(self.dose_data)
            return slice_data / max_val * 100 if max_val > 0 else slice_data
        
        return slice_data
    
    def _get_colormap(self) -> plt.cm:
        """
        Lấy bảng màu hiện tại.
        
        Returns
        -------
        plt.cm
            Bảng màu matplotlib
        """
        if self.colormap == DoseColormap.CUSTOM and self.custom_colormap:
            return self.custom_colormap
        return self.colormaps.get(self.colormap, self.colormaps[DoseColormap.STANDARD])
    
    def _get_display_limits(self, display_data: np.ndarray) -> Tuple[float, float]:
        """
        Lấy giới hạn hiển thị.
        
        Parameters
        ----------
        display_data : np.ndarray
            Dữ liệu hiển thị
            
        Returns
        -------
        Tuple[float, float]
            Giới hạn (min, max)
        """
        min_val = self.min_display_value if self.min_display_value is not None else np.min(display_data)
        max_val = self.max_display_value if self.max_display_value is not None else np.max(display_data)
        return min_val, max_val
    
    def display_slice(self, axis: str = 'z', index: Optional[int] = None, 
                     ax: Optional[plt.Axes] = None, show_colorbar: bool = True,
                     show_isodose: bool = True, show_structures: bool = False,
                     structures: Optional[Dict[str, np.ndarray]] = None,
                     structure_colors: Optional[Dict[str, str]] = None,
                     figsize: Tuple[int, int] = (8, 8),
                     title: Optional[str] = None) -> plt.Figure:
        """
        Hiển thị một lát cắt của phân bố liều.
        
        Parameters
        ----------
        axis : str
            Trục lát cắt ('x', 'y', hoặc 'z')
        index : int, optional
            Chỉ số lát cắt, None = lấy lát cắt giữa
        ax : plt.Axes, optional
            Axes để vẽ, None = tạo mới
        show_colorbar : bool
            Hiển thị thanh màu
        show_isodose : bool
            Hiển thị đường đồng liều
        show_structures : bool
            Hiển thị các cấu trúc
        structures : Dict[str, np.ndarray], optional
            Dict chứa mặt nạ nhị phân 3D cho mỗi cấu trúc
        structure_colors : Dict[str, str], optional
            Dict chứa màu cho mỗi cấu trúc
        figsize : Tuple[int, int]
            Kích thước hình (inch)
        title : str, optional
            Tiêu đề hình
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure matplotlib
        """
        # Xác định lát cắt dữ liệu
        if axis.lower() == 'x':
            if index is None:
                index = self.dose_data.shape[0] // 2
            slice_data = self.dose_data[index, :, :]
            xlabel, ylabel = 'Y', 'Z'
        elif axis.lower() == 'y':
            if index is None:
                index = self.dose_data.shape[1] // 2
            slice_data = self.dose_data[:, index, :]
            xlabel, ylabel = 'X', 'Z'
        else:  # 'z'
            if index is None:
                index = self.dose_data.shape[2] // 2
            slice_data = self.dose_data[:, :, index]
            xlabel, ylabel = 'X', 'Y'
        
        # Chuyển đổi dữ liệu theo chế độ hiển thị
        display_data = self._get_display_data(slice_data)
        vmin, vmax = self._get_display_limits(display_data)
        
        # Tạo hình mới nếu cần
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Hiển thị phân bố liều
        im = ax.imshow(display_data, cmap=self._get_colormap(), vmin=vmin, vmax=vmax, 
                      origin='lower', interpolation='nearest')
        
        # Hiển thị đường đồng liều
        if show_isodose and self.isodose_levels:
            contour_levels = []
            for level in self.isodose_levels:
                if self.display_mode == DoseDisplayMode.ABSOLUTE:
                    contour_levels.append(level)
                elif self.display_mode == DoseDisplayMode.RELATIVE and self.prescription_dose:
                    contour_levels.append(level * self.prescription_dose / 100)
                elif self.display_mode == DoseDisplayMode.NORMALIZED:
                    max_val = np.max(self.dose_data)
                    contour_levels.append(level * max_val / 100)
                    
            if contour_levels:
                contours = ax.contour(display_data, levels=contour_levels, 
                                     colors='white', linewidths=0.5, alpha=0.7)
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Hiển thị cấu trúc
        if show_structures and structures:
            for struct_name, struct_mask in structures.items():
                struct_color = structure_colors.get(struct_name, 'red') if structure_colors else 'red'
                
                # Lấy lát cắt của cấu trúc
                if axis.lower() == 'x':
                    struct_slice = struct_mask[index, :, :]
                elif axis.lower() == 'y':
                    struct_slice = struct_mask[:, index, :]
                else:  # 'z'
                    struct_slice = struct_mask[:, :, index]
                
                # Lấy đường viền
                from scipy import ndimage
                struct_erode = ndimage.binary_erosion(struct_slice).astype(int)
                struct_contour = struct_slice - struct_erode
                
                # Hiển thị đường viền
                y_idx, x_idx = np.where(struct_contour > 0)
                ax.scatter(x_idx, y_idx, s=1, c=struct_color, alpha=0.7, 
                          label=struct_name if struct_name not in ax.get_legend_handles_labels()[1] else '')
        
        # Thêm thanh màu
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            if self.display_mode == DoseDisplayMode.ABSOLUTE:
                cbar.set_label('Liều (Gy)')
            elif self.display_mode == DoseDisplayMode.RELATIVE:
                cbar.set_label('Liều (% của liều kê đơn)')
            elif self.display_mode == DoseDisplayMode.NORMALIZED:
                cbar.set_label('Liều (% của giá trị tối đa)')
        
        # Cấu hình trục
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Đặt tiêu đề
        if title:
            ax.set_title(title)
        else:
            ax_name = {'x': 'X', 'y': 'Y', 'z': 'Z'}.get(axis.lower(), 'Z')
            ax.set_title(f'Phân bố liều - Lát cắt {ax_name}={index}')
        
        # Hiển thị chú thích nếu có cấu trúc
        if show_structures and structures and len(structures) > 0:
            ax.legend(loc='upper right', fontsize=8)
        
        return fig
    
    def display_3d_isosurface(self, level: float, ax: Optional[plt.Axes] = None,
                             color: str = 'red', alpha: float = 0.5,
                             figsize: Tuple[int, int] = (10, 8),
                             title: Optional[str] = None) -> plt.Figure:
        """
        Hiển thị bề mặt đồng liều 3D.
        
        Parameters
        ----------
        level : float
            Mức đồng liều (theo đơn vị của chế độ hiển thị hiện tại)
        ax : plt.Axes, optional
            Axes 3D để vẽ, None = tạo mới
        color : str
            Màu bề mặt
        alpha : float
            Độ trong suốt (0-1)
        figsize : Tuple[int, int]
            Kích thước hình (inch)
        title : str, optional
            Tiêu đề hình
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure matplotlib
        """
        try:
            from skimage import measure
        except ImportError:
            logger.error("Cần cài đặt scikit-image để hiển thị bề mặt đồng liều 3D")
            raise ImportError("Cần cài đặt scikit-image để hiển thị bề mặt đồng liều 3D")
        
        # Chuyển đổi mức đồng liều theo chế độ hiển thị
        actual_level = level
        if self.display_mode == DoseDisplayMode.RELATIVE and self.prescription_dose:
            actual_level = level * self.prescription_dose / 100
        elif self.display_mode == DoseDisplayMode.NORMALIZED:
            max_val = np.max(self.dose_data)
            actual_level = level * max_val / 100
        
        # Tạo hình mới nếu cần
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Tạo lưới không gian
        x, y, z = np.indices(self.dose_data.shape)
        
        # Tính bề mặt đồng liều
        verts, faces, _, _ = measure.marching_cubes(self.dose_data, actual_level)
        
        # Điều chỉnh tọa độ theo spacing và origin
        verts[:, 0] = verts[:, 0] * self.spacing[0] + self.origin[0]
        verts[:, 1] = verts[:, 1] * self.spacing[1] + self.origin[1]
        verts[:, 2] = verts[:, 2] * self.spacing[2] + self.origin[2]
        
        # Hiển thị bề mặt đồng liều
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                      triangles=faces, color=color, alpha=alpha, shade=True)
        
        # Đặt nhãn trục
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Đặt tiêu đề
        if title:
            ax.set_title(title)
        else:
            unit = 'Gy'
            if self.display_mode == DoseDisplayMode.RELATIVE or self.display_mode == DoseDisplayMode.NORMALIZED:
                unit = '%'
            ax.set_title(f'Bề mặt đồng liều {level} {unit}')
        
        return fig
    
    def save_slice_image(self, file_path: str, axis: str = 'z', index: Optional[int] = None,
                        show_colorbar: bool = True, show_isodose: bool = True,
                        show_structures: bool = False, structures: Optional[Dict[str, np.ndarray]] = None,
                        structure_colors: Optional[Dict[str, str]] = None,
                        figsize: Tuple[int, int] = (8, 8), dpi: int = 300,
                        title: Optional[str] = None):
        """
        Lưu hình ảnh lát cắt ra file.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file
        axis : str
            Trục lát cắt ('x', 'y', hoặc 'z')
        index : int, optional
            Chỉ số lát cắt, None = lấy lát cắt giữa
        show_colorbar : bool
            Hiển thị thanh màu
        show_isodose : bool
            Hiển thị đường đồng liều
        show_structures : bool
            Hiển thị các cấu trúc
        structures : Dict[str, np.ndarray], optional
            Dict chứa mặt nạ nhị phân 3D cho mỗi cấu trúc
        structure_colors : Dict[str, str], optional
            Dict chứa màu cho mỗi cấu trúc
        figsize : Tuple[int, int]
            Kích thước hình (inch)
        dpi : int
            Độ phân giải (điểm trên inch)
        title : str, optional
            Tiêu đề hình
        """
        # Hiển thị lát cắt
        fig = self.display_slice(
            axis=axis,
            index=index,
            ax=None,
            show_colorbar=show_colorbar,
            show_isodose=show_isodose,
            show_structures=show_structures,
            structures=structures,
            structure_colors=structure_colors,
            figsize=figsize,
            title=title
        )
        
        # Lưu hình
        try:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Đã lưu hình ảnh vào {file_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu hình ảnh: {str(e)}")
    
    def create_dose_volume_histogram(self, structure_masks: Dict[str, np.ndarray],
                                   structure_names: Optional[Dict[str, str]] = None,
                                   ax: Optional[plt.Axes] = None,
                                   figsize: Tuple[int, int] = (10, 6),
                                   title: Optional[str] = None,
                                   colors: Optional[Dict[str, str]] = None,
                                   linewidth: float = 2.0,
                                   grid: bool = True) -> plt.Figure:
        """
        Tạo biểu đồ thể tích - liều (DVH).
        
        Parameters
        ----------
        structure_masks : Dict[str, np.ndarray]
            Dict chứa mặt nạ nhị phân 3D cho mỗi cấu trúc
        structure_names : Dict[str, str], optional
            Dict ánh xạ ID cấu trúc sang tên hiển thị
        ax : plt.Axes, optional
            Axes để vẽ, None = tạo mới
        figsize : Tuple[int, int]
            Kích thước hình (inch)
        title : str, optional
            Tiêu đề hình
        colors : Dict[str, str], optional
            Dict chứa màu cho mỗi cấu trúc
        linewidth : float
            Độ rộng đường
        grid : bool
            Hiển thị lưới
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure matplotlib
        """
        from quangtps.planning.evaluation import DVHAnalysis, DVHType
        
        # Tạo hình mới nếu cần
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Chuẩn bị màu
        if colors is None:
            # Sử dụng bảng màu mặc định
            color_cycle = plt.cm.tab10.colors
            colors = {}
            
        # Chuẩn bị tên cấu trúc
        if structure_names is None:
            structure_names = {}
        
        # Tính DVH cho mỗi cấu trúc
        for i, (struct_id, mask) in enumerate(structure_masks.items()):
            # Lấy dữ liệu liều trong cấu trúc
            struct_dose = self.dose_data[mask > 0]
            
            # Bỏ qua nếu không có điểm
            if len(struct_dose) == 0:
                continue
            
            # Tính DVH
            dvh_analysis = DVHAnalysis()
            bin_centers, cumulative_dvh = dvh_analysis.calculate_dvh_data(
                struct_dose, DVHType.CUMULATIVE
            )
            
            # Xác định màu và tên
            color = colors.get(struct_id, color_cycle[i % len(color_cycle)])
            name = structure_names.get(struct_id, struct_id)
            
            # Vẽ đường DVH
            ax.plot(bin_centers, cumulative_dvh * 100, '-', color=color, 
                   linewidth=linewidth, label=name)
        
        # Cấu hình biểu đồ
        ax.set_xlabel('Liều (Gy)')
        ax.set_ylabel('Thể tích (%)')
        ax.set_xlim(0, np.max(self.dose_data) * 1.05)
        ax.set_ylim(0, 105)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
            
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Biểu đồ thể tích - liều tích lũy (DVH)')
            
        # Thêm chú thích
        if len(structure_masks) > 0:
            ax.legend(loc='best')
            
        return fig
