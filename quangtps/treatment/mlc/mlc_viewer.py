#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hiển thị và trực quan hóa hệ thống MLC (Multi-Leaf Collimator).

Module này cung cấp các lớp và phương thức để hiển thị và 
trực quan hóa hệ thống MLC trong giao diện đồ họa.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.treatment.mlc.mlc_controller import MLCController
from quangtps.treatment.mlc.mlc_simulation import MLCSimulation

logger = logging.getLogger(__name__)

class MLCViewer:
    """
    Lớp hiển thị và trực quan hóa hệ thống MLC (Multi-Leaf Collimator).
    
    Lớp này cung cấp các phương thức để hiển thị và trực quan hóa
    hệ thống MLC trong giao diện đồ họa, bao gồm việc hiển thị
    vị trí của các lá, bản đồ fluence, và các thông số khác.
    """
    
    def __init__(self, mlc_model: MLCModel, controller: Optional[MLCController] = None, 
                 simulation: Optional[MLCSimulation] = None):
        """
        Khởi tạo một đối tượng hiển thị MLC.
        
        Parameters
        ----------
        mlc_model : MLCModel
            Mô hình MLC được sử dụng
        controller : Optional[MLCController], optional
            Bộ điều khiển MLC (nếu có)
        simulation : Optional[MLCSimulation], optional
            Mô phỏng MLC (nếu có)
        """
        self.mlc_model = mlc_model
        self.controller = controller
        self.simulation = simulation
        
        # Lưu trữ các hình ảnh và đồ thị
        self.figures = {}
        self.axes = {}
        self.artists = {}
        
        # Thiết lập màu sắc và kiểu hiển thị
        self.colors = {
            "leaf": "#1f77b4",         # Xanh dương
            "leaf_inactive": "#aec7e8", # Xanh dương nhạt
            "background": "#f5f5f5",   # Xám nhạt
            "field": "#ffff00",        # Vàng
            "beam": "#ff7f0e",         # Cam
            "dose": "#d62728",         # Đỏ
            "grid": "#cccccc"          # Xám
        }
    
    def create_leaf_view(self, figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Tạo hình ảnh hiển thị các lá MLC theo góc nhìn Beam's Eye View.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Kích thước hình ảnh
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Hình ảnh và đồ thị
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f"MLC Beam's Eye View - {self.mlc_model.get_model_name()}")
        
        # Thiết lập giới hạn đồ thị
        max_field_size = self.mlc_model.get_max_field_size()
        half_width = max_field_size[0] / 2
        half_height = max_field_size[1] / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)
        
        # Thiết lập các nhãn trục
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, linestyle="--", alpha=0.7, color=self.colors["grid"])
        
        # Lưu trữ hình ảnh và đồ thị
        self.figures["leaf_view"] = fig
        self.axes["leaf_view"] = ax
        
        return fig, ax
    
    def update_leaf_view(self, positions: Optional[Dict[int, float]] = None) -> None:
        """
        Cập nhật hình ảnh hiển thị các lá MLC.
        
        Parameters
        ----------
        positions : Optional[Dict[int, float]], optional
            Dictionary chứa vị trí của các lá, nếu None thì sẽ lấy từ controller
        """
        if "leaf_view" not in self.figures or "leaf_view" not in self.axes:
            logger.warning("Leaf view not created yet, call create_leaf_view() first")
            return
        
        ax = self.axes["leaf_view"]
        ax.clear()
        
        # Thiết lập lại giới hạn đồ thị
        max_field_size = self.mlc_model.get_max_field_size()
        half_width = max_field_size[0] / 2
        half_height = max_field_size[1] / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, linestyle="--", alpha=0.7, color=self.colors["grid"])
        
        # Lấy vị trí của các lá
        if positions is None and self.controller:
            positions = self.controller.get_current_positions()
        
        if not positions:
            logger.warning("No leaf positions provided and no controller available")
            return
        
        # Vẽ các lá MLC
        leaf_width = self.mlc_model.get_leaf_width()
        leaf_count = self.mlc_model.get_leaf_count()
        
        # Vẽ từng lá
        y_offset = -half_height + leaf_width / 2
        
        for i in range(leaf_count):
            # Giả sử các lá được chia đều vào hai bank với số lượng bằng nhau
            is_left_bank = i < leaf_count // 2
            
            # Lấy vị trí của lá
            leaf_id = i
            position = positions.get(leaf_id, 0)
            
            if is_left_bank:
                # Lá ở bank trái
                rect = plt.Rectangle(
                    (-half_width, y_offset),
                    half_width + position,
                    leaf_width,
                    color=self.colors["leaf"],
                    alpha=0.8
                )
            else:
                # Lá ở bank phải
                rect = plt.Rectangle(
                    (position, y_offset),
                    half_width - position,
                    leaf_width,
                    color=self.colors["leaf"],
                    alpha=0.8
                )
            
            ax.add_patch(rect)
            y_offset += leaf_width
        
        # Cập nhật hình ảnh
        self.figures["leaf_view"].canvas.draw_idle()
    
    def create_fluence_view(self, figsize: Tuple[int, int] = (8, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Tạo hình ảnh hiển thị bản đồ fluence (cường độ chùm tia).
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Kích thước hình ảnh
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Hình ảnh và đồ thị
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f"MLC Fluence Map - {self.mlc_model.get_model_name()}")
        
        # Thiết lập giới hạn đồ thị
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        
        # Khởi tạo bản đồ fluence trống
        max_field_size = self.mlc_model.get_max_field_size()
        resolution = (100, 100)
        extent = (-max_field_size[0]/2, max_field_size[0]/2, -max_field_size[1]/2, max_field_size[1]/2)
        fluence = np.zeros(resolution)
        
        # Hiển thị bản đồ fluence
        im = ax.imshow(fluence, cmap="hot", interpolation="bilinear", extent=extent, origin="lower")
        plt.colorbar(im, ax=ax, label="Relative Fluence")
        
        # Lưu trữ hình ảnh và đồ thị
        self.figures["fluence_view"] = fig
        self.axes["fluence_view"] = ax
        self.artists["fluence_view"] = im
        
        return fig, ax
    
    def update_fluence_view(self, fluence_map: Optional[np.ndarray] = None) -> None:
        """
        Cập nhật hình ảnh hiển thị bản đồ fluence.
        
        Parameters
        ----------
        fluence_map : Optional[np.ndarray], optional
            Bản đồ fluence, nếu None thì sẽ tính toán từ simulation
        """
        if "fluence_view" not in self.figures or "fluence_view" not in self.axes:
            logger.warning("Fluence view not created yet, call create_fluence_view() first")
            return
        
        # Lấy bản đồ fluence
        if fluence_map is None and self.simulation:
            fluence_map = self.simulation.compute_fluence_map()
        
        if fluence_map is None:
            logger.warning("No fluence map provided and no simulation available")
            return
        
        # Cập nhật bản đồ fluence
        self.artists["fluence_view"].set_data(fluence_map)
        self.artists["fluence_view"].set_clim(vmin=0, vmax=np.max(fluence_map) or 1)
        
        # Cập nhật hình ảnh
        self.figures["fluence_view"].canvas.draw_idle()
    
    def create_3d_view(self, figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Tạo hình ảnh hiển thị MLC theo góc nhìn 3D.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Kích thước hình ảnh
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Hình ảnh và đồ thị
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        fig.suptitle(f"MLC 3D View - {self.mlc_model.get_model_name()}")
        
        # Thiết lập giới hạn đồ thị
        max_field_size = self.mlc_model.get_max_field_size()
        half_width = max_field_size[0] / 2
        half_height = max_field_size[1] / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_width, half_width)
        ax.set_zlim(0, half_height)
        
        # Thiết lập các nhãn trục
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_zlabel("Y (mm)")
        
        # Lưu trữ hình ảnh và đồ thị
        self.figures["3d_view"] = fig
        self.axes["3d_view"] = ax
        
        return fig, ax
    
    def update_3d_view(self, positions: Optional[Dict[int, float]] = None) -> None:
        """
        Cập nhật hình ảnh hiển thị MLC theo góc nhìn 3D.
        
        Parameters
        ----------
        positions : Optional[Dict[int, float]], optional
            Dictionary chứa vị trí của các lá, nếu None thì sẽ lấy từ controller
        """
        if "3d_view" not in self.figures or "3d_view" not in self.axes:
            logger.warning("3D view not created yet, call create_3d_view() first")
            return
        
        ax = self.axes["3d_view"]
        ax.clear()
        
        # Thiết lập lại giới hạn đồ thị
        max_field_size = self.mlc_model.get_max_field_size()
        half_width = max_field_size[0] / 2
        half_height = max_field_size[1] / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_width, half_width)
        ax.set_zlim(0, half_height)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_zlabel("Y (mm)")
        
        # Lấy vị trí của các lá
        if positions is None and self.controller:
            positions = self.controller.get_current_positions()
        
        if not positions:
            logger.warning("No leaf positions provided and no controller available")
            return
        
        # Vẽ các lá MLC trong 3D
        # (đoạn mã thực tế sẽ phức tạp hơn nhiều để vẽ các lá trong 3D)
        
        # Cập nhật hình ảnh
        self.figures["3d_view"].canvas.draw_idle()
    
    def show(self, view_type: str = "leaf_view") -> None:
        """
        Hiển thị hình ảnh MLC.
        
        Parameters
        ----------
        view_type : str, optional
            Loại hình ảnh cần hiển thị, có thể là 'leaf_view', 'fluence_view', 
            hoặc '3d_view'
        """
        if view_type not in self.figures:
            logger.warning(f"View type {view_type} not created yet")
            return
        
        plt.figure(self.figures[view_type].number)
        plt.show()
    
    def save_to_file(self, view_type: str, filename: str, dpi: int = 300) -> bool:
        """
        Lưu hình ảnh MLC vào file.
        
        Parameters
        ----------
        view_type : str
            Loại hình ảnh cần lưu, có thể là 'leaf_view', 'fluence_view', 
            hoặc '3d_view'
        filename : str
            Tên file để lưu
        dpi : int, optional
            Độ phân giải của hình ảnh
            
        Returns
        -------
        bool
            True nếu lưu thành công, False nếu có lỗi
        """
        if view_type not in self.figures:
            logger.warning(f"View type {view_type} not created yet")
            return False
        
        try:
            self.figures[view_type].savefig(filename, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved {view_type} to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving {view_type} to {filename}: {str(e)}")
            return False
    
    def close_all(self) -> None:
        """
        Đóng tất cả các hình ảnh MLC.
        """
        for fig in self.figures.values():
            plt.close(fig)
        
        self.figures = {}
        self.axes = {}
        self.artists = {}