#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module trực quan hóa kỹ thuật xạ trị 3D CRT (3D Conformal Radiation Therapy).

Module này cung cấp các lớp và hàm để hiển thị trực quan các thành phần
của kỹ thuật xạ trị 3D CRT, bao gồm hình dạng chùm tia, MLC, wedge và
các thành phần khác.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Tuple, Any, Optional, Union

try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout
    from PyQt5.QtCore import Qt, pyqtSignal
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.planning.mlc import MLC
from quangtps.treatment.beams.beam_modifiers import Wedge, Block

logger = logging.getLogger(__name__)

class CRTVisualizer:
    """
    Lớp trực quan hóa kỹ thuật xạ trị 3D CRT.
    
    Lớp này cung cấp các phương thức để hiển thị trực quan các thành phần
    của kỹ thuật xạ trị 3D CRT, bao gồm chùm tia, MLC, wedge và các
    thành phần khác.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng trực quan hóa 3D CRT."""
        # Kích thước mặc định của hình vẽ (inch)
        self.figure_size = (10, 8)
        
        # Màu sắc mặc định
        self.colors = {
            'beam': 'yellow',
            'mlc': 'gray',
            'wedge': 'orange',
            'block': 'red',
            'isocenter': 'blue',
            'field': 'green',
            'body': 'lightblue',
            'target': 'red',
            'oar': 'purple'
        }
        
        # Kiểu đường
        self.line_styles = {
            'beam': '-',
            'mlc': '--',
            'field': '-.',
            'block': '-'
        }
        
        # Mật độ của đường fill
        self.alpha = {
            'beam': 0.3,
            'mlc': 0.7,
            'wedge': 0.5,
            'block': 0.6,
            'field': 0.2,
            'body': 0.3,
            'target': 0.5,
            'oar': 0.4
        }
    
    def visualize_beam_eye_view(self, beam: Beam, figure: Optional[Figure] = None,
                               show_mlc: bool = True, show_blocks: bool = True,
                               show_wedge: bool = True) -> Figure:
        """
        Hiển thị góc nhìn từ chùm tia (Beam's Eye View - BEV).
        
        Parameters
        ----------
        beam : Beam
            Chùm tia cần hiển thị
        figure : matplotlib.figure.Figure, optional
            Đối tượng Figure để vẽ lên. Nếu không cung cấp, sẽ tạo figure mới.
        show_mlc : bool, optional
            Hiển thị MLC (nếu có)
        show_blocks : bool, optional
            Hiển thị blocks (nếu có)
        show_wedge : bool, optional
            Hiển thị wedge (nếu có)
            
        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure chứa góc nhìn BEV
        """
        # Tạo figure mới nếu cần
        if figure is None:
            figure = plt.figure(figsize=self.figure_size)
            ax = figure.add_subplot(111)
        else:
            ax = figure.gca()
        
        # Lấy kích thước trường
        field_width = beam.field_size[0] if hasattr(beam, 'field_size') else 10.0
        field_height = beam.field_size[1] if hasattr(beam, 'field_size') else 10.0
        
        # Hiển thị kích thước trường
        field_rect = Rectangle(
            (-field_width/2, -field_height/2),
            field_width, field_height,
            linewidth=2,
            edgecolor=self.colors['field'],
            facecolor=self.colors['field'],
            alpha=self.alpha['field']
        )
        ax.add_patch(field_rect)
        
        # Hiển thị tâm chùm tia (isocenter)
        ax.plot(0, 0, 'o', color=self.colors['isocenter'], markersize=10)
        
        # Hiển thị MLC nếu cần
        if show_mlc and hasattr(beam, 'mlc') and beam.mlc is not None:
            self._draw_mlc(ax, beam.mlc, field_width, field_height)
        
        # Hiển thị block nếu cần
        if show_blocks and hasattr(beam, 'blocks') and beam.blocks:
            for block in beam.blocks:
                self._draw_block(ax, block)
        
        # Hiển thị wedge nếu cần
        if show_wedge and hasattr(beam, 'wedges') and beam.wedges:
            for wedge in beam.wedges:
                self._draw_wedge(ax, wedge, field_width, field_height)
        
        # Thiết lập các thuộc tính của trục
        ax.set_xlim(-field_width, field_width)
        ax.set_ylim(-field_height, field_height)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f"Beam's Eye View: {beam.name}")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        
        # Thêm chỉ dẫn hướng
        ax.text(field_width * 0.8, field_height * 0.8, 'Superior', 
                ha='center', va='center', fontsize=10)
        ax.text(field_width * 0.8, -field_height * 0.8, 'Inferior', 
                ha='center', va='center', fontsize=10)
        ax.text(-field_width * 0.8, field_height * 0.8, 'Superior', 
                ha='center', va='center', fontsize=10)
        ax.text(-field_width * 0.8, -field_height * 0.8, 'Inferior', 
                ha='center', va='center', fontsize=10)
        
        return figure
    
    def _draw_mlc(self, ax, mlc: MLC, field_width: float, field_height: float) -> None:
        """
        Vẽ MLC lên trục.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Trục để vẽ
        mlc : MLC
            Đối tượng MLC cần vẽ
        field_width : float
            Chiều rộng trường
        field_height : float
            Chiều cao trường
        """
        try:
            # Kiểm tra xem MLC có dữ liệu leaf positions không
            if not hasattr(mlc, 'leaf_positions') or not mlc.leaf_positions:
                logger.warning("MLC không có dữ liệu về vị trí của các lá")
                return
            
            # Lấy số lượng lá
            num_leaves = len(mlc.leaf_positions) // 2
            
            # Tính chiều cao của mỗi lá
            leaf_height = field_height / num_leaves
            
            # Vẽ các lá MLC
            for i in range(num_leaves):
                # Lá bên trái
                left_pos = mlc.leaf_positions[i]
                left_rect = Rectangle(
                    (-field_width/2, -field_height/2 + i * leaf_height),
                    field_width/2 + left_pos,
                    leaf_height,
                    linewidth=1,
                    edgecolor=self.colors['mlc'],
                    facecolor=self.colors['mlc'],
                    alpha=self.alpha['mlc']
                )
                ax.add_patch(left_rect)
                
                # Lá bên phải
                right_pos = mlc.leaf_positions[i + num_leaves]
                right_rect = Rectangle(
                    (right_pos, -field_height/2 + i * leaf_height),
                    field_width/2 - right_pos,
                    leaf_height,
                    linewidth=1,
                    edgecolor=self.colors['mlc'],
                    facecolor=self.colors['mlc'],
                    alpha=self.alpha['mlc']
                )
                ax.add_patch(right_rect)
        except Exception as e:
            logger.error(f"Lỗi khi vẽ MLC: {e}")
    
    def _draw_block(self, ax, block: Block) -> None:
        """
        Vẽ block lên trục.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Trục để vẽ
        block : Block
            Đối tượng Block cần vẽ
        """
        try:
            if hasattr(block, 'contour') and block.contour:
                # Tạo polygon từ contour
                polygon = Polygon(
                    block.contour,
                    closed=True,
                    linewidth=2,
                    edgecolor=self.colors['block'],
                    facecolor=self.colors['block'],
                    alpha=self.alpha['block']
                )
                ax.add_patch(polygon)
                
                # Thêm nhãn cho block
                if hasattr(block, 'name') and block.name:
                    centroid = np.mean(block.contour, axis=0)
                    ax.text(centroid[0], centroid[1], block.name,
                           ha='center', va='center', fontsize=8)
        except Exception as e:
            logger.error(f"Lỗi khi vẽ block: {e}")
    
    def _draw_wedge(self, ax, wedge: Wedge, field_width: float, field_height: float) -> None:
        """
        Vẽ wedge lên trục.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Trục để vẽ
        wedge : Wedge
            Đối tượng Wedge cần vẽ
        field_width : float
            Chiều rộng trường
        field_height : float
            Chiều cao trường
        """
        try:
            if hasattr(wedge, 'angle') and hasattr(wedge, 'orientation'):
                # Xác định hướng wedge
                is_horizontal = wedge.orientation in ['IN', 'OUT']
                
                # Tạo các điểm cho hình tam giác biểu diễn wedge
                if is_horizontal:
                    if wedge.orientation == 'IN':
                        # Wedge dày ở phía trong (isocenter)
                        points = [
                            (-field_width/2, -field_height/2),
                            (-field_width/2, field_height/2),
                            (field_width/2, 0)
                        ]
                    else:
                        # Wedge dày ở phía ngoài
                        points = [
                            (field_width/2, -field_height/2),
                            (field_width/2, field_height/2),
                            (-field_width/2, 0)
                        ]
                else:  # Vertical
                    if wedge.orientation == 'LEFT':
                        # Wedge dày ở bên trái
                        points = [
                            (-field_width/2, -field_height/2),
                            (field_width/2, -field_height/2),
                            (0, field_height/2)
                        ]
                    else:  # 'RIGHT'
                        # Wedge dày ở bên phải
                        points = [
                            (-field_width/2, field_height/2),
                            (field_width/2, field_height/2),
                            (0, -field_height/2)
                        ]
                
                # Vẽ wedge
                polygon = Polygon(
                    points,
                    closed=True,
                    linewidth=2,
                    edgecolor=self.colors['wedge'],
                    facecolor=self.colors['wedge'],
                    alpha=self.alpha['wedge']
                )
                ax.add_patch(polygon)
                
                # Thêm nhãn cho wedge
                ax.text(0, 0, f"{wedge.angle}°",
                       ha='center', va='center', fontsize=10,
                       weight='bold')
        except Exception as e:
            logger.error(f"Lỗi khi vẽ wedge: {e}")
    
    def visualize_3d_beams(self, plan: Plan, figure: Optional[Figure] = None,
                         show_patient: bool = True, show_target: bool = True) -> Figure:
        """
        Hiển thị chùm tia 3D trong không gian.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị cần hiển thị
        figure : matplotlib.figure.Figure, optional
            Đối tượng Figure để vẽ lên. Nếu không cung cấp, sẽ tạo figure mới.
        show_patient : bool, optional
            Hiển thị đường viền cơ thể bệnh nhân
        show_target : bool, optional
            Hiển thị mục tiêu (PTV)
            
        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure chứa hình ảnh 3D
        """
        # Tạo figure mới nếu cần
        if figure is None:
            figure = plt.figure(figsize=self.figure_size)
            ax = figure.add_subplot(111, projection='3d')
        else:
            if len(figure.axes) == 0 or not isinstance(figure.axes[0], Axes3D):
                figure.clear()
                ax = figure.add_subplot(111, projection='3d')
            else:
                ax = figure.axes[0]
                ax.clear()
        
        # Hiển thị isocenter
        if hasattr(plan, 'isocenter') and plan.isocenter:
            ax.scatter(
                plan.isocenter[0], 
                plan.isocenter[1], 
                plan.isocenter[2],
                s=100, 
                color=self.colors['isocenter'], 
                marker='o', 
                label='Isocenter'
            )
        
        # Hiển thị chùm tia
        if hasattr(plan, 'beams') and plan.beams:
            for i, beam in enumerate(plan.beams):
                self._draw_3d_beam(ax, beam, i)
        
        # Hiển thị đường viền cơ thể bệnh nhân nếu cần
        if show_patient and hasattr(plan, 'body_contour') and plan.body_contour is not None:
            self._draw_3d_contour(ax, plan.body_contour, self.colors['body'], self.alpha['body'], 'Body')
        
        # Hiển thị mục tiêu nếu cần
        if show_target and hasattr(plan, 'target_contour') and plan.target_contour is not None:
            self._draw_3d_contour(ax, plan.target_contour, self.colors['target'], self.alpha['target'], 'Target')
        
        # Thiết lập các thuộc tính của trục
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title('3D Conformal Radiation Therapy Plan')
        
        # Thêm chú thích
        ax.legend()
        
        return figure
    
    def _draw_3d_beam(self, ax, beam: Beam, index: int) -> None:
        """
        Vẽ chùm tia trong không gian 3D.
        
        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D
            Trục 3D để vẽ
        beam : Beam
            Chùm tia cần vẽ
        index : int
            Chỉ số của chùm tia (dùng để tạo màu khác nhau)
        """
        try:
            # Lấy thông tin về chùm tia
            source_position = getattr(beam, 'source_position', [0, 0, 100])
            isocenter = getattr(beam, 'isocenter', [0, 0, 0])
            
            # Vẽ đường từ nguồn đến isocenter
            ax.plot(
                [source_position[0], isocenter[0]],
                [source_position[1], isocenter[1]],
                [source_position[2], isocenter[2]],
                color=self.colors['beam'],
                linestyle=self.line_styles['beam'],
                label=f"Beam: {beam.name}" if index == 0 else ""
            )
            
            # Vẽ hình nón của chùm tia (đơn giản hóa)
            # TODO: Cải thiện hình dạng chùm tia dựa trên kích thước trường và góc collimator
            
            # Vẽ nhãn cho chùm tia
            if hasattr(beam, 'name'):
                ax.text(
                    source_position[0], 
                    source_position[1], 
                    source_position[2],
                    beam.name,
                    fontsize=8
                )
        except Exception as e:
            logger.error(f"Lỗi khi vẽ chùm tia 3D: {e}")
    
    def _draw_3d_contour(self, ax, contour: Any, color: str, alpha: float, label: str) -> None:
        """
        Vẽ đường viền 3D.
        
        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D
            Trục 3D để vẽ
        contour : Any
            Đường viền cần vẽ
        color : str
            Màu của đường viền
        alpha : float
            Độ trong suốt
        label : str
            Nhãn cho đường viền
        """
        # Phương thức này cần được cài đặt dựa trên cấu trúc dữ liệu contour cụ thể
        # Đây là một phiên bản đơn giản
        try:
            if isinstance(contour, list) and len(contour) > 0:
                for poly in contour:
                    collection = Poly3DCollection([poly], alpha=alpha)
                    collection.set_facecolor(color)
                    collection.set_edgecolor('black')
                    ax.add_collection3d(collection)
        except Exception as e:
            logger.error(f"Lỗi khi vẽ đường viền 3D: {e}")
    
    def visualize_dose_3d(self, plan: Plan, dose_data: np.ndarray,
                        figure: Optional[Figure] = None) -> Figure:
        """
        Hiển thị phân bố liều 3D.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị
        dose_data : np.ndarray
            Dữ liệu phân bố liều 3D
        figure : matplotlib.figure.Figure, optional
            Đối tượng Figure để vẽ lên. Nếu không cung cấp, sẽ tạo figure mới.
            
        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure chứa hình ảnh phân bố liều 3D
        """
        # Tạo figure mới nếu cần
        if figure is None:
            figure = plt.figure(figsize=self.figure_size)
            ax = figure.add_subplot(111, projection='3d')
        else:
            if len(figure.axes) == 0 or not isinstance(figure.axes[0], Axes3D):
                figure.clear()
                ax = figure.add_subplot(111, projection='3d')
            else:
                ax = figure.axes[0]
                ax.clear()
        
        # Tính toán giá trị max và min của liều
        dose_max = np.max(dose_data)
        dose_min = np.min(dose_data)
        
        # Xác định các mức isodose (ví dụ: 95%, 80%, 50%, 20% của liều max)
        isodose_levels = [0.95, 0.8, 0.5, 0.2]
        isodose_values = [level * dose_max for level in isodose_levels]
        isodose_colors = ['red', 'orange', 'yellow', 'green']
        
        # TODO: Thêm mã để hiển thị các bề mặt isodose 3D
        # Đây là một tác vụ phức tạp và cần cài đặt thuật toán marching cubes
        # và xử lý dữ liệu liều 3D chi tiết
        
        # Thiết lập các thuộc tính của trục
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title('3D Dose Distribution')
        
        # Thêm chú thích cho các mức isodose
        for level, color in zip(isodose_levels, isodose_colors):
            ax.plot([], [], [], color=color, label=f"{int(level*100)}% ({level*dose_max:.1f} Gy)")
        
        ax.legend()
        
        return figure


if QT_AVAILABLE:
    class CRTVisualizerWidget(QWidget):
        """
        Widget trực quan hóa kỹ thuật xạ trị 3D CRT.
        
        Widget này cung cấp giao diện PyQt để hiển thị trực quan các thành phần
        của kỹ thuật xạ trị 3D CRT.
        """
        
        beam_selected = pyqtSignal(Beam)
        
        def __init__(self, parent=None):
            """
            Khởi tạo widget trực quan hóa 3D CRT.
            
            Parameters
            ----------
            parent : QWidget, optional
                Widget cha
            """
            super().__init__(parent)
            
            self.visualizer = CRTVisualizer()
            self.plan = None
            self.beams = []
            self.current_beam_index = -1
            
            self._init_ui()
        
        def _init_ui(self):
            """Khởi tạo giao diện người dùng."""
            layout = QVBoxLayout(self)
            
            # Tạo figure và canvas
            self.figure = plt.figure(figsize=self.visualizer.figure_size)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            
            # Thêm toolbar
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            # Thêm vào layout
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
        
        def set_plan(self, plan: Plan):
            """
            Đặt kế hoạch điều trị cần hiển thị.
            
            Parameters
            ----------
            plan : Plan
                Kế hoạch điều trị
            """
            self.plan = plan
            
            if hasattr(plan, 'beams'):
                self.beams = plan.beams
                self.current_beam_index = 0 if self.beams else -1
                
                if self.current_beam_index >= 0:
                    self.visualize_current_beam()
        
        def visualize_current_beam(self):
            """Hiển thị chùm tia hiện tại."""
            if not self.beams or self.current_beam_index < 0:
                return
            
            # Xóa figure
            self.figure.clear()
            
            # Lấy chùm tia hiện tại
            beam = self.beams[self.current_beam_index]
            
            # Hiển thị BEV
            self.visualizer.visualize_beam_eye_view(beam, self.figure)
            
            # Cập nhật canvas
            self.canvas.draw()
        
        def visualize_3d(self):
            """Hiển thị góc nhìn 3D của kế hoạch điều trị."""
            if not self.plan:
                return
            
            # Xóa figure
            self.figure.clear()
            
            # Hiển thị cấu hình 3D
            self.visualizer.visualize_3d_beams(self.plan, self.figure)
            
            # Cập nhật canvas
            self.canvas.draw()
        
        def next_beam(self):
            """Chuyển đến chùm tia tiếp theo."""
            if not self.beams:
                return
            
            self.current_beam_index = (self.current_beam_index + 1) % len(self.beams)
            self.visualize_current_beam()
            
            # Phát tín hiệu chùm tia đã chọn
            self.beam_selected.emit(self.beams[self.current_beam_index])
        
        def previous_beam(self):
            """Chuyển đến chùm tia trước đó."""
            if not self.beams:
                return
            
            self.current_beam_index = (self.current_beam_index - 1) % len(self.beams)
            self.visualize_current_beam()
            
            # Phát tín hiệu chùm tia đã chọn
            self.beam_selected.emit(self.beams[self.current_beam_index])


if __name__ == "__main__":
    # Ví dụ sử dụng trực quan hóa 3D CRT
    
    if QT_AVAILABLE:
        import sys
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        
        window = CRTVisualizerWidget()
        window.setWindowTitle("3D CRT Visualizer")
        window.resize(800, 600)
        window.show()
        
        sys.exit(app.exec_())
    else:
        # Ví dụ không sử dụng PyQt
        visualizer = CRTVisualizer()
        
        # Tạo chùm tia mẫu
        class DummyBeam:
            def __init__(self):
                self.name = "Test Beam"
                self.field_size = (10, 10)
                self.mlc = None
                self.blocks = []
                self.wedges = []
        
        beam = DummyBeam()
        
        # Hiển thị BEV
        fig = visualizer.visualize_beam_eye_view(beam)
        plt.show() 