"""
Giao diện chỉnh sửa contour tương tác.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import logging

# Sử dụng try-except để đảm bảo code hoạt động cả khi không có GUI
try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSlider
    from PyQt5.QtCore import Qt, pyqtSignal
except ImportError:
    # Fallback khi không có GUI
    logging.warning("PyQt5 không được cài đặt. Một số chức năng GUI sẽ không hoạt động.")
    QWidget = object
    QVBoxLayout = object
    QHBoxLayout = object
    QLabel = object
    QPushButton = object
    QComboBox = object
    QSlider = object
    Qt = object
    pyqtSignal = type('pyqtSignal', (), {"__call__": lambda *args, **kwargs: None})

from quangtps.segmentation.contour.contour_tools import ContourTool, BrushTool, ThresholdTool, RegionGrowingTool, WatershedTool

logger = logging.getLogger(__name__)

class ContourCanvas(FigureCanvas):
    """Canvas hiển thị hình ảnh và contour"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Khởi tạo canvas.
        
        Parameters:
            parent (QWidget, optional): Widget cha
            width (int, optional): Chiều rộng của figure
            height (int, optional): Chiều cao của figure
            dpi (int, optional): DPI của figure
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        super().__init__(fig)
        self.setParent(parent)
        
        self.image_data = None
        self.contour_tool = ContourTool()
        self.window_center = None
        self.window_width = None
        
        # Kết nối sự kiện chuột
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Trạng thái chuột
        self.mouse_pressed = False
        self.last_point = None
        self.mode = 'draw'  # 'draw', 'move', 'erase'
        
        # Điểm đang kéo
        self.dragging_point_index = None
    
    def set_image(self, image_data, window_center=None, window_width=None):
        """
        Đặt dữ liệu hình ảnh.
        
        Parameters:
            image_data (numpy.ndarray): Dữ liệu hình ảnh
            window_center (float, optional): Trung tâm cửa sổ hiển thị
            window_width (float, optional): Độ rộng của cửa sổ hiển thị
        """
        self.image_data = image_data
        
        # Tính window tự động nếu không được chỉ định
        if window_center is None or window_width is None:
            min_val = image_data.min()
            max_val = image_data.max()
            window_width = max_val - min_val
            window_center = min_val + window_width / 2
        
        self.window_center = window_center
        self.window_width = window_width
        
        self.update_display()
    
    def set_contour_tool(self, tool):
        """
        Đặt công cụ contour.
        
        Parameters:
            tool (ContourTool): Công cụ contour
        """
        self.contour_tool = tool
        self.update_display()
    
    def set_mode(self, mode):
        """
        Đặt chế độ hoạt động.
        
        Parameters:
            mode (str): Chế độ ('draw', 'move', 'erase')
        """
        self.mode = mode
    
    def update_display(self):
        """Cập nhật hiển thị hình ảnh và contour"""
        if self.image_data is None:
            return
        
        self.axes.clear()
        
        # Hiển thị hình ảnh
        self.axes.imshow(
            self.image_data, 
            cmap='gray', 
            vmin=self.window_center - self.window_width / 2, 
            vmax=self.window_center + self.window_width / 2
        )
        
        # Hiển thị contour
        if self.contour_tool:
            self.contour_tool.draw_contours(self.axes)
        
        self.axes.set_title(f"C: {self.window_center:.1f}, W: {self.window_width:.1f}")
        self.axes.axis('off')  # Tắt trục để hiển thị chỉ hình ảnh
        self.draw()
    
    def on_mouse_press(self, event):
        """Xử lý sự kiện nhấn chuột"""
        if event.inaxes != self.axes or self.image_data is None:
            return
        
        self.mouse_pressed = True
        x, y = event.xdata, event.ydata
        
        if self.mode == 'draw':
            # Nếu đang sử dụng BrushTool
            if isinstance(self.contour_tool, BrushTool):
                self.contour_tool.draw((x, y))
            else:
                # Nếu đang sử dụng ContourTool thông thường
                self.contour_tool.add_point((x, y))
        
        elif self.mode == 'move':
            # Tìm điểm gần nhất để kéo
            if self.contour_tool.active_contour is not None:
                contour = self.contour_tool.get_contour_points()
                if contour:
                    distances = [(i, (p[0] - x)**2 + (p[1] - y)**2) for i, p in enumerate(contour)]
                    idx, dist = min(distances, key=lambda x: x[1])
                    
                    # Chỉ kéo nếu đủ gần
                    if dist < 100:  # Ngưỡng khoảng cách (tùy chỉnh)
                        self.dragging_point_index = idx
        
        elif self.mode == 'erase':
            # Nếu đang sử dụng BrushTool
            if isinstance(self.contour_tool, BrushTool):
                self.contour_tool.erase((x, y))
        
        self.last_point = (x, y)
        self.update_display()
    
    def on_mouse_release(self, event):
        """Xử lý sự kiện thả chuột"""
        self.mouse_pressed = False
        self.dragging_point_index = None
    
    def on_mouse_move(self, event):
        """Xử lý sự kiện di chuyển chuột"""
        if not self.mouse_pressed or event.inaxes != self.axes or self.image_data is None:
            return
        
        x, y = event.xdata, event.ydata
        
        if self.mode == 'draw':
            # Nếu đang sử dụng BrushTool
            if isinstance(self.contour_tool, BrushTool):
                self.contour_tool.draw((x, y))
            else:
                # Đối với ContourTool, chỉ thêm điểm nếu di chuyển đủ xa
                if self.last_point:
                    dx = x - self.last_point[0]
                    dy = y - self.last_point[1]
                    dist = dx*dx + dy*dy
                    
                    if dist > 25:  # Ngưỡng khoảng cách (tùy chỉnh)
                        self.contour_tool.add_point((x, y))
                        self.last_point = (x, y)
        
        elif self.mode == 'move':
            # Di chuyển điểm đang kéo
            if self.dragging_point_index is not None:
                self.contour_tool.move_point(self.dragging_point_index, (x, y))
        
        elif self.mode == 'erase':
            # Nếu đang sử dụng BrushTool
            if isinstance(self.contour_tool, BrushTool):
                self.contour_tool.erase((x, y))
        
        self.update_display()


class ContourEditor(QWidget):
    """Widget chỉnh sửa contour"""
    
    # Tín hiệu phát ra khi contour thay đổi
    contour_changed = pyqtSignal(object)  # Phát ra contour tool
    
    def __init__(self, parent=None):
        """
        Khởi tạo editor.
        
        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)
        
        # Tạo layout
        self.layout = QVBoxLayout(self)
        
        # Tạo canvas hiển thị hình ảnh và contour
        self.canvas = ContourCanvas(self)
        self.layout.addWidget(self.canvas)
        
        # Tạo điều khiển
        self.create_controls()
        
        # Công cụ contour
        self.contour_tools = {
            'manual': ContourTool(),
            'brush': BrushTool(),
            'threshold': ThresholdTool(),
            'region_growing': RegionGrowingTool(),
            'watershed': WatershedTool()
        }
        
        self.current_tool_name = 'manual'
        self.canvas.set_contour_tool(self.contour_tools[self.current_tool_name])
    
    def create_controls(self):
        """Tạo các điều khiển"""
        # Layout cho điều khiển công cụ
        tool_layout = QHBoxLayout()
        
        # Combo box chọn công cụ
        tool_layout.addWidget(QLabel("Công cụ:"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(['Manual Contour', 'Brush', 'Threshold', 'Region Growing', 'Watershed'])
        self.tool_combo.currentIndexChanged.connect(self.on_tool_changed)
        tool_layout.addWidget(self.tool_combo)
        
        # Combo box chọn chế độ
        tool_layout.addWidget(QLabel("Chế độ:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Draw', 'Move', 'Erase'])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        tool_layout.addWidget(self.mode_combo)
        
        # Nút đóng contour
        self.close_button = QPushButton("Đóng Contour")
        self.close_button.clicked.connect(self.on_close_contour)
        tool_layout.addWidget(self.close_button)
        
        # Nút xóa contour
        self.clear_button = QPushButton("Xóa Contour")
        self.clear_button.clicked.connect(self.on_clear_contour)
        tool_layout.addWidget(self.clear_button)
        
        # Thêm layout điều khiển vào layout chính
        self.layout.addLayout(tool_layout)
        
        # Layout cho điều khiển brush
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Kích thước bút:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(20)
        self.brush_slider.setValue(5)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        brush_layout.addWidget(self.brush_slider)
        self.layout.addLayout(brush_layout)
        
        # Layout cho điều khiển threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Ngưỡng min:"))
        self.threshold_min_slider = QSlider(Qt.Horizontal)
        self.threshold_min_slider.setMinimum(-1000)
        self.threshold_min_slider.setMaximum(3000)
        self.threshold_min_slider.setValue(0)
        self.threshold_min_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_min_slider)
        
        threshold_layout.addWidget(QLabel("Ngưỡng max:"))
        self.threshold_max_slider = QSlider(Qt.Horizontal)
        self.threshold_max_slider.setMinimum(-1000)
        self.threshold_max_slider.setMaximum(3000)
        self.threshold_max_slider.setValue(100)
        self.threshold_max_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_max_slider)
        
        self.layout.addLayout(threshold_layout)
        
        # Layout cho điều khiển region growing
        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("Tolerance:"))
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(1)
        self.tolerance_slider.setMaximum(100)
        self.tolerance_slider.setValue(10)
        self.tolerance_slider.valueChanged.connect(self.on_tolerance_changed)
        region_layout.addWidget(self.tolerance_slider)
        self.layout.addLayout(region_layout)
        
        # Ẩn các điều khiển không cần thiết ban đầu
        self.brush_slider.setVisible(False)
        self.threshold_min_slider.setVisible(False)
        self.threshold_max_slider.setVisible(False)
        self.tolerance_slider.setVisible(False)
    
    def set_image(self, image_data, window_center=None, window_width=None):
        """
        Đặt dữ liệu hình ảnh.
        
        Parameters:
            image_data (numpy.ndarray): Dữ liệu hình ảnh
            window_center (float, optional): Trung tâm cửa sổ hiển thị
            window_width (float, optional): Độ rộng của cửa sổ hiển thị
        """
        self.canvas.set_image(image_data, window_center, window_width)
        
        # Khởi tạo công cụ brush và watershed
        if isinstance(image_data, np.ndarray):
            shape = image_data.shape
            if len(shape) == 2:  # Hình ảnh 2D
                self.contour_tools['brush'].initialize(shape)
                self.contour_tools['watershed'].initialize(shape)
    
    def on_tool_changed(self, index):
        """Xử lý khi công cụ thay đổi"""
        tool_map = {
            0: 'manual',
            1: 'brush',
            2: 'threshold',
            3: 'region_growing',
            4: 'watershed'
        }
        
        self.current_tool_name = tool_map[index]
        self.canvas.set_contour_tool(self.contour_tools[self.current_tool_name])
        
        # Hiển thị/ẩn các điều khiển phù hợp
        self.brush_slider.setVisible(self.current_tool_name == 'brush')
        self.threshold_min_slider.setVisible(self.current_tool_name == 'threshold')
        self.threshold_max_slider.setVisible(self.current_tool_name == 'threshold')
        self.tolerance_slider.setVisible(self.current_tool_name == 'region_growing')
        
        # Cập nhật chế độ mặc định cho công cụ
        if self.current_tool_name == 'brush':
            self.canvas.set_mode('draw')
            self.mode_combo.setCurrentIndex(0)
        elif self.current_tool_name in ['threshold', 'region_growing', 'watershed']:
            self.canvas.set_mode('draw')
            self.mode_combo.setCurrentIndex(0)
        
        # Phát tín hiệu contour thay đổi
        self.contour_changed.emit(self.contour_tools[self.current_tool_name])
    
    def on_mode_changed(self, index):
        """Xử lý khi chế độ thay đổi"""
        mode_map = {
            0: 'draw',
            1: 'move',
            2: 'erase'
        }
        
        self.canvas.set_mode(mode_map[index])
    
    def on_brush_size_changed(self, value):
        """Xử lý khi kích thước bút thay đổi"""
        if self.current_tool_name == 'brush':
            self.contour_tools['brush'].set_brush_size(value)
    
    def on_threshold_changed(self, value):
        """Xử lý khi ngưỡng thay đổi"""
        if self.current_tool_name == 'threshold':
            min_val = self.threshold_min_slider.value()
            max_val = self.threshold_max_slider.value()
            self.contour_tools['threshold'].set_threshold(min_val, max_val)
    
    def on_tolerance_changed(self, value):
        """Xử lý khi tolerance thay đổi"""
        if self.current_tool_name == 'region_growing':
            self.contour_tools['region_growing'].set_tolerance(value)
    
    def on_close_contour(self):
        """Xử lý khi nhấn nút đóng contour"""
        if self.current_tool_name == 'manual':
            self.contour_tools['manual'].close_contour()
            self.canvas.update_display()
            
            # Phát tín hiệu contour thay đổi
            self.contour_changed.emit(self.contour_tools[self.current_tool_name])
    
    def on_clear_contour(self):
        """Xử lý khi nhấn nút xóa contour"""
        if self.current_tool_name == 'manual':
            self.contour_tools['manual'].clear_contours()
        elif self.current_tool_name == 'brush':
            self.contour_tools['brush'].clear()
        
        self.canvas.update_display()
        
        # Phát tín hiệu contour thay đổi
        self.contour_changed.emit(self.contour_tools[self.current_tool_name])
    
    def get_contours(self):
        """
        Lấy contours hiện tại.
        
        Returns:
            list: Danh sách các contour
        """
        if self.current_tool_name == 'manual':
            return self.contour_tools['manual'].get_all_contours()
        elif self.current_tool_name == 'brush':
            return self.contour_tools['brush'].extract_contours()
        
        return []
    
    def get_mask(self):
        """
        Lấy mask từ contour hiện tại.
        
        Returns:
            numpy.ndarray: Mask nhị phân
        """
        if self.current_tool_name == 'brush':
            return self.contour_tools['brush'].get_mask()
        
        return None
    
    def set_contours(self, contours):
        """
        Đặt contours.
        
        Parameters:
            contours (list): Danh sách các contour
        """
        # Chuyển sang công cụ manual
        self.tool_combo.setCurrentIndex(0)
        self.current_tool_name = 'manual'
        
        # Xóa contours hiện tại
        self.contour_tools['manual'].clear_contours()
        
        # Thêm contours mới
        for contour in contours:
            self.contour_tools['manual'].contours.append(contour)
        
        if contours:
            self.contour_tools['manual'].active_contour = 0
        
        self.canvas.update_display()
