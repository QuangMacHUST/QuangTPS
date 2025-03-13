"""
Hiển thị ảnh cơ bản.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import SimpleITK as sitk

# Sửa import từ PyQt5
try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout
    from PyQt5.QtCore import Qt, pyqtSignal
except ImportError:
    # Fallback khi không có GUI
    logging.warning("PyQt5 không được cài đặt. Một số chức năng GUI sẽ không hoạt động.")
    QWidget = object
    QVBoxLayout = object
    QSlider = object
    QLabel = object
    QHBoxLayout = object
    Qt = object
    pyqtSignal = type('pyqtSignal', (), {"__call__": lambda *args, **kwargs: None})

from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class ImageCanvas(FigureCanvas):
    """Canvas hiển thị hình ảnh"""
    
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
        self.image_plot = None
        self.window_center = None
        self.window_width = None
    
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
    
    def update_display(self):
        """Cập nhật hiển thị hình ảnh"""
        if self.image_data is None:
            return
        
        self.axes.clear()
        self.image_plot = self.axes.imshow(
            self.image_data, 
            cmap='gray', 
            vmin=self.window_center - self.window_width / 2, 
            vmax=self.window_center + self.window_width / 2
        )
        self.axes.set_title(f"C: {self.window_center:.1f}, W: {self.window_width:.1f}")
        self.axes.axis('off')  # Tắt trục để hiển thị chỉ hình ảnh
        self.draw()
    
    def update_window(self, center, width):
        """
        Cập nhật window hiển thị.
        
        Parameters:
            center (float): Trung tâm cửa sổ mới
            width (float): Độ rộng cửa sổ mới
        """
        if width <= 0:
            width = 1.0  # Đảm bảo width luôn dương
        
        self.window_center = center
        self.window_width = width
        
        # Cập nhật hiển thị nếu đã có hình ảnh
        if self.image_plot is not None:
            self.image_plot.set_clim(
                self.window_center - self.window_width / 2,
                self.window_center + self.window_width / 2
            )
            self.axes.set_title(f"C: {self.window_center:.1f}, W: {self.window_width:.1f}")
            self.draw()


class ImageViewer(QWidget):
    """Widget hiển thị hình ảnh với các điều khiển"""
    
    # Tín hiệu được phát khi người dùng tương tác
    window_changed = pyqtSignal(float, float)  # center, width
    
    def __init__(self, parent=None):
        """
        Khởi tạo viewer.
        
        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)
        
        # Tạo layout
        self.layout = QVBoxLayout(self)
        
        # Tạo canvas hiển thị hình ảnh
        self.canvas = ImageCanvas(self)
        self.layout.addWidget(self.canvas)
        
        # Tạo điều khiển window
        self.create_window_controls()
        
        # Thông tin hình ảnh
        self.info_label = QLabel("Không có hình ảnh")
        self.layout.addWidget(self.info_label)
        
        # Dữ liệu hình ảnh
        self.image_data = None
        self.metadata = {}
    
    def create_window_controls(self):
        """Tạo các điều khiển cho window"""
        # Layout cho điều khiển
        control_layout = QHBoxLayout()
        
        # Nhãn
        control_layout.addWidget(QLabel("Center:"))
        
        # Slider cho window center
        self.center_slider = QSlider(Qt.Horizontal)
        self.center_slider.setMinimum(-1000)
        self.center_slider.setMaximum(3000)
        self.center_slider.setValue(40)
        self.center_slider.valueChanged.connect(self.on_center_changed)
        control_layout.addWidget(self.center_slider)
        
        # Nhãn
        control_layout.addWidget(QLabel("Width:"))
        
        # Slider cho window width
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setMinimum(1)
        self.width_slider.setMaximum(4000)
        self.width_slider.setValue(400)
        self.width_slider.valueChanged.connect(self.on_width_changed)
        control_layout.addWidget(self.width_slider)
        
        # Thêm layout điều khiển vào layout chính
        self.layout.addLayout(control_layout)
    
    def set_image(self, image_data, metadata=None):
        """
        Đặt dữ liệu hình ảnh và metadata.
        
        Parameters:
            image_data (numpy.ndarray): Dữ liệu hình ảnh
            metadata (dict, optional): Metadata của hình ảnh
        """
        self.image_data = image_data
        
        if metadata is None:
            metadata = {}
        self.metadata = metadata
        
        # Cập nhật range của slider dựa trên dữ liệu hình ảnh
        min_val = image_data.min()
        max_val = image_data.max()
        
        # Thiết lập giá trị mặc định cho window center và width
        if 'window_center' in metadata and 'window_width' in metadata:
            window_center = metadata['window_center']
            window_width = metadata['window_width']
        else:
            window_width = max_val - min_val
            window_center = min_val + window_width / 2
        
        # Cập nhật slider
        self.center_slider.setMinimum(int(min_val))
        self.center_slider.setMaximum(int(max_val))
        self.center_slider.setValue(int(window_center))
        
        self.width_slider.setMinimum(1)
        self.width_slider.setMaximum(int(max_val - min_val) * 2)
        self.width_slider.setValue(int(window_width))
        
        # Cập nhật canvas
        self.canvas.set_image(image_data, window_center, window_width)
        
        # Cập nhật thông tin hình ảnh
        self.update_info_label()
    
    def on_center_changed(self, value):
        """Xử lý khi window center thay đổi"""
        self.canvas.update_window(value, self.width_slider.value())
        self.window_changed.emit(value, self.width_slider.value())
    
    def on_width_changed(self, value):
        """Xử lý khi window width thay đổi"""
        self.canvas.update_window(self.center_slider.value(), value)
        self.window_changed.emit(self.center_slider.value(), value)
    
    def update_info_label(self):
        """Cập nhật nhãn thông tin hình ảnh"""
        if self.image_data is None:
            self.info_label.setText("Không có hình ảnh")
            return
        
        shape = self.image_data.shape
        min_val = self.image_data.min()
        max_val = self.image_data.max()
        
        pixel_size = "N/A"
        if 'pixel_spacing' in self.metadata:
            spacing = self.metadata['pixel_spacing']
            pixel_size = f"{spacing[0]:.2f}mm x {spacing[1]:.2f}mm"
        
        info_text = f"Kích thước: {shape[1]}x{shape[0]}, Range: [{min_val:.1f}, {max_val:.1f}], Pixel size: {pixel_size}"
        self.info_label.setText(info_text)


class MPRViewer(QWidget):
    """Widget hiển thị đa mặt phẳng (MPR - Multi-Planar Reconstruction)"""
    
    def __init__(self, parent=None):
        """
        Khởi tạo MPR viewer.
        
        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)
        
        # Tạo layout chính
        self.layout = QVBoxLayout(self)
        
        # Layout cho 3 view
        self.views_layout = QHBoxLayout()
        
        # Tạo 3 viewer cho 3 mặt phẳng
        self.axial_viewer = ImageViewer()
        self.sagittal_viewer = ImageViewer()
        self.coronal_viewer = ImageViewer()
        
        # Thêm các viewer vào layout
        self.views_layout.addWidget(self.axial_viewer)
        self.views_layout.addWidget(self.sagittal_viewer)
        self.views_layout.addWidget(self.coronal_viewer)
        
        # Thêm layout views vào layout chính
        self.layout.addLayout(self.views_layout)
        
        # Tạo các slider để chọn slice
        self.create_slice_controls()
        
        # Dữ liệu volume
        self.volume = None
        self.sitk_image = None
    
    def create_slice_controls(self):
        """Tạo các điều khiển cho việc chọn slice"""
        # Layout cho các điều khiển
        controls_layout = QHBoxLayout()
        
        # Slider cho mặt phẳng Axial
        axial_layout = QVBoxLayout()
        axial_layout.addWidget(QLabel("Axial:"))
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.valueChanged.connect(self.on_axial_slice_changed)
        axial_layout.addWidget(self.axial_slider)
        controls_layout.addLayout(axial_layout)
        
        # Slider cho mặt phẳng Sagittal
        sagittal_layout = QVBoxLayout()
        sagittal_layout.addWidget(QLabel("Sagittal:"))
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.valueChanged.connect(self.on_sagittal_slice_changed)
        sagittal_layout.addWidget(self.sagittal_slider)
        controls_layout.addLayout(sagittal_layout)
        
        # Slider cho mặt phẳng Coronal
        coronal_layout = QVBoxLayout()
        coronal_layout.addWidget(QLabel("Coronal:"))
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.valueChanged.connect(self.on_coronal_slice_changed)
        coronal_layout.addWidget(self.coronal_slider)
        controls_layout.addLayout(coronal_layout)
        
        # Thêm layout điều khiển vào layout chính
        self.layout.addLayout(controls_layout)
    
    def set_volume(self, volume_data, metadata=None):
        """
        Đặt dữ liệu volume.
        
        Parameters:
            volume_data (numpy.ndarray): Dữ liệu 3D volume
            metadata (dict, optional): Metadata của volume
        """
        if len(volume_data.shape) != 3:
            raise ValidationError("Volume data must be 3D")
        
        self.volume = volume_data
        
        # Cập nhật range của slider
        depth, height, width = volume_data.shape
        
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(depth - 1)
        self.axial_slider.setValue(depth // 2)
        
        self.sagittal_slider.setMinimum(0)
        self.sagittal_slider.setMaximum(width - 1)
        self.sagittal_slider.setValue(width // 2)
        
        self.coronal_slider.setMinimum(0)
        self.coronal_slider.setMaximum(height - 1)
        self.coronal_slider.setValue(height // 2)
        
        # Hiển thị ban đầu
        self.update_all_views()
    
    def set_sitk_image(self, sitk_image):
        """
        Đặt dữ liệu volume từ SimpleITK Image.
        
        Parameters:
            sitk_image (sitk.Image): Ảnh SimpleITK
        """
        self.sitk_image = sitk_image
        
        # Chuyển đổi thành numpy array
        volume_data = sitk.GetArrayFromImage(sitk_image)
        
        # Thêm metadata từ sitk_image
        metadata = {
            'spacing': sitk_image.GetSpacing(),
            'origin': sitk_image.GetOrigin(),
            'direction': sitk_image.GetDirection()
        }
        
        # Đặt volume
        self.set_volume(volume_data, metadata)
    
    def on_axial_slice_changed(self, value):
        """Xử lý khi slice axial thay đổi"""
        self.update_axial_view()
    
    def on_sagittal_slice_changed(self, value):
        """Xử lý khi slice sagittal thay đổi"""
        self.update_sagittal_view()
    
    def on_coronal_slice_changed(self, value):
        """Xử lý khi slice coronal thay đổi"""
        self.update_coronal_view()
    
    def update_axial_view(self):
        """Cập nhật view axial"""
        if self.volume is None:
            return
        
        slice_idx = self.axial_slider.value()
        axial_slice = self.volume[slice_idx, :, :]
        
        self.axial_viewer.set_image(axial_slice)
    
    def update_sagittal_view(self):
        """Cập nhật view sagittal"""
        if self.volume is None:
            return
        
        slice_idx = self.sagittal_slider.value()
        sagittal_slice = self.volume[:, :, slice_idx]
        
        self.sagittal_viewer.set_image(sagittal_slice)
    
    def update_coronal_view(self):
        """Cập nhật view coronal"""
        if self.volume is None:
            return
        
        slice_idx = self.coronal_slider.value()
        coronal_slice = self.volume[:, slice_idx, :]
        
        self.coronal_viewer.set_image(coronal_slice)
    
    def update_all_views(self):
        """Cập nhật tất cả các view"""
        self.update_axial_view()
        self.update_sagittal_view()
        self.update_coronal_view()
        
        
class ImageFusion(QWidget):
    """Widget hiển thị chồng hình đa phương thức"""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget chồng hình.
        
        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)
        
        # Tạo layout
        self.layout = QVBoxLayout(self)
        
        # Tạo canvas hiển thị hình ảnh
        self.canvas = ImageCanvas(self)
        self.layout.addWidget(self.canvas)
        
        # Tạo điều khiển
        self.create_controls()
        
        # Dữ liệu hình ảnh
        self.primary_image = None
        self.secondary_image = None
        self.fusion_alpha = 0.5  # Độ trong suốt của hình thứ hai
        self.colormaps = {
            'gray': plt.cm.gray,
            'hot': plt.cm.hot,
            'jet': plt.cm.jet,
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma
        }
        self.primary_cmap = 'gray'
        self.secondary_cmap = 'hot'
    
    def create_controls(self):
        """Tạo các điều khiển"""
        # Layout cho điều khiển
        control_layout = QVBoxLayout()
        
        # Điều khiển Alpha
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Fusion Alpha:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        alpha_layout.addWidget(self.alpha_slider)
        control_layout.addLayout(alpha_layout)
        
        # Thêm layout điều khiển vào layout chính
        self.layout.addLayout(control_layout)
    
    def set_images(self, primary_image, secondary_image, 
                   primary_window=None, secondary_window=None,
                   primary_cmap='gray', secondary_cmap='hot'):
        """
        Đặt cặp hình ảnh để chồng.
        
        Parameters:
            primary_image (numpy.ndarray): Hình ảnh chính
            secondary_image (numpy.ndarray): Hình ảnh thứ hai
            primary_window (tuple, optional): (center, width) cho hình chính
            secondary_window (tuple, optional): (center, width) cho hình thứ hai
            primary_cmap (str, optional): Colormap cho hình chính
            secondary_cmap (str, optional): Colormap cho hình thứ hai
        
        Raises:
            ValidationError: Nếu kích thước hai hình không khớp
        """
        if primary_image.shape != secondary_image.shape:
            raise ValidationError("Primary and secondary images must have the same shape")
        
        self.primary_image = primary_image
        self.secondary_image = secondary_image
        
        # Lưu các thông số
        self.primary_cmap = primary_cmap
        self.secondary_cmap = secondary_cmap
        
        # Tính window tự động nếu không được chỉ định
        if primary_window is None:
            min_val = primary_image.min()
            max_val = primary_image.max()
            primary_window = (min_val + (max_val - min_val) / 2, max_val - min_val)
        
        if secondary_window is None:
            min_val = secondary_image.min()
            max_val = secondary_image.max()
            secondary_window = (min_val + (max_val - min_val) / 2, max_val - min_val)
        
        self.primary_window = primary_window
        self.secondary_window = secondary_window
        
        # Cập nhật hiển thị
        self.update_fusion()
    
    def on_alpha_changed(self, value):
        """Xử lý khi alpha thay đổi"""
        self.fusion_alpha = value / 100.0
        self.update_fusion()
    
    def update_fusion(self):
        """Cập nhật hiển thị chồng hình"""
        if self.primary_image is None or self.secondary_image is None:
            return
        
        # Lấy colormaps
        primary_cmap = self.colormaps.get(self.primary_cmap, plt.cm.gray)
        secondary_cmap = self.colormaps.get(self.secondary_cmap, plt.cm.hot)
        
        # Chuẩn hóa hình chính
        p_center, p_width = self.primary_window
        p_min = p_center - p_width / 2
        p_max = p_center + p_width / 2
        primary_norm = np.clip((self.primary_image - p_min) / (p_max - p_min), 0, 1)
        
        # Chuẩn hóa hình thứ hai
        s_center, s_width = self.secondary_window
        s_min = s_center - s_width / 2
        s_max = s_center + s_width / 2
        secondary_norm = np.clip((self.secondary_image - s_min) / (s_max - s_min), 0, 1)
        
        # Chuyển đổi thành RGB
        primary_rgb = primary_cmap(primary_norm)
        secondary_rgb = secondary_cmap(secondary_norm)
        
        # Trộn hai hình
        fusion = (1 - self.fusion_alpha) * primary_rgb + self.fusion_alpha * secondary_rgb
        
        # Hiển thị
        self.canvas.axes.clear()
        self.canvas.axes.imshow(fusion)
        self.canvas.axes.axis('off')
        self.canvas.axes.set_title(f"Fusion Alpha: {self.fusion_alpha:.2f}")
        self.canvas.draw()
