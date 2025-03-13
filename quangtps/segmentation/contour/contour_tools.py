"""
Công cụ tạo và chỉnh sửa contour.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
import SimpleITK as sitk
import logging
from enum import Enum

from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class ContourType(Enum):
    """Các loại contour"""
    MANUAL = 1
    AUTO = 2
    INTERPOLATED = 3
    BOOLEAN = 4

class ContourTool:
    """Lớp cơ sở cho các công cụ contour"""
    
    def __init__(self):
        """Khởi tạo công cụ contour"""
        self.contours = []
        self.active_contour = None
        self.contour_type = ContourType.MANUAL
        self.contour_colors = {}
        
    def set_active_contour(self, contour_index):
        """
        Đặt contour hiện tại.
        
        Parameters:
            contour_index (int): Chỉ số của contour
        """
        if 0 <= contour_index < len(self.contours):
            self.active_contour = contour_index
        else:
            self.active_contour = None
    
    def add_point(self, point):
        """
        Thêm điểm vào contour hiện tại.
        
        Parameters:
            point (tuple): Tọa độ điểm (x, y)
        
        Returns:
            bool: True nếu thêm thành công
        """
        if self.active_contour is None:
            # Tạo contour mới
            self.contours.append([])
            self.active_contour = len(self.contours) - 1
        
        self.contours[self.active_contour].append(point)
        return True
    
    def remove_point(self, index):
        """
        Xóa điểm khỏi contour hiện tại.
        
        Parameters:
            index (int): Chỉ số của điểm
        
        Returns:
            bool: True nếu xóa thành công
        """
        if self.active_contour is None:
            return False
        
        if 0 <= index < len(self.contours[self.active_contour]):
            del self.contours[self.active_contour][index]
            return True
        
        return False
    
    def move_point(self, index, new_point):
        """
        Di chuyển điểm trong contour hiện tại.
        
        Parameters:
            index (int): Chỉ số của điểm
            new_point (tuple): Tọa độ mới (x, y)
        
        Returns:
            bool: True nếu di chuyển thành công
        """
        if self.active_contour is None:
            return False
        
        if 0 <= index < len(self.contours[self.active_contour]):
            self.contours[self.active_contour][index] = new_point
            return True
        
        return False
    
    def close_contour(self):
        """
        Đóng contour hiện tại (thêm điểm đầu tiên vào cuối).
        
        Returns:
            bool: True nếu đóng thành công
        """
        if self.active_contour is None:
            return False
        
        contour = self.contours[self.active_contour]
        if len(contour) < 3:
            return False
        
        if contour[0] != contour[-1]:
            contour.append(contour[0])
            return True
        
        return False
    
    def create_mask(self, shape):
        """
        Tạo mask từ contour hiện tại.
        
        Parameters:
            shape (tuple): Kích thước mask (height, width)
        
        Returns:
            numpy.ndarray: Mask nhị phân
        """
        if self.active_contour is None:
            return np.zeros(shape, dtype=np.uint8)
        
        contour = self.contours[self.active_contour]
        if len(contour) < 3:
            return np.zeros(shape, dtype=np.uint8)
        
        # Chuyển đổi danh sách điểm thành định dạng phù hợp với cv2
        points = np.array(contour, dtype=np.int32)
        
        # Tạo mask
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    def draw_contours(self, ax, color=None, active_color='red', line_width=2):
        """
        Vẽ tất cả contours.
        
        Parameters:
            ax (matplotlib.axes.Axes): Axes để vẽ
            color (str, optional): Màu mặc định cho contour
            active_color (str, optional): Màu cho contour đang hoạt động
            line_width (int, optional): Độ rộng đường vẽ
        """
        for i, contour in enumerate(self.contours):
            if len(contour) < 2:
                continue
            
            # Chọn màu
            c = color
            if i in self.contour_colors:
                c = self.contour_colors[i]
            if i == self.active_contour:
                c = active_color
            
            # Vẽ contour
            contour_array = np.array(contour)
            ax.plot(contour_array[:, 0], contour_array[:, 1], '-', color=c, linewidth=line_width)
            
            # Vẽ các điểm
            ax.plot(contour_array[:, 0], contour_array[:, 1], 'o', color=c, markersize=4)
    
    def get_contour_points(self, contour_index=None):
        """
        Lấy danh sách điểm của contour.
        
        Parameters:
            contour_index (int, optional): Chỉ số contour, mặc định là contour hiện tại
        
        Returns:
            list: Danh sách các điểm [(x, y), ...]
        """
        if contour_index is None:
            contour_index = self.active_contour
        
        if contour_index is None or contour_index >= len(self.contours):
            return []
        
        return self.contours[contour_index]
    
    def get_all_contours(self):
        """
        Lấy tất cả contours.
        
        Returns:
            list: Danh sách các contour
        """
        return self.contours
    
    def clear_contours(self):
        """Xóa tất cả contours"""
        self.contours = []
        self.active_contour = None
    
    def delete_contour(self, contour_index=None):
        """
        Xóa một contour.
        
        Parameters:
            contour_index (int, optional): Chỉ số contour, mặc định là contour hiện tại
        
        Returns:
            bool: True nếu xóa thành công
        """
        if contour_index is None:
            contour_index = self.active_contour
        
        if contour_index is None or contour_index >= len(self.contours):
            return False
        
        del self.contours[contour_index]
        
        # Cập nhật active_contour
        if self.active_contour == contour_index:
            if len(self.contours) > 0:
                self.active_contour = 0
            else:
                self.active_contour = None
        elif self.active_contour > contour_index:
            self.active_contour -= 1
        
        return True
    
    def set_contour_color(self, contour_index, color):
        """
        Đặt màu cho contour.
        
        Parameters:
            contour_index (int): Chỉ số contour
            color (str): Mã màu
        """
        self.contour_colors[contour_index] = color
    
    def interpolate_contours(self, contour1, contour2, ratio=0.5):
        """
        Nội suy giữa hai contour.
        
        Parameters:
            contour1 (list): Contour thứ nhất
            contour2 (list): Contour thứ hai
            ratio (float): Tỉ lệ nội suy (0-1)
        
        Returns:
            list: Contour đã nội suy
        """
        if len(contour1) < 3 or len(contour2) < 3:
            return []
        
        # Đảm bảo cả hai contour đều đóng
        if contour1[0] != contour1[-1]:
            contour1 = contour1 + [contour1[0]]
        
        if contour2[0] != contour2[-1]:
            contour2 = contour2 + [contour2[0]]
        
        # Cần resampling nếu số điểm khác nhau
        if len(contour1) != len(contour2):
            # Đơn giản hóa bằng cách chọn contour có nhiều điểm hơn
            if len(contour1) > len(contour2):
                contour1 = self._resample_contour(contour1, len(contour2))
            else:
                contour2 = self._resample_contour(contour2, len(contour1))
        
        # Nội suy tuyến tính
        result = []
        for p1, p2 in zip(contour1, contour2):
            x = p1[0] * (1 - ratio) + p2[0] * ratio
            y = p1[1] * (1 - ratio) + p2[1] * ratio
            result.append((x, y))
        
        return result
    
    def _resample_contour(self, contour, n_points):
        """
        Lấy mẫu lại contour để có số điểm mong muốn.
        
        Parameters:
            contour (list): Contour gốc
            n_points (int): Số điểm mong muốn
        
        Returns:
            list: Contour đã lấy mẫu lại
        """
        if n_points < 3:
            return contour
        
        # Tính tổng độ dài contour
        total_length = 0
        segments = []
        
        for i in range(len(contour) - 1):
            p1 = contour[i]
            p2 = contour[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            total_length += length
            segments.append((p1, p2, length))
        
        # Tạo contour mới
        result = [contour[0]]  # Bắt đầu với điểm đầu tiên
        
        for i in range(1, n_points - 1):
            # Vị trí tương đối dọc theo contour
            target_pos = total_length * i / (n_points - 1)
            
            # Tìm đoạn chứa vị trí này
            current_pos = 0
            for start, end, length in segments:
                if current_pos <= target_pos < current_pos + length:
                    # Tính toán vị trí nội suy
                    ratio = (target_pos - current_pos) / length
                    x = start[0] + ratio * (end[0] - start[0])
                    y = start[1] + ratio * (end[1] - start[1])
                    result.append((x, y))
                    break
                current_pos += length
        
        result.append(contour[-1])  # Kết thúc với điểm cuối cùng
        return result


class BrushTool(ContourTool):
    """Công cụ vẽ bằng bút"""
    
    def __init__(self, brush_size=5):
        """
        Khởi tạo công cụ bút.
        
        Parameters:
            brush_size (int, optional): Kích thước bút
        """
        super().__init__()
        self.brush_size = brush_size
        self.mask = None
        self.shape = None
    
    def set_brush_size(self, size):
        """
        Thiết lập kích thước bút.
        
        Parameters:
            size (int): Kích thước bút
        """
        self.brush_size = max(1, size)
    
    def initialize(self, shape):
        """
        Khởi tạo mask trắng.
        
        Parameters:
            shape (tuple): Kích thước mask (height, width)
        """
        self.mask = np.zeros(shape, dtype=np.uint8)
        self.shape = shape
    
    def draw(self, point):
        """
        Vẽ tại điểm chỉ định.
        
        Parameters:
            point (tuple): Tọa độ điểm (x, y)
        
        Returns:
            bool: True nếu vẽ thành công
        """
        if self.mask is None:
            return False
        
        x, y = int(point[0]), int(point[1])
        cv2.circle(self.mask, (x, y), self.brush_size, 1, -1)
        return True
    
    def erase(self, point):
        """
        Xóa tại điểm chỉ định.
        
        Parameters:
            point (tuple): Tọa độ điểm (x, y)
        
        Returns:
            bool: True nếu xóa thành công
        """
        if self.mask is None:
            return False
        
        x, y = int(point[0]), int(point[1])
        cv2.circle(self.mask, (x, y), self.brush_size, 0, -1)
        return True
    
    def get_mask(self):
        """
        Lấy mask hiện tại.
        
        Returns:
            numpy.ndarray: Mask
        """
        return self.mask
    
    def clear(self):
        """Xóa mask"""
        if self.mask is not None:
            self.mask = np.zeros_like(self.mask)
    
    def extract_contours(self):
        """
        Trích xuất contours từ mask.
        
        Returns:
            list: Danh sách các contour
        """
        if self.mask is None:
            return []
        
        # Tìm contours
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chuyển đổi định dạng
        result = []
        for contour in contours:
            points = []
            for point in contour:
                x, y = point[0]
                points.append((x, y))
            if len(points) > 2:
                points.append(points[0])  # Đóng contour
                result.append(points)
        
        return result


class ThresholdTool(ContourTool):
    """Công cụ phân đoạn bằng ngưỡng"""
    
    def __init__(self):
        """Khởi tạo công cụ ngưỡng"""
        super().__init__()
        self.threshold_min = 0
        self.threshold_max = 255
    
    def set_threshold(self, min_val, max_val):
        """
        Thiết lập ngưỡng.
        
        Parameters:
            min_val (float): Ngưỡng dưới
            max_val (float): Ngưỡng trên
        """
        self.threshold_min = min_val
        self.threshold_max = max_val
    
    def apply_threshold(self, image):
        """
        Áp dụng ngưỡng lên hình ảnh.
        
        Parameters:
            image (numpy.ndarray): Hình ảnh đầu vào
        
        Returns:
            numpy.ndarray: Mask sau khi áp dụng ngưỡng
        """
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[(image >= self.threshold_min) & (image <= self.threshold_max)] = 1
        return mask
    
    def apply_threshold_region(self, image, seed_point, connectivity=4):
        """
        Áp dụng ngưỡng trong vùng kết nối với seed point.
        
        Parameters:
            image (numpy.ndarray): Hình ảnh đầu vào
            seed_point (tuple): Điểm hạt giống (x, y)
            connectivity (int, optional): Loại kết nối (4 hoặc 8)
        
        Returns:
            numpy.ndarray: Mask sau khi áp dụng ngưỡng
        """
        x, y = int(seed_point[0]), int(seed_point[1])
        
        # Tạo mask ban đầu dựa trên ngưỡng
        mask = self.apply_threshold(image)
        
        # Áp dụng region growing từ seed point
        if mask[y, x] == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Sử dụng connectedComponents để lấy vùng kết nối
        markers = np.zeros_like(mask, dtype=np.int32)
        markers[y, x] = 2  # Đánh dấu seed point
        
        # Áp dụng connected components
        if connectivity == 4:
            connectivity_type = 4
        else:
            connectivity_type = 8
        
        cv2.floodFill(mask, None, (x, y), 2, 0, 0, connectivity_type)
        region_mask = (mask == 2).astype(np.uint8)
        
        return region_mask


class RegionGrowingTool(ContourTool):
    """Công cụ phân đoạn bằng region growing"""
    
    def __init__(self):
        """Khởi tạo công cụ region growing"""
        super().__init__()
        self.tolerance = 10
    
    def set_tolerance(self, tolerance):
        """
        Thiết lập độ dung sai.
        
        Parameters:
            tolerance (float): Độ dung sai
        """
        self.tolerance = tolerance
    
    def apply_region_growing(self, image, seed_point):
        """
        Áp dụng thuật toán region growing.
        
        Parameters:
            image (numpy.ndarray): Hình ảnh đầu vào
            seed_point (tuple): Điểm hạt giống (x, y)
        
        Returns:
            numpy.ndarray: Mask sau khi áp dụng region growing
        """
        x, y = int(seed_point[0]), int(seed_point[1])
        
        # Tạo mask ban đầu
        mask = np.zeros_like(image, dtype=np.uint8)
        h, w = mask.shape
        
        if x < 0 or y < 0 or x >= w or y >= h:
            return mask
        
        # Lấy giá trị tại seed point
        seed_value = float(image[y, x])
        
        # Tạo mask cho flood fill
        mask = np.zeros((h+2, w+2), dtype=np.uint8)
        
        # Thiết lập các thông số cho floodFill
        lo_diff = self.tolerance
        up_diff = self.tolerance
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        
        # Áp dụng flood fill
        cv2.floodFill(image.astype(np.float32), mask, (x, y), 255, lo_diff, up_diff, flags)
        
        # Lấy kết quả (bỏ qua border)
        result = mask[1:-1, 1:-1]
        return result


class WatershedTool(ContourTool):
    """Công cụ phân đoạn bằng watershed"""
    
    def __init__(self):
        """Khởi tạo công cụ watershed"""
        super().__init__()
        self.markers = None
        self.current_label = 1
    
    def initialize(self, shape):
        """
        Khởi tạo markers.
        
        Parameters:
            shape (tuple): Kích thước markers (height, width)
        """
        self.markers = np.zeros(shape, dtype=np.int32)
        self.current_label = 1
    
    def add_marker(self, point, label=None):
        """
        Thêm marker.
        
        Parameters:
            point (tuple): Tọa độ điểm (x, y)
            label (int, optional): Nhãn, mặc định là nhãn hiện tại
        
        Returns:
            bool: True nếu thêm thành công
        """
        if self.markers is None:
            return False
        
        if label is None:
            label = self.current_label
        
        x, y = int(point[0]), int(point[1])
        cv2.circle(self.markers, (x, y), 5, label, -1)
        return True
    
    def set_current_label(self, label):
        """
        Thiết lập nhãn hiện tại.
        
        Parameters:
            label (int): Nhãn mới
        """
        self.current_label = label
    
    def next_label(self):
        """Chuyển sang nhãn tiếp theo"""
        self.current_label += 1
    
    def apply_watershed(self, image):
        """
        Áp dụng thuật toán watershed.
        
        Parameters:
            image (numpy.ndarray): Hình ảnh đầu vào
        
        Returns:
            numpy.ndarray: Kết quả phân đoạn
        """
        if self.markers is None or np.max(self.markers) == 0:
            return np.zeros_like(image, dtype=np.int32)
        
        # Chuyển đổi hình ảnh để tính gradient
        if len(image.shape) == 2:
            # Grayscale
            gradient = cv2.Laplacian(image, cv2.CV_64F)
        else:
            # RGB/Multi-channel
            gradient = np.zeros_like(image)
            for i in range(image.shape[2]):
                gradient[:,:,i] = cv2.Laplacian(image[:,:,i], cv2.CV_64F)
        
        # Chuẩn hóa gradient về [0, 1]
        gradient = np.abs(gradient)
        gradient = gradient / (np.max(gradient) + 1e-10)
        
        # Chuyển về định dạng phù hợp cho watershed
        gradient = (gradient * 255).astype(np.uint8)
        
        # Áp dụng watershed
        markers = self.markers.copy()
        cv2.watershed(gradient, markers)
        
        return markers
    
    def get_mask_for_label(self, markers, label):
        """
        Lấy mask cho một nhãn cụ thể.
        
        Parameters:
            markers (numpy.ndarray): Kết quả watershed
            label (int): Nhãn cần lấy mask
        
        Returns:
            numpy.ndarray: Mask nhị phân
        """
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 1
        return mask
