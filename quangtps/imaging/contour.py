"""
Module chứa các lớp và hàm liên quan đến đường viền trong xạ trị.

Module này cung cấp các công cụ để tạo, chỉnh sửa và quản lý đường viền
của các cấu trúc giải phẫu trong hệ thống lập kế hoạch xạ trị.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Set, Any, Iterator
import logging
import numpy as np
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ContourType(Enum):
    """Các loại đường viền hỗ trợ."""
    CLOSED = auto()  # Đường viền kín (điểm đầu = điểm cuối)
    OPEN = auto()    # Đường viền hở (điểm đầu != điểm cuối)
    POINT = auto()   # Điểm đơn


class Contour:
    """
    Biểu diễn một đường viền trên một lát cắt ảnh.
    
    Một đường viền là một chuỗi các điểm trong không gian 3D tạo thành một
    đường khép kín hoặc hở để mô tả hình dạng của một cấu trúc giải phẫu.
    """
    
    def __init__(
        self,
        points: np.ndarray,
        contour_type: ContourType = ContourType.CLOSED,
        slice_index: int = 0,
        slice_position: float = 0.0
    ):
        """
        Khởi tạo đường viền.
        
        Parameters
        ----------
        points : np.ndarray
            Mảng điểm của đường viền, hình dạng (N, 3) cho N điểm trong không gian 3D
        contour_type : ContourType, optional
            Loại đường viền (kín, hở, điểm), mặc định là CLOSED
        slice_index : int, optional
            Chỉ số của lát cắt chứa đường viền, mặc định là 0
        slice_position : float, optional
            Vị trí z của lát cắt (mm), mặc định là 0.0
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)
            
        # Đảm bảo hình dạng (N, 3)
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != 3:
            raise ValueError(f"Mảng điểm phải có hình dạng (N, 3), nhận được {points.shape}")
            
        self.points = points
        self.contour_type = contour_type
        self.slice_index = slice_index
        self.slice_position = slice_position
        
        # Một số thuộc tính bổ sung
        self.metadata: Dict[str, Any] = {}
        self.is_hole = False  # True nếu đây là một lỗ trong đường viền cha
        self.is_selected = False  # Trạng thái chọn trong giao diện
    
    @property
    def num_points(self) -> int:
        """Số điểm trong đường viền."""
        return self.points.shape[0]
    
    @property
    def is_closed(self) -> bool:
        """Kiểm tra xem đường viền có khép kín không."""
        return self.contour_type == ContourType.CLOSED
    
    @property
    def is_empty(self) -> bool:
        """Kiểm tra xem đường viền có rỗng không."""
        return self.num_points == 0
    
    def get_center(self) -> np.ndarray:
        """
        Tính điểm trung tâm của đường viền.
        
        Returns
        -------
        np.ndarray
            Tọa độ 3D của điểm trung tâm
        """
        if self.is_empty:
            return np.zeros(3)
        return np.mean(self.points, axis=0)
    
    def get_area(self) -> float:
        """
        Tính diện tích của đường viền.
        
        Returns
        -------
        float
            Diện tích (mm²)
        """
        if not self.is_closed or self.num_points < 3:
            return 0.0
            
        # Sử dụng công thức Green để tính diện tích
        x = self.points[:, 0]
        y = self.points[:, 1]
        
        # Công thức: 0.5 * |∑(x_i * y_{i+1} - x_{i+1} * y_i)|
        return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
    
    def get_perimeter(self) -> float:
        """
        Tính chu vi của đường viền.
        
        Returns
        -------
        float
            Chu vi (mm)
        """
        if self.num_points < 2:
            return 0.0
            
        # Tính tổng khoảng cách giữa các điểm liên tiếp
        perimeter = np.sum(np.sqrt(np.sum(np.diff(self.points, axis=0) ** 2, axis=1)))
        
        # Nếu đường viền kín, thêm khoảng cách từ điểm cuối đến điểm đầu
        if self.is_closed and self.num_points > 1:
            perimeter += np.sqrt(np.sum((self.points[-1] - self.points[0]) ** 2))
            
        return perimeter
    
    def add_point(self, point: np.ndarray, index: Optional[int] = None) -> None:
        """
        Thêm điểm vào đường viền.
        
        Parameters
        ----------
        point : np.ndarray
            Tọa độ 3D của điểm cần thêm
        index : Optional[int], optional
            Vị trí chèn, mặc định là thêm vào cuối
        """
        point_array = np.array(point, dtype=np.float64).reshape(1, 3)
        
        if index is None or index >= self.num_points:
            self.points = np.vstack([self.points, point_array])
        else:
            self.points = np.vstack([
                self.points[:index],
                point_array,
                self.points[index:]
            ])
    
    def remove_point(self, index: int) -> bool:
        """
        Xóa điểm khỏi đường viền.
        
        Parameters
        ----------
        index : int
            Chỉ số của điểm cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công
        """
        if 0 <= index < self.num_points:
            if self.num_points <= 1:
                self.points = np.zeros((0, 3))
            else:
                self.points = np.delete(self.points, index, axis=0)
            return True
        return False
    
    def move_point(self, index: int, new_position: np.ndarray) -> bool:
        """
        Di chuyển điểm đến vị trí mới.
        
        Parameters
        ----------
        index : int
            Chỉ số của điểm cần di chuyển
        new_position : np.ndarray
            Tọa độ 3D mới
            
        Returns
        -------
        bool
            True nếu di chuyển thành công
        """
        if 0 <= index < self.num_points:
            self.points[index] = np.array(new_position, dtype=np.float64)
            return True
        return False
    
    def translate(self, vector: np.ndarray) -> None:
        """
        Dịch chuyển toàn bộ đường viền.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector dịch chuyển 3D
        """
        self.points += np.array(vector, dtype=np.float64)
    
    def scale(self, factor: Union[float, np.ndarray], center: Optional[np.ndarray] = None) -> None:
        """
        Co giãn đường viền.
        
        Parameters
        ----------
        factor : Union[float, np.ndarray]
            Hệ số co giãn, có thể là số vô hướng hoặc vector 3D
        center : Optional[np.ndarray], optional
            Tâm co giãn, mặc định là trung tâm đường viền
        """
        if center is None:
            center = self.get_center()
        else:
            center = np.array(center, dtype=np.float64)
            
        # Dịch đường viền về gốc tọa độ
        self.points -= center
        
        # Co giãn
        if isinstance(factor, (int, float)):
            self.points *= factor
        else:
            self.points *= np.array(factor, dtype=np.float64)
            
        # Dịch trở lại vị trí gốc
        self.points += center
    
    def rotate(self, angle: float, axis: np.ndarray = np.array([0, 0, 1]), center: Optional[np.ndarray] = None) -> None:
        """
        Xoay đường viền.
        
        Parameters
        ----------
        angle : float
            Góc xoay (radian)
        axis : np.ndarray, optional
            Trục xoay, mặc định là trục Z
        center : Optional[np.ndarray], optional
            Tâm xoay, mặc định là trung tâm đường viền
        """
        if center is None:
            center = self.get_center()
        else:
            center = np.array(center, dtype=np.float64)
            
        # Chuẩn hóa trục
        axis = np.array(axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        
        # Ma trận xoay (Rodrigues' rotation formula)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = cos_angle * np.eye(3) + sin_angle * K + (1 - cos_angle) * np.outer(axis, axis)
        
        # Dịch đường viền về gốc tọa độ, xoay và dịch trở lại
        centered_points = self.points - center
        self.points = np.dot(centered_points, R.T) + center
    
    def resample(self, num_points: int) -> None:
        """
        Tái lấy mẫu đường viền với số điểm mới.
        
        Parameters
        ----------
        num_points : int
            Số điểm mới
        """
        if not self.is_closed or self.num_points < 3 or num_points < 3:
            return
            
        # Tính tổng chiều dài
        points = self.points
        if self.is_closed:
            points = np.vstack([points, points[0]])
            
        # Tính chiều dài tích lũy dọc theo đường viền
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cumulative_length[-1]
        
        # Tạo các điểm mới cách đều nhau
        target_distances = np.linspace(0, total_length, num_points)
        new_points = np.zeros((num_points, 3))
        
        for i, target in enumerate(target_distances):
            if target <= 0:
                new_points[i] = points[0]
            elif target >= total_length:
                new_points[i] = points[-1]
            else:
                # Tìm đoạn chứa điểm mục tiêu
                idx = np.searchsorted(cumulative_length, target) - 1
                
                # Nội suy tuyến tính trong đoạn
                segment_length = segment_lengths[idx]
                if segment_length > 0:
                    alpha = (target - cumulative_length[idx]) / segment_length
                    new_points[i] = points[idx] + alpha * diffs[idx]
                else:
                    new_points[i] = points[idx]
                    
        self.points = new_points
    
    def smooth(self, kernel_size: int = 3) -> None:
        """
        Làm mịn đường viền bằng bộ lọc trung bình.
        
        Parameters
        ----------
        kernel_size : int, optional
            Kích thước kernel làm mịn, mặc định là 3
        """
        if not self.is_closed or self.num_points < 3 or kernel_size < 2:
            return
            
        # Tạo mảng điểm với wrap-around cho đường viền kín
        if self.is_closed:
            half = kernel_size // 2
            wrapped_points = np.vstack([
                self.points[-half:],
                self.points,
                self.points[:half]
            ])
        else:
            wrapped_points = self.points
            
        # Áp dụng bộ lọc trung bình di động
        smoothed = np.zeros_like(self.points)
        for i in range(self.num_points):
            if self.is_closed:
                start = i
                end = i + kernel_size
                smoothed[i] = np.mean(wrapped_points[start:end], axis=0)
            else:
                # Xử lý đặc biệt cho điểm đầu và cuối của đường viền hở
                left = max(0, i - kernel_size // 2)
                right = min(self.num_points, i + kernel_size // 2 + 1)
                smoothed[i] = np.mean(wrapped_points[left:right], axis=0)
                
        self.points = smoothed
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đường viền thành từ điển.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin đường viền
        """
        return {
            "points": self.points.tolist(),
            "contour_type": self.contour_type.name,
            "slice_index": self.slice_index,
            "slice_position": self.slice_position,
            "is_hole": self.is_hole,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Contour:
        """
        Tạo đường viền từ từ điển.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin đường viền
            
        Returns
        -------
        Contour
            Đối tượng đường viền mới
        """
        points = np.array(data["points"], dtype=np.float64)
        
        contour_type = ContourType[data["contour_type"]] \
            if "contour_type" in data else ContourType.CLOSED
            
        contour = cls(
            points=points,
            contour_type=contour_type,
            slice_index=data.get("slice_index", 0),
            slice_position=data.get("slice_position", 0.0)
        )
        
        contour.is_hole = data.get("is_hole", False)
        contour.metadata = data.get("metadata", {})
        
        return contour
    
    def copy(self) -> Contour:
        """
        Tạo một bản sao độc lập của đường viền.
        
        Returns
        -------
        Contour
            Bản sao của đường viền
        """
        contour = Contour(
            points=self.points.copy(),
            contour_type=self.contour_type,
            slice_index=self.slice_index,
            slice_position=self.slice_position
        )
        
        contour.is_hole = self.is_hole
        contour.is_selected = self.is_selected
        contour.metadata = self.metadata.copy()
        
        return contour
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của đường viền."""
        return f"Contour(points={self.num_points}, type={self.contour_type.name}, slice={self.slice_index})"


class ContourCollection:
    """
    Tập hợp các đường viền, thường liên quan đến một cấu trúc trên nhiều lát cắt.
    
    ContourCollection quản lý một tập hợp đường viền và cung cấp các phương thức
    để truy cập, thêm, xóa và xử lý các đường viền theo lát cắt.
    """
    
    def __init__(self, structure_id: Optional[str] = None):
        """
        Khởi tạo tập hợp đường viền.
        
        Parameters
        ----------
        structure_id : Optional[str], optional
            ID của cấu trúc liên quan, mặc định là None
        """
        self.structure_id = structure_id
        
        # Từ điển ánh xạ từ chỉ số lát cắt đến danh sách đường viền
        self.contours: Dict[int, List[Contour]] = {}
        
        # Siêu dữ liệu
        self.metadata: Dict[str, Any] = {}
    
    def add_contour(self, contour: Contour) -> None:
        """
        Thêm đường viền vào tập hợp.
        
        Parameters
        ----------
        contour : Contour
            Đường viền cần thêm
        """
        slice_index = contour.slice_index
        
        if slice_index not in self.contours:
            self.contours[slice_index] = []
            
        self.contours[slice_index].append(contour)
    
    def remove_contour(self, slice_index: int, contour_index: int) -> bool:
        """
        Xóa đường viền khỏi tập hợp.
        
        Parameters
        ----------
        slice_index : int
            Chỉ số lát cắt
        contour_index : int
            Chỉ số đường viền trong lát cắt
            
        Returns
        -------
        bool
            True nếu xóa thành công
        """
        if slice_index in self.contours and 0 <= contour_index < len(self.contours[slice_index]):
            self.contours[slice_index].pop(contour_index)
            
            # Nếu lát cắt không còn đường viền nào, xóa khỏi từ điển
            if not self.contours[slice_index]:
                del self.contours[slice_index]
                
            return True
        return False
    
    def clear_slice(self, slice_index: int) -> bool:
        """
        Xóa tất cả đường viền trên một lát cắt.
        
        Parameters
        ----------
        slice_index : int
            Chỉ số lát cắt
            
        Returns
        -------
        bool
            True nếu xóa thành công
        """
        if slice_index in self.contours:
            del self.contours[slice_index]
            return True
        return False
    
    def clear_all(self) -> None:
        """Xóa tất cả đường viền trong tập hợp."""
        self.contours = {}
    
    def get_contour(self, slice_index: int, contour_index: int) -> Optional[Contour]:
        """
        Lấy đường viền theo chỉ số.
        
        Parameters
        ----------
        slice_index : int
            Chỉ số lát cắt
        contour_index : int
            Chỉ số đường viền trong lát cắt
            
        Returns
        -------
        Optional[Contour]
            Đường viền nếu tìm thấy, None nếu không
        """
        if slice_index in self.contours and 0 <= contour_index < len(self.contours[slice_index]):
            return self.contours[slice_index][contour_index]
        return None
    
    def get_slice_contours(self, slice_index: int) -> List[Contour]:
        """
        Lấy tất cả đường viền trên một lát cắt.
        
        Parameters
        ----------
        slice_index : int
            Chỉ số lát cắt
            
        Returns
        -------
        List[Contour]
            Danh sách đường viền trên lát cắt
        """
        return self.contours.get(slice_index, [])
    
    def has_slice(self, slice_index: int) -> bool:
        """
        Kiểm tra xem lát cắt có đường viền không.
        
        Parameters
        ----------
        slice_index : int
            Chỉ số lát cắt
            
        Returns
        -------
        bool
            True nếu lát cắt có ít nhất một đường viền
        """
        return slice_index in self.contours and len(self.contours[slice_index]) > 0
    
    def get_slice_indices(self) -> List[int]:
        """
        Lấy danh sách chỉ số lát cắt có đường viền.
        
        Returns
        -------
        List[int]
            Danh sách chỉ số lát cắt, sắp xếp tăng dần
        """
        return sorted(self.contours.keys())
    
    def get_num_contours(self) -> int:
        """
        Đếm tổng số đường viền trong tập hợp.
        
        Returns
        -------
        int
            Tổng số đường viền
        """
        return sum(len(contours) for contours in self.contours.values())
    
    def get_num_slices(self) -> int:
        """
        Đếm số lát cắt có đường viền.
        
        Returns
        -------
        int
            Số lát cắt có đường viền
        """
        return len(self.contours)
    
    def is_empty(self) -> bool:
        """
        Kiểm tra xem tập hợp có rỗng không.
        
        Returns
        -------
        bool
            True nếu tập hợp không có đường viền nào
        """
        return self.get_num_contours() == 0
    
    def iterate_contours(self) -> Iterator[Tuple[int, int, Contour]]:
        """
        Duyệt qua tất cả đường viền trong tập hợp.
        
        Yields
        ------
        Iterator[Tuple[int, int, Contour]]
            (slice_index, contour_index, contour) cho mỗi đường viền
        """
        for slice_index, slice_contours in sorted(self.contours.items()):
            for contour_index, contour in enumerate(slice_contours):
                yield slice_index, contour_index, contour
    
    def translate_all(self, vector: np.ndarray) -> None:
        """
        Dịch chuyển tất cả đường viền.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector dịch chuyển 3D
        """
        vector = np.array(vector, dtype=np.float64)
        
        for _, _, contour in self.iterate_contours():
            contour.translate(vector)
    
    def scale_all(self, factor: Union[float, np.ndarray], center: Optional[np.ndarray] = None) -> None:
        """
        Co giãn tất cả đường viền.
        
        Parameters
        ----------
        factor : Union[float, np.ndarray]
            Hệ số co giãn
        center : Optional[np.ndarray], optional
            Tâm co giãn, mặc định là trung tâm của tất cả đường viền
        """
        if center is None:
            # Tính trung tâm của tất cả đường viền
            all_points = np.vstack([contour.points for _, _, contour in self.iterate_contours()])
            center = np.mean(all_points, axis=0) if len(all_points) > 0 else np.zeros(3)
            
        for _, _, contour in self.iterate_contours():
            contour.scale(factor, center)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi tập hợp đường viền thành từ điển.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin tập hợp đường viền
        """
        contour_dict = {}
        
        for slice_index, slice_contours in self.contours.items():
            contour_dict[str(slice_index)] = [contour.to_dict() for contour in slice_contours]
            
        return {
            "structure_id": self.structure_id,
            "contours": contour_dict,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContourCollection:
        """
        Tạo tập hợp đường viền từ từ điển.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin tập hợp đường viền
            
        Returns
        -------
        ContourCollection
            Đối tượng tập hợp đường viền mới
        """
        collection = cls(structure_id=data.get("structure_id"))
        collection.metadata = data.get("metadata", {})
        
        if "contours" in data:
            for slice_str, contour_data_list in data["contours"].items():
                slice_index = int(slice_str)
                
                for contour_data in contour_data_list:
                    contour = Contour.from_dict(contour_data)
                    collection.add_contour(contour)
                    
        return collection
    
    def copy(self) -> ContourCollection:
        """
        Tạo một bản sao độc lập của tập hợp đường viền.
        
        Returns
        -------
        ContourCollection
            Bản sao của tập hợp đường viền
        """
        collection = ContourCollection(structure_id=self.structure_id)
        collection.metadata = self.metadata.copy()
        
        for slice_index, slice_contours in self.contours.items():
            collection.contours[slice_index] = [contour.copy() for contour in slice_contours]
            
        return collection
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của tập hợp đường viền."""
        return f"ContourCollection(slices={self.get_num_slices()}, contours={self.get_num_contours()})"
