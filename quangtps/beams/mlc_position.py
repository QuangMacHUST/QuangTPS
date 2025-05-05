"""
Lớp MLCPosition đại diện cho vị trí của các lá MLC trong một điểm điều khiển.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple


class MLCPosition:
    """
    Lớp đại diện cho vị trí của các lá MLC trong một điểm điều khiển.

    Thuộc tính
    ----------
    leaf_positions_a : np.ndarray
        Vị trí của các lá ngân A (mm)
    leaf_positions_b : np.ndarray
        Vị trí của các lá ngân B (mm)
    mlc_type : str
        Loại MLC
    num_leaves : int
        Số lượng lá MLC
    """

    def __init__(
        self,
        leaf_positions_a: Optional[Union[List[float], np.ndarray]] = None,
        leaf_positions_b: Optional[Union[List[float], np.ndarray]] = None,
        mlc_type: str = "Millennium120",
        num_leaves: int = 120,
    ):
        """
        Khởi tạo một vị trí MLC mới.

        Parameters
        ----------
        leaf_positions_a : Optional[Union[List[float], np.ndarray]], optional
            Vị trí của các lá ngân A (mm), mặc định là tất cả -200
        leaf_positions_b : Optional[Union[List[float], np.ndarray]], optional
            Vị trí của các lá ngân B (mm), mặc định là tất cả 200
        mlc_type : str, optional
            Loại MLC, mặc định là "Millennium120"
        num_leaves : int, optional
            Số lượng lá MLC, mặc định là 120
        """
        self.mlc_type = mlc_type
        self.num_leaves = num_leaves

        # Khởi tạo vị trí mặc định
        if leaf_positions_a is None:
            self.leaf_positions_a = np.full(num_leaves, -200.0)
        else:
            self.leaf_positions_a = np.array(leaf_positions_a, dtype=float)

        if leaf_positions_b is None:
            self.leaf_positions_b = np.full(num_leaves, 200.0)
        else:
            self.leaf_positions_b = np.array(leaf_positions_b, dtype=float)

        # Đảm bảo kích thước mảng đúng
        if len(self.leaf_positions_a) != num_leaves:
            self.leaf_positions_a = np.resize(self.leaf_positions_a, num_leaves)

        if len(self.leaf_positions_b) != num_leaves:
            self.leaf_positions_b = np.resize(self.leaf_positions_b, num_leaves)

    def set_leaf_positions(
        self, bank: str, positions: Union[List[float], np.ndarray]
    ) -> None:
        """
        Thiết lập vị trí các lá MLC.

        Parameters
        ----------
        bank : str
            Ngân MLC ('A' hoặc 'B')
        positions : Union[List[float], np.ndarray]
            Danh sách vị trí các lá MLC (mm)
        """
        positions_array = np.array(positions, dtype=float)

        if bank.upper() == "A":
            self.leaf_positions_a = np.resize(positions_array, self.num_leaves)
        elif bank.upper() == "B":
            self.leaf_positions_b = np.resize(positions_array, self.num_leaves)
        else:
            raise ValueError(f"Ngân không hợp lệ: {bank}. Chỉ chấp nhận 'A' hoặc 'B'")

    def set_single_leaf_position(
        self, bank: str, leaf_index: int, position: float
    ) -> None:
        """
        Thiết lập vị trí của một lá MLC cụ thể.

        Parameters
        ----------
        bank : str
            Ngân MLC ('A' hoặc 'B')
        leaf_index : int
            Chỉ số của lá MLC
        position : float
            Vị trí của lá MLC (mm)
        """
        if leaf_index < 0 or leaf_index >= self.num_leaves:
            raise IndexError(
                f"Chỉ số lá MLC không hợp lệ: {leaf_index}. Phải từ 0 đến {self.num_leaves - 1}"
            )

        if bank.upper() == "A":
            self.leaf_positions_a[leaf_index] = float(position)
        elif bank.upper() == "B":
            self.leaf_positions_b[leaf_index] = float(position)
        else:
            raise ValueError(f"Ngân không hợp lệ: {bank}. Chỉ chấp nhận 'A' hoặc 'B'")

    def set_rectangular_field(self, field_size: Tuple[float, float]) -> None:
        """
        Thiết lập trường chữ nhật.

        Parameters
        ----------
        field_size : Tuple[float, float]
            Kích thước trường (width, height) tính bằng mm
        """
        half_width = field_size[0] / 2

        # Thiết lập tất cả các lá sang hai bên để tạo trường chữ nhật
        self.leaf_positions_a = np.full(self.num_leaves, -half_width)
        self.leaf_positions_b = np.full(self.num_leaves, half_width)

    def get_leaf_gaps(self) -> np.ndarray:
        """
        Tính toán khoảng cách giữa các lá MLC.

        Returns
        -------
        np.ndarray
            Mảng các khoảng cách (mm) giữa các lá MLC
        """
        return self.leaf_positions_b - self.leaf_positions_a

    def get_transmission_map(
        self, resolution: Tuple[int, int] = (100, 100)
    ) -> np.ndarray:
        """
        Tạo bản đồ truyền qua của MLC với độ phân giải chỉ định.

        Parameters
        ----------
        resolution : Tuple[int, int], optional
            Độ phân giải của bản đồ (width, height), mặc định là (100, 100)

        Returns
        -------
        np.ndarray
            Bản đồ truyền qua 2D (1 = mở, 0 = đóng)
        """
        width, height = resolution
        transmission_map = np.zeros((height, width))

        # Tính toán các thông số cần thiết
        leaf_width = height / self.num_leaves
        x_min = min(np.min(self.leaf_positions_a), np.min(self.leaf_positions_b))
        x_max = max(np.max(self.leaf_positions_a), np.max(self.leaf_positions_b))
        x_range = x_max - x_min

        # Chuyển đổi tọa độ MLC thành chỉ số pixel
        for i in range(self.num_leaves):
            y_start = int(i * leaf_width)
            y_end = int((i + 1) * leaf_width)

            x_start = int((self.leaf_positions_a[i] - x_min) / x_range * width)
            x_end = int((self.leaf_positions_b[i] - x_min) / x_range * width)

            # Đảm bảo giá trị nằm trong phạm vi hợp lệ
            x_start = max(0, min(width - 1, x_start))
            x_end = max(0, min(width - 1, x_end))

            # Đánh dấu vùng mở
            if x_end > x_start:
                transmission_map[y_start:y_end, x_start:x_end] = 1.0

        return transmission_map

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi vị trí MLC thành từ điển.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin vị trí MLC
        """
        return {
            "leaf_positions_a": self.leaf_positions_a.tolist(),
            "leaf_positions_b": self.leaf_positions_b.tolist(),
            "mlc_type": self.mlc_type,
            "num_leaves": self.num_leaves,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLCPosition":
        """
        Tạo vị trí MLC từ từ điển.

        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin vị trí MLC

        Returns
        -------
        MLCPosition
            Đối tượng vị trí MLC mới
        """
        return cls(
            leaf_positions_a=data["leaf_positions_a"],
            leaf_positions_b=data["leaf_positions_b"],
            mlc_type=data.get("mlc_type", "Millennium120"),
            num_leaves=data.get("num_leaves", 120),
        )

    def __str__(self) -> str:
        """
        Biểu diễn chuỗi của vị trí MLC.

        Returns
        -------
        str
            Chuỗi mô tả vị trí MLC
        """
        return f"MLCPosition(type={self.mlc_type}, leaves={self.num_leaves})"
