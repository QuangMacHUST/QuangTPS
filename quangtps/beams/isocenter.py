"""
Lớp Isocenter đại diện cho tâm iso của chùm tia xạ trị.
"""

from typing import Tuple, Dict, Any, Optional, List, Union


class Isocenter:
    """
    Lớp đại diện cho tâm iso của chùm tia xạ trị.

    Thuộc tính
    ----------
    position : Tuple[float, float, float]
        Tọa độ 3D của tâm iso (x, y, z) tính bằng mm
    name : Optional[str]
        Tên của tâm iso (nếu có)
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: Optional[str] = None,
    ):
        """
        Khởi tạo một tâm iso mới.

        Parameters
        ----------
        position : Tuple[float, float, float], optional
            Tọa độ 3D của tâm iso (x, y, z) tính bằng mm, mặc định là (0, 0, 0)
        name : Optional[str], optional
            Tên của tâm iso, mặc định là None
        """
        self.position = position
        self.name = name if name is not None else "Iso"

    @property
    def x(self) -> float:
        """Tọa độ x của tâm iso."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Tọa độ y của tâm iso."""
        return self.position[1]

    @property
    def z(self) -> float:
        """Tọa độ z của tâm iso."""
        return self.position[2]

    def set_position(self, x: float, y: float, z: float) -> None:
        """
        Thiết lập tọa độ của tâm iso.

        Parameters
        ----------
        x : float
            Tọa độ x (mm)
        y : float
            Tọa độ y (mm)
        z : float
            Tọa độ z (mm)
        """
        self.position = (x, y, z)

    def distance_to(
        self, other: Union["Isocenter", Tuple[float, float, float]]
    ) -> float:
        """
        Tính khoảng cách từ tâm iso này đến tâm iso khác hoặc điểm 3D.

        Parameters
        ----------
        other : Union[Isocenter, Tuple[float, float, float]]
            Tâm iso khác hoặc điểm 3D

        Returns
        -------
        float
            Khoảng cách Euclidean (mm)
        """
        if isinstance(other, Isocenter):
            other_pos = other.position
        else:
            other_pos = other

        return (
            (self.x - other_pos[0]) ** 2
            + (self.y - other_pos[1]) ** 2
            + (self.z - other_pos[2]) ** 2
        ) ** 0.5

    def move(
        self, delta_x: float = 0.0, delta_y: float = 0.0, delta_z: float = 0.0
    ) -> None:
        """
        Di chuyển tâm iso một khoảng.

        Parameters
        ----------
        delta_x : float, optional
            Khoảng di chuyển theo hướng x (mm), mặc định là 0
        delta_y : float, optional
            Khoảng di chuyển theo hướng y (mm), mặc định là 0
        delta_z : float, optional
            Khoảng di chuyển theo hướng z (mm), mặc định là 0
        """
        self.position = (self.x + delta_x, self.y + delta_y, self.z + delta_z)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi tâm iso thành từ điển.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin tâm iso
        """
        return {"position": self.position, "name": self.name}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Isocenter":
        """
        Tạo tâm iso từ từ điển.

        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin tâm iso

        Returns
        -------
        Isocenter
            Đối tượng tâm iso mới
        """
        return cls(position=tuple(data["position"]), name=data.get("name"))

    def __str__(self) -> str:
        """
        Biểu diễn chuỗi của tâm iso.

        Returns
        -------
        str
            Chuỗi mô tả tâm iso
        """
        return f"Isocenter(name={self.name}, position=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}))"
