#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các thiết bị điều biến chùm tia xạ trị (Beam Modifiers).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý
các thiết bị điều biến chùm tia như wedge, block, bolus, và bộ bù liều.
"""

import uuid
import logging
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModifierType(str, Enum):
    """Enum đại diện cho các loại thiết bị điều biến."""

    WEDGE = "WEDGE"
    BLOCK = "BLOCK"
    BOLUS = "BOLUS"
    COMPENSATOR = "COMPENSATOR"
    MLC = "MLC"


class BeamModifier:
    """
    Lớp cơ sở cho các thiết bị điều biến chùm tia.

    Các lớp con bao gồm: Wedge, Block, Bolus, và Compensator.
    """

    def __init__(
        self, name: str, modifier_type: ModifierType, modifier_id: Optional[str] = None
    ):
        """
        Khởi tạo một thiết bị điều biến chùm tia.

        Parameters
        ----------
        name : str
            Tên của thiết bị điều biến
        modifier_type : ModifierType
            Loại thiết bị điều biến (WEDGE, BLOCK, BOLUS, ...)
        modifier_id : str, optional
            ID duy nhất của thiết bị điều biến
        """
        self.name = name
        self.modifier_type = modifier_type
        self.modifier_id = modifier_id if modifier_id else str(uuid.uuid4())
        self.created_at = None
        self.updated_at = None

    def get_name(self) -> str:
        """
        Lấy tên của thiết bị điều biến.

        Returns
        -------
        str
            Tên của thiết bị điều biến
        """
        return self.name

    def get_id(self) -> str:
        """
        Lấy ID của thiết bị điều biến.

        Returns
        -------
        str
            ID của thiết bị điều biến
        """
        return self.modifier_id

    def get_type(self) -> ModifierType:
        """
        Lấy loại của thiết bị điều biến.

        Returns
        -------
        ModifierType
            Loại thiết bị điều biến
        """
        return self.modifier_type

    def set_name(self, name: str) -> None:
        """
        Thiết lập tên cho thiết bị điều biến.

        Parameters
        ----------
        name : str
            Tên của thiết bị điều biến
        """
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin thiết bị điều biến thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin cơ bản của thiết bị điều biến
        """
        return {
            "name": self.name,
            "modifier_id": self.modifier_id,
            "modifier_type": self.modifier_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Wedge(BeamModifier):
    """
    Lớp đại diện cho một wedge (nêm).

    Wedge là thiết bị điều biến làm thay đổi đường đồng liều của chùm tia,
    có thể là vật lý (physical) hoặc động (dynamic/virtual).
    """

    def __init__(
        self,
        name: str,
        angle: float,
        orientation: float = 0.0,
        wedge_id: Optional[str] = None,
        wedge_type: str = "PHYSICAL",
    ):
        """
        Khởi tạo một wedge.

        Parameters
        ----------
        name : str
            Tên của wedge
        angle : float
            Góc của wedge (độ)
        orientation : float, optional
            Hướng của wedge (độ)
        wedge_id : str, optional
            ID duy nhất của wedge
        wedge_type : str, optional
            Loại wedge (PHYSICAL, DYNAMIC, VIRTUAL)
        """
        super().__init__(name, ModifierType.WEDGE, wedge_id)
        self.angle = angle
        self.orientation = orientation
        self.wedge_type = wedge_type
        self.direction = "IN"  # IN hoặc OUT
        self.material = "Lead"
        self.attenuation_factor = self._calculate_attenuation_factor()

    def _calculate_attenuation_factor(self) -> float:
        """
        Tính toán hệ số suy giảm dựa trên góc wedge.

        Returns
        -------
        float
            Hệ số suy giảm
        """
        # Công thức ước lượng đơn giản
        # Trong thực tế, cần có dữ liệu thực nghiệm hoặc từ nhà sản xuất
        if self.wedge_type == "PHYSICAL":
            return 1.0 - (self.angle / 180.0) * 0.5
        else:
            return 1.0  # Wedge động không gây suy giảm đáng kể

    def get_attenuation_factor(self) -> float:
        """
        Lấy hệ số suy giảm của wedge.

        Returns
        -------
        float
            Hệ số suy giảm
        """
        return self.attenuation_factor

    def set_angle(self, angle: float):
        """
        Thiết lập góc wedge.

        Parameters
        ----------
        angle : float
            Góc wedge (độ)
        """
        self.angle = angle
        self.attenuation_factor = self._calculate_attenuation_factor()

    def set_orientation(self, orientation: float):
        """
        Thiết lập hướng wedge.

        Parameters
        ----------
        orientation : float
            Hướng wedge (độ)
        """
        self.orientation = orientation

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin wedge thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin wedge
        """
        data = super().to_dict()
        data.update(
            {
                "angle": self.angle,
                "orientation": self.orientation,
                "wedge_type": self.wedge_type,
                "direction": self.direction,
                "material": self.material,
                "attenuation_factor": self.attenuation_factor,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Wedge":
        """
        Tạo đối tượng Wedge từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin wedge

        Returns
        -------
        Wedge
            Đối tượng Wedge
        """
        wedge = cls(
            name=data["name"],
            angle=data["angle"],
            orientation=data["orientation"],
            wedge_id=data["modifier_id"],
            wedge_type=data.get("wedge_type", "PHYSICAL"),
        )
        wedge.is_active = data["is_active"]
        wedge.direction = data.get("direction", "IN")
        wedge.material = data.get("material", "Lead")
        wedge.attenuation_factor = data.get(
            "attenuation_factor", wedge._calculate_attenuation_factor()
        )
        wedge.metadata = data.get("metadata", {})
        return wedge


class Block(BeamModifier):
    """
    Lớp đại diện cho một block (khối chắn).

    Block là thiết bị điều biến dùng để chắn chùm tia theo hình dạng tùy chỉnh.
    """

    def __init__(
        self,
        name: str,
        points: List[Tuple[float, float]],
        block_id: Optional[str] = None,
    ):
        """
        Khởi tạo một block.

        Parameters
        ----------
        name : str
            Tên của block
        points : List[Tuple[float, float]]
            Danh sách các điểm tạo thành hình dạng của block (cm)
        block_id : str, optional
            ID duy nhất của block
        """
        super().__init__(name, ModifierType.BLOCK, block_id)
        self.points = points
        self.thickness = 7.5  # Độ dày block (cm)
        self.material = "Cerrobend"  # Vật liệu
        self.density = 9.4  # Mật độ vật liệu (g/cm³)
        self.transmission_factor = 0.05  # Hệ số truyền qua

        # Tính toán kích thước block
        self._calculate_dimensions()

    def _calculate_dimensions(self):
        """Tính toán kích thước của block dựa trên tập hợp các điểm."""
        if not self.points:
            self.width = 0.0
            self.height = 0.0
            self.area = 0.0
            return

        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]

        # Tính kích thước
        self.width = max(x_coords) - min(x_coords)
        self.height = max(y_coords) - min(y_coords)

        # Tính diện tích xấp xỉ bằng phương pháp shoelace
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        self.area = abs(area) / 2.0

    def get_effective_size_change(self) -> Tuple[float, float]:
        """
        Lấy sự thay đổi kích thước hiệu dụng khi sử dụng block.

        Returns
        -------
        Tuple[float, float]
            Sự thay đổi kích thước theo X và Y (cm)
        """
        if not self.is_active:
            return (0.0, 0.0)

        return (self.width, self.height)

    def get_attenuation_factor(self) -> float:
        """
        Lấy hệ số suy giảm của block.

        Returns
        -------
        float
            Hệ số suy giảm
        """
        if not self.is_active:
            return 1.0

        return self.transmission_factor

    def set_points(self, points: List[Tuple[float, float]]):
        """
        Thiết lập các điểm tạo thành hình dạng của block.

        Parameters
        ----------
        points : List[Tuple[float, float]]
            Danh sách các điểm tạo thành hình dạng của block (cm)
        """
        self.points = points
        self._calculate_dimensions()

    def set_material_properties(
        self, material: str, density: float, transmission_factor: float
    ):
        """
        Thiết lập thuộc tính vật liệu của block.

        Parameters
        ----------
        material : str
            Vật liệu của block
        density : float
            Mật độ của vật liệu (g/cm³)
        transmission_factor : float
            Hệ số truyền qua
        """
        self.material = material
        self.density = density
        self.transmission_factor = transmission_factor

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin block thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin block
        """
        data = super().to_dict()
        data.update(
            {
                "points": self.points,
                "thickness": self.thickness,
                "material": self.material,
                "density": self.density,
                "transmission_factor": self.transmission_factor,
                "width": self.width,
                "height": self.height,
                "area": self.area,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        """
        Tạo đối tượng Block từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin block

        Returns
        -------
        Block
            Đối tượng Block
        """
        block = cls(
            name=data["name"], points=data["points"], block_id=data["modifier_id"]
        )
        block.is_active = data["is_active"]
        block.thickness = data.get("thickness", 7.5)
        block.material = data.get("material", "Cerrobend")
        block.density = data.get("density", 9.4)
        block.transmission_factor = data.get("transmission_factor", 0.05)
        block.width = data.get("width", 0.0)
        block.height = data.get("height", 0.0)
        block.area = data.get("area", 0.0)
        block.metadata = data.get("metadata", {})
        return block


class Bolus(BeamModifier):
    """
    Lớp đại diện cho một bolus.

    Bolus là vật liệu tương đương mô được đặt trên bề mặt bệnh nhân để tăng liều bề mặt.
    """

    def __init__(self, name: str, thickness: float, bolus_id: Optional[str] = None):
        """
        Khởi tạo một bolus.

        Parameters
        ----------
        name : str
            Tên của bolus
        thickness : float
            Độ dày của bolus (cm)
        bolus_id : str, optional
            ID duy nhất của bolus
        """
        super().__init__(name, ModifierType.BOLUS, bolus_id)
        self.thickness = thickness
        self.material = "Tissue Equivalent"
        self.relative_electron_density = 1.0  # Tương đương mô
        self.shape = "RECTANGULAR"  # RECTANGULAR, CUSTOM
        self.dimensions = (10.0, 10.0)  # Kích thước (cm) cho hình chữ nhật
        self.contour_points = []  # Điểm contour cho hình dạng tùy chỉnh

    def set_thickness(self, thickness: float):
        """
        Thiết lập độ dày của bolus.

        Parameters
        ----------
        thickness : float
            Độ dày của bolus (cm)
        """
        self.thickness = thickness

    def set_rectangular_dimensions(self, width: float, height: float):
        """
        Thiết lập kích thước hình chữ nhật cho bolus.

        Parameters
        ----------
        width : float
            Chiều rộng (cm)
        height : float
            Chiều cao (cm)
        """
        self.shape = "RECTANGULAR"
        self.dimensions = (width, height)

    def set_custom_shape(self, contour_points: List[Tuple[float, float, float]]):
        """
        Thiết lập hình dạng tùy chỉnh cho bolus.

        Parameters
        ----------
        contour_points : List[Tuple[float, float, float]]
            Danh sách các điểm tạo thành hình dạng của bolus (cm)
        """
        self.shape = "CUSTOM"
        self.contour_points = contour_points

    def get_attenuation_factor(self) -> float:
        """
        Lấy hệ số suy giảm của bolus.

        Returns
        -------
        float
            Hệ số suy giảm
        """
        if not self.is_active:
            return 1.0

        # Suy giảm đơn giản, trong thực tế cần dùng dữ liệu vật lý
        return 1.0 - 0.05 * self.thickness * self.relative_electron_density

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin bolus thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin bolus
        """
        data = super().to_dict()
        data.update(
            {
                "thickness": self.thickness,
                "material": self.material,
                "relative_electron_density": self.relative_electron_density,
                "shape": self.shape,
                "dimensions": self.dimensions,
            }
        )

        if self.shape == "CUSTOM":
            data["contour_points"] = self.contour_points

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bolus":
        """
        Tạo đối tượng Bolus từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin bolus

        Returns
        -------
        Bolus
            Đối tượng Bolus
        """
        bolus = cls(
            name=data["name"], thickness=data["thickness"], bolus_id=data["modifier_id"]
        )
        bolus.is_active = data["is_active"]
        bolus.material = data.get("material", "Tissue Equivalent")
        bolus.relative_electron_density = data.get("relative_electron_density", 1.0)
        bolus.shape = data.get("shape", "RECTANGULAR")
        bolus.dimensions = data.get("dimensions", (10.0, 10.0))

        if bolus.shape == "CUSTOM" and "contour_points" in data:
            bolus.contour_points = data["contour_points"]

        bolus.metadata = data.get("metadata", {})
        return bolus


class Compensator(BeamModifier):
    """
    Lớp đại diện cho một bộ bù liều (compensator).

    Compensator là thiết bị điều biến có độ dày thay đổi để cân bằng liều trong
    trường xạ, bù cho sự khác biệt về độ dày và mật độ của mô bệnh nhân.
    """

    def __init__(self, name: str, compensator_id: Optional[str] = None):
        """
        Khởi tạo một compensator.

        Parameters
        ----------
        name : str
            Tên của compensator
        compensator_id : str, optional
            ID duy nhất của compensator
        """
        super().__init__(name, ModifierType.COMPENSATOR, compensator_id)
        self.material = "Brass"  # Vật liệu: Brass, Aluminum, Lead, etc.
        self.density = 8.5  # Mật độ vật liệu (g/cm³)
        self.mount_distance = 15.0  # Khoảng cách từ compensator đến isocenter (cm)
        self.dimensions = (10.0, 10.0)  # Kích thước compensator (cm)
        self.resolution = (10, 10)  # Độ phân giải của lưới (pixels)
        self.thickness_map = np.zeros(self.resolution)  # Bản đồ độ dày (cm)
        self.max_thickness = 0.0  # Độ dày tối đa (cm)

    def set_dimensions(
        self, width: float, height: float, resolution_x: int, resolution_y: int
    ):
        """
        Thiết lập kích thước và độ phân giải của compensator.

        Parameters
        ----------
        width : float
            Chiều rộng compensator (cm)
        height : float
            Chiều cao compensator (cm)
        resolution_x : int
            Số điểm chia theo chiều X
        resolution_y : int
            Số điểm chia theo chiều Y
        """
        self.dimensions = (width, height)
        self.resolution = (resolution_x, resolution_y)
        self.thickness_map = np.zeros(self.resolution)

    def set_thickness_map(self, thickness_map: np.ndarray):
        """
        Thiết lập bản đồ độ dày cho compensator.

        Parameters
        ----------
        thickness_map : np.ndarray
            Mảng 2D chứa độ dày tại mỗi điểm (cm)
        """
        if thickness_map.shape != self.resolution:
            logger.error(
                f"Invalid thickness map shape: {thickness_map.shape}, expected {self.resolution}"
            )
            return

        self.thickness_map = thickness_map.copy()
        self.max_thickness = np.max(self.thickness_map)

    def set_material_properties(self, material: str, density: float):
        """
        Thiết lập thuộc tính vật liệu của compensator.

        Parameters
        ----------
        material : str
            Vật liệu của compensator
        density : float
            Mật độ của vật liệu (g/cm³)
        """
        self.material = material
        self.density = density

    def get_attenuation_factor(
        self, position: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Lấy hệ số suy giảm của compensator tại một vị trí.

        Parameters
        ----------
        position : Tuple[float, float], optional
            Vị trí cần tính hệ số suy giảm (cm)

        Returns
        -------
        float
            Hệ số suy giảm
        """
        if not self.is_active:
            return 1.0

        if position is None:
            # Trả về hệ số suy giảm trung bình
            mean_thickness = np.mean(self.thickness_map)
            return np.exp(-0.05 * mean_thickness * self.density)

        # Chuyển đổi vị trí thành chỉ số trong bản đồ độ dày
        width, height = self.dimensions
        res_x, res_y = self.resolution

        x, y = position
        x_idx = int((x + width / 2) / width * res_x)
        y_idx = int((y + height / 2) / height * res_y)

        # Kiểm tra giới hạn
        if x_idx < 0 or x_idx >= res_x or y_idx < 0 or y_idx >= res_y:
            return 1.0

        # Tính hệ số suy giảm dựa trên độ dày và vật liệu
        thickness = self.thickness_map[y_idx, x_idx]
        return np.exp(-0.05 * thickness * self.density)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin compensator thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin compensator
        """
        data = super().to_dict()
        data.update(
            {
                "material": self.material,
                "density": self.density,
                "mount_distance": self.mount_distance,
                "dimensions": self.dimensions,
                "resolution": self.resolution,
                "thickness_map": self.thickness_map.tolist(),
                "max_thickness": self.max_thickness,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Compensator":
        """
        Tạo đối tượng Compensator từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin compensator

        Returns
        -------
        Compensator
            Đối tượng Compensator
        """
        try:
            compensator = cls(name=data["name"], compensator_id=data["modifier_id"])
            compensator.is_active = data["is_active"]
            compensator.material = data.get("material", "Brass")
            compensator.density = data.get("density", 8.5)
            compensator.mount_distance = data.get("mount_distance", 15.0)
            compensator.dimensions = data.get("dimensions", (10.0, 10.0))
            compensator.resolution = data.get("resolution", (10, 10))

            if "thickness_map" in data:
                compensator.thickness_map = np.array(data["thickness_map"])
            else:
                compensator.thickness_map = np.zeros(compensator.resolution)

            compensator.max_thickness = data.get("max_thickness", 0.0)
            compensator.metadata = data.get("metadata", {})
            return compensator
        except KeyError as e:
            logger.error(f"Missing key in compensator data: {e}")
            # Trả về một compensator mặc định với dữ liệu tối thiểu
            return cls(data.get("name", "Unknown Compensator"))


class MLC(BeamModifier):
    """
    Lớp đại diện cho Multi-Leaf Collimator (MLC).

    MLC là hệ thống lá chắn kim loại có thể di chuyển để tạo ra
    các hình dạng trường chiếu phức tạp.
    """

    def __init__(self, name: str = "MLC", mlc_id: Optional[str] = None):
        """
        Khởi tạo MLC.

        Parameters
        ----------
        name : str
            Tên của MLC
        mlc_id : str, optional
            ID duy nhất của MLC
        """
        super().__init__(name, ModifierType.MLC, mlc_id)
        self.leaf_positions = {}  # Dictionary {leaf_number: position}
        self.leaf_width = 5.0  # mm
        self.total_leaves = 120
        self.leaf_travel = 150.0  # mm

    def set_leaf_positions(self, positions: Dict[int, Tuple[float, float]]):
        """Thiết lập vị trí các lá MLC."""
        self.leaf_positions = positions

    def get_leaf_positions(self) -> Dict[int, Tuple[float, float]]:
        """Lấy vị trí các lá MLC."""
        return self.leaf_positions

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin MLC thành dictionary."""
        data = super().to_dict()
        data.update(
            {
                "leaf_positions": self.leaf_positions,
                "leaf_width": self.leaf_width,
                "total_leaves": self.total_leaves,
                "leaf_travel": self.leaf_travel,
            }
        )
        return data


class Applicator(BeamModifier):
    """
    Lớp đại diện cho Applicator (ống chuẩn trực).

    Applicator được sử dụng để định nghĩa kích thước trường chiếu
    cho electron beam therapy.
    """

    def __init__(
        self,
        name: str = "Applicator",
        size: str = "10x10",
        applicator_id: Optional[str] = None,
    ):
        """
        Khởi tạo Applicator.

        Parameters
        ----------
        name : str
            Tên của applicator
        size : str
            Kích thước applicator (ví dụ: "10x10")
        applicator_id : str, optional
            ID duy nhất của applicator
        """
        super().__init__(
            name, ModifierType.WEDGE, applicator_id
        )  # Sử dụng WEDGE type tạm thời
        self.size = size
        self.opening_x = 100.0  # mm
        self.opening_y = 100.0  # mm
        self.thickness = 50.0  # mm

    def set_opening(self, x: float, y: float):
        """Thiết lập kích thước lỗ mở của applicator."""
        self.opening_x = x
        self.opening_y = y

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin applicator thành dictionary."""
        data = super().to_dict()
        data.update(
            {
                "size": self.size,
                "opening_x": self.opening_x,
                "opening_y": self.opening_y,
                "thickness": self.thickness,
            }
        )
        return data


class RangeShifter(BeamModifier):
    """
    Lớp đại diện cho Range Shifter.

    Range Shifter được sử dụng trong proton therapy để điều chỉnh
    phạm vi của chùm proton.
    """

    def __init__(
        self,
        name: str = "Range Shifter",
        thickness: float = 10.0,
        shifter_id: Optional[str] = None,
    ):
        """
        Khởi tạo Range Shifter.

        Parameters
        ----------
        name : str
            Tên của range shifter
        thickness : float
            Độ dày của range shifter (mm)
        shifter_id : str, optional
            ID duy nhất của range shifter
        """
        super().__init__(
            name, ModifierType.COMPENSATOR, shifter_id
        )  # Sử dụng COMPENSATOR type
        self.thickness = thickness
        self.material = "Acrylic"
        self.water_equivalent_thickness = thickness * 1.16  # WET for acrylic

    def set_thickness(self, thickness: float):
        """Thiết lập độ dày range shifter."""
        self.thickness = thickness
        self.water_equivalent_thickness = thickness * 1.16

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin range shifter thành dictionary."""
        data = super().to_dict()
        data.update(
            {
                "thickness": self.thickness,
                "material": self.material,
                "water_equivalent_thickness": self.water_equivalent_thickness,
            }
        )
        return data


class Filter(BeamModifier):
    """
    Lớp đại diện cho Beam Filter.

    Filter được sử dụng để điều chỉnh cường độ và phổ năng lượng
    của chùm tia.
    """

    def __init__(
        self,
        name: str = "Filter",
        material: str = "Lead",
        thickness: float = 1.0,
        filter_id: Optional[str] = None,
    ):
        """
        Khởi tạo Filter.

        Parameters
        ----------
        name : str
            Tên của filter
        material : str
            Vật liệu của filter
        thickness : float
            Độ dày của filter (mm)
        filter_id : str, optional
            ID duy nhất của filter
        """
        super().__init__(name, ModifierType.COMPENSATOR, filter_id)
        self.material = material
        self.thickness = thickness
        self.attenuation_coefficient = self._get_attenuation_coefficient()

    def _get_attenuation_coefficient(self) -> float:
        """Tính hệ số suy giảm dựa trên vật liệu."""
        material_coefficients = {
            "Lead": 0.15,
            "Aluminum": 0.05,
            "Copper": 0.08,
            "Tungsten": 0.20,
        }
        return material_coefficients.get(self.material, 0.1)

    def get_transmission_factor(self) -> float:
        """Tính hệ số truyền qua."""
        return np.exp(-self.attenuation_coefficient * self.thickness)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin filter thành dictionary."""
        data = super().to_dict()
        data.update(
            {
                "material": self.material,
                "thickness": self.thickness,
                "attenuation_coefficient": self.attenuation_coefficient,
            }
        )
        return data


class Collimator(BeamModifier):
    """
    Lớp đại diện cho Collimator (bộ chuẩn trực).

    Collimator được sử dụng để định nghĩa hình dạng và kích thước
    của trường chiếu.
    """

    def __init__(self, name: str = "Collimator", collimator_id: Optional[str] = None):
        """
        Khởi tạo Collimator.

        Parameters
        ----------
        name : str
            Tên của collimator
        collimator_id : str, optional
            ID duy nhất của collimator
        """
        super().__init__(name, ModifierType.BLOCK, collimator_id)
        self.jaw_x1 = -50.0  # mm
        self.jaw_x2 = 50.0  # mm
        self.jaw_y1 = -50.0  # mm
        self.jaw_y2 = 50.0  # mm
        self.rotation_angle = 0.0  # degrees

    def set_jaw_positions(self, x1: float, x2: float, y1: float, y2: float):
        """Thiết lập vị trí các jaw của collimator."""
        self.jaw_x1 = x1
        self.jaw_x2 = x2
        self.jaw_y1 = y1
        self.jaw_y2 = y2

    def set_rotation(self, angle: float):
        """Thiết lập góc xoay của collimator."""
        self.rotation_angle = angle

    def get_field_size(self) -> Tuple[float, float]:
        """Lấy kích thước trường chiếu."""
        width = abs(self.jaw_x2 - self.jaw_x1)
        height = abs(self.jaw_y2 - self.jaw_y1)
        return (width, height)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin collimator thành dictionary."""
        data = super().to_dict()
        data.update(
            {
                "jaw_x1": self.jaw_x1,
                "jaw_x2": self.jaw_x2,
                "jaw_y1": self.jaw_y1,
                "jaw_y2": self.jaw_y2,
                "rotation_angle": self.rotation_angle,
            }
        )
        return data


# Export classes
__all__ = [
    "ModifierType",
    "BeamModifier",
    "Wedge",
    "Block",
    "Bolus",
    "Compensator",
    "MLC",
    "Applicator",
    "RangeShifter",
    "Filter",
    "Collimator",
]
