#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý hình học của chùm tia xạ trị (Beam Geometry).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý
hình học của chùm tia xạ trị, bao gồm các thông số góc, khoảng cách,
và điểm tham chiếu.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)


class CoordinateSystem(str, Enum):
    """Enum đại diện cho các hệ tọa độ."""

    PATIENT = "PATIENT"  # Hệ tọa độ bệnh nhân
    MACHINE = "MACHINE"  # Hệ tọa độ máy
    BEV = "BEV"  # Hệ tọa độ Beam's Eye View
    IEC = "IEC"  # Hệ tọa độ IEC
    DICOM = "DICOM"  # Hệ tọa độ DICOM


class GantryDirection(str, Enum):
    """Enum đại diện cho hướng quay của gantry."""

    CW = "CW"  # Theo chiều kim đồng hồ (Clockwise)
    CCW = "CCW"  # Ngược chiều kim đồng hồ (Counter-Clockwise)


class CollimatorDirection(str, Enum):
    """Enum đại diện cho hướng quay của collimator."""

    CW = "CW"  # Theo chiều kim đồng hồ (Clockwise)
    CCW = "CCW"  # Ngược chiều kim đồng hồ (Counter-Clockwise)


class CouchDirection(str, Enum):
    """Enum đại diện cho hướng quay của bàn điều trị."""

    CW = "CW"  # Theo chiều kim đồng hồ (Clockwise)
    CCW = "CCW"  # Ngược chiều kim đồng hồ (Counter-Clockwise)


class BeamGeometry:
    """
    Lớp đại diện cho hình học của một chùm tia xạ trị.

    Lớp này chứa thông tin về hình học của chùm tia xạ trị, bao gồm các
    thông số góc, khoảng cách, và điểm tham chiếu.
    """

    def __init__(self):
        """
        Khởi tạo một đối tượng BeamGeometry với các giá trị mặc định.
        """
        # Các góc
        self.gantry_angle = 0.0  # Góc gantry (độ)
        self.gantry_direction = GantryDirection.CW

        self.collimator_angle = 0.0  # Góc collimator (độ)
        self.collimator_direction = CollimatorDirection.CW

        self.couch_angle = 0.0  # Góc bàn điều trị (độ)
        self.couch_direction = CouchDirection.CW

        # Khoảng cách
        self.source_surface_distance = (
            100.0  # Khoảng cách từ nguồn đến bề mặt (SSD) (cm)
        )
        self.source_axis_distance = (
            100.0  # Khoảng cách từ nguồn đến trục quay (SAD) (cm)
        )

        # Tọa độ isocenter
        self.isocenter = (0.0, 0.0, 0.0)  # x, y, z (cm)

        # Kích thước trường
        self.field_size = (10.0, 10.0)  # Kích thước trường tại isocenter (cm)
        self.effective_field_size = (10.0, 10.0)  # Kích thước trường hiệu dụng (cm)

        # Dịch chuyển collimator
        self.collimator_x1 = -5.0  # Vị trí cạnh X1 của collimator (cm)
        self.collimator_x2 = 5.0  # Vị trí cạnh X2 của collimator (cm)
        self.collimator_y1 = -5.0  # Vị trí cạnh Y1 của collimator (cm)
        self.collimator_y2 = 5.0  # Vị trí cạnh Y2 của collimator (cm)

        # Thông tin bổ sung
        self.metadata = {}

    def set_gantry_angle(
        self, angle: float, direction: GantryDirection = GantryDirection.CW
    ):
        """
        Thiết lập góc gantry.

        Parameters
        ----------
        angle : float
            Góc gantry (độ)
        direction : GantryDirection, optional
            Hướng quay của gantry
        """
        self.gantry_angle = angle
        self.gantry_direction = direction

    def set_collimator_angle(
        self, angle: float, direction: CollimatorDirection = CollimatorDirection.CW
    ):
        """
        Thiết lập góc collimator.

        Parameters
        ----------
        angle : float
            Góc collimator (độ)
        direction : CollimatorDirection, optional
            Hướng quay của collimator
        """
        self.collimator_angle = angle
        self.collimator_direction = direction

    def set_couch_angle(
        self, angle: float, direction: CouchDirection = CouchDirection.CW
    ):
        """
        Thiết lập góc bàn điều trị.

        Parameters
        ----------
        angle : float
            Góc bàn điều trị (độ)
        direction : CouchDirection, optional
            Hướng quay của bàn điều trị
        """
        self.couch_angle = angle
        self.couch_direction = direction

    def set_isocenter(self, x: float, y: float, z: float):
        """
        Thiết lập tọa độ isocenter.

        Parameters
        ----------
        x : float
            Tọa độ x của isocenter (cm)
        y : float
            Tọa độ y của isocenter (cm)
        z : float
            Tọa độ z của isocenter (cm)
        """
        self.isocenter = (x, y, z)

    def set_field_size(self, width: float, height: float):
        """
        Thiết lập kích thước trường tại isocenter.

        Parameters
        ----------
        width : float
            Chiều rộng trường (cm)
        height : float
            Chiều cao trường (cm)
        """
        self.field_size = (width, height)
        self.collimator_x1 = -width / 2
        self.collimator_x2 = width / 2
        self.collimator_y1 = -height / 2
        self.collimator_y2 = height / 2
        self.effective_field_size = (width, height)

    def set_collimator_positions(self, x1: float, x2: float, y1: float, y2: float):
        """
        Thiết lập vị trí các cạnh của collimator.

        Parameters
        ----------
        x1 : float
            Vị trí cạnh X1 của collimator (cm)
        x2 : float
            Vị trí cạnh X2 của collimator (cm)
        y1 : float
            Vị trí cạnh Y1 của collimator (cm)
        y2 : float
            Vị trí cạnh Y2 của collimator (cm)
        """
        self.collimator_x1 = x1
        self.collimator_x2 = x2
        self.collimator_y1 = y1
        self.collimator_y2 = y2
        self.field_size = (abs(x2 - x1), abs(y2 - y1))
        self.effective_field_size = self.field_size

    def set_source_surface_distance(self, ssd: float):
        """
        Thiết lập khoảng cách từ nguồn đến bề mặt (SSD).

        Parameters
        ----------
        ssd : float
            Khoảng cách từ nguồn đến bề mặt (cm)
        """
        self.source_surface_distance = ssd

    def set_source_axis_distance(self, sad: float):
        """
        Thiết lập khoảng cách từ nguồn đến trục quay (SAD).

        Parameters
        ----------
        sad : float
            Khoảng cách từ nguồn đến trục quay (cm)
        """
        self.source_axis_distance = sad

    def get_beam_eye_view_coordinates(
        self, point: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """
        Chuyển đổi tọa độ từ hệ tọa độ bệnh nhân sang hệ tọa độ Beam's Eye View (BEV).

        Parameters
        ----------
        point : Tuple[float, float, float]
            Tọa độ điểm trong hệ tọa độ bệnh nhân (cm)

        Returns
        -------
        Tuple[float, float]
            Tọa độ điểm trong hệ tọa độ BEV (cm)
        """
        # Tính toán vector từ isocenter đến điểm
        x, y, z = point
        iso_x, iso_y, iso_z = self.isocenter
        dx = x - iso_x
        dy = y - iso_y
        dz = z - iso_z

        # Tính toán góc trong hệ tọa độ radian
        gantry_rad = np.radians(self.gantry_angle)
        collimator_rad = np.radians(self.collimator_angle)
        couch_rad = np.radians(self.couch_angle)

        # Áp dụng phép biến đổi cho góc gantry
        dx_gantry = dx * np.cos(gantry_rad) + dz * np.sin(gantry_rad)
        dy_gantry = dy
        dz_gantry = -dx * np.sin(gantry_rad) + dz * np.cos(gantry_rad)

        # Áp dụng phép biến đổi cho góc collimator
        dx_collimator = dx_gantry * np.cos(collimator_rad) - dy_gantry * np.sin(
            collimator_rad
        )
        dy_collimator = dx_gantry * np.sin(collimator_rad) + dy_gantry * np.cos(
            collimator_rad
        )

        # Tính toán tọa độ BEV
        magnification = self.source_axis_distance / (
            self.source_axis_distance - dz_gantry
        )
        bev_x = dx_collimator * magnification
        bev_y = dy_collimator * magnification

        return (bev_x, bev_y)

    def get_source_position(self) -> Tuple[float, float, float]:
        """
        Lấy tọa độ của nguồn trong hệ tọa độ bệnh nhân.

        Returns
        -------
        Tuple[float, float, float]
            Tọa độ của nguồn (cm)
        """
        iso_x, iso_y, iso_z = self.isocenter

        # Tính toán góc trong hệ tọa độ radian
        gantry_rad = np.radians(self.gantry_angle)

        # Tính toán vị trí nguồn dựa trên góc gantry và SAD
        source_x = iso_x - self.source_axis_distance * np.sin(gantry_rad)
        source_y = iso_y
        source_z = iso_z - self.source_axis_distance * np.cos(gantry_rad)

        return (source_x, source_y, source_z)

    def calculate_effective_field_size(
        self, modifiers: List[Any] = None
    ) -> Tuple[float, float]:
        """
        Tính toán kích thước trường hiệu dụng sau khi áp dụng các bộ điều chỉnh.

        Parameters
        ----------
        modifiers : List[Any], optional
            Danh sách các bộ điều chỉnh

        Returns
        -------
        Tuple[float, float]
            Kích thước trường hiệu dụng (cm)
        """
        if modifiers is None:
            return self.field_size

        width, height = self.field_size

        # Tính toán ảnh hưởng của các bộ điều chỉnh
        for modifier in modifiers:
            if hasattr(modifier, "get_effective_size_change"):
                mod_width, mod_height = modifier.get_effective_size_change()
                width = max(0, width - mod_width)
                height = max(0, height - mod_height)

        self.effective_field_size = (width, height)
        return self.effective_field_size

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin hình học chùm tia thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin hình học chùm tia
        """
        return {
            "gantry_angle": self.gantry_angle,
            "gantry_direction": self.gantry_direction.value,
            "collimator_angle": self.collimator_angle,
            "collimator_direction": self.collimator_direction.value,
            "couch_angle": self.couch_angle,
            "couch_direction": self.couch_direction.value,
            "source_surface_distance": self.source_surface_distance,
            "source_axis_distance": self.source_axis_distance,
            "isocenter": self.isocenter,
            "field_size": self.field_size,
            "effective_field_size": self.effective_field_size,
            "collimator_x1": self.collimator_x1,
            "collimator_x2": self.collimator_x2,
            "collimator_y1": self.collimator_y1,
            "collimator_y2": self.collimator_y2,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamGeometry":
        """
        Tạo đối tượng BeamGeometry từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin hình học chùm tia

        Returns
        -------
        BeamGeometry
            Đối tượng BeamGeometry
        """
        geometry = cls()

        # Cập nhật các thuộc tính
        geometry.gantry_angle = data["gantry_angle"]
        geometry.gantry_direction = GantryDirection(data["gantry_direction"])

        geometry.collimator_angle = data["collimator_angle"]
        geometry.collimator_direction = CollimatorDirection(
            data["collimator_direction"]
        )

        geometry.couch_angle = data["couch_angle"]
        geometry.couch_direction = CouchDirection(data["couch_direction"])

        geometry.source_surface_distance = data["source_surface_distance"]
        geometry.source_axis_distance = data["source_axis_distance"]

        geometry.isocenter = data["isocenter"]
        geometry.field_size = data["field_size"]
        geometry.effective_field_size = data["effective_field_size"]

        geometry.collimator_x1 = data["collimator_x1"]
        geometry.collimator_x2 = data["collimator_x2"]
        geometry.collimator_y1 = data["collimator_y1"]
        geometry.collimator_y2 = data["collimator_y2"]

        geometry.metadata = data.get("metadata", {})

        return geometry


class BEVTransform:
    """
    Lớp thực hiện chuyển đổi tọa độ từ hệ tọa độ bệnh nhân sang hệ tọa độ Beam's Eye View (BEV).

    Lớp này cung cấp các phương thức để chuyển đổi điểm và tập hợp điểm từ không gian 3D
    của bệnh nhân sang mặt phẳng 2D của góc nhìn chùm tia, phục vụ cho việc hiển thị BEV
    và tạo hình MLC.
    """

    def __init__(
        self,
        gantry_angle: float,
        collimator_angle: float,
        couch_angle: float,
        isocenter: Tuple[float, float, float],
        sad: float = 100.0,
    ):
        """
        Khởi tạo đối tượng BEVTransform với các tham số góc và isocenter.

        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
        collimator_angle : float
            Góc collimator (độ)
        couch_angle : float
            Góc bàn điều trị (độ)
        isocenter : Tuple[float, float, float]
            Tọa độ isocenter (cm)
        sad : float, optional
            Khoảng cách từ nguồn đến trục quay (cm), mặc định là 100.0 cm
        """
        # Lưu các thông số
        self.gantry_angle = gantry_angle
        self.collimator_angle = collimator_angle
        self.couch_angle = couch_angle
        self.isocenter = np.array(isocenter, dtype=float)
        self.sad = sad

        # Tính toán ma trận chuyển đổi
        self._calculate_transformation_matrix()

    def _calculate_transformation_matrix(self):
        """
        Tính toán ma trận chuyển đổi từ tọa độ bệnh nhân sang tọa độ BEV.
        """
        # Chuyển các góc sang radian
        gantry_rad = np.radians(self.gantry_angle)
        collimator_rad = np.radians(self.collimator_angle)
        couch_rad = np.radians(self.couch_angle)

        # Tính ma trận quay cho gantry
        cos_g = np.cos(gantry_rad)
        sin_g = np.sin(gantry_rad)
        R_gantry = np.array([[cos_g, 0, sin_g], [0, 1, 0], [-sin_g, 0, cos_g]])

        # Tính ma trận quay cho bàn điều trị
        cos_c = np.cos(couch_rad)
        sin_c = np.sin(couch_rad)
        R_couch = np.array([[cos_c, -sin_c, 0], [sin_c, cos_c, 0], [0, 0, 1]])

        # Tính ma trận quay cho collimator
        cos_col = np.cos(collimator_rad)
        sin_col = np.sin(collimator_rad)
        R_collimator = np.array(
            [[cos_col, -sin_col, 0], [sin_col, cos_col, 0], [0, 0, 1]]
        )

        # Kết hợp các ma trận quay
        # Thứ tự quay: trước tiên là couch, sau đó là gantry, cuối cùng là collimator
        self.rotation_matrix = np.matmul(np.matmul(R_collimator, R_gantry), R_couch)

        # Tính ma trận nghịch đảo cho phép biến đổi ngược
        self.inverse_rotation_matrix = np.linalg.inv(self.rotation_matrix)

        # Tính vị trí nguồn phát xạ trong hệ tọa độ bệnh nhân
        self.source_position = self._calculate_source_position()

    def _calculate_source_position(self) -> np.ndarray:
        """
        Tính vị trí nguồn phát xạ trong hệ tọa độ bệnh nhân.

        Returns
        -------
        np.ndarray
            Tọa độ nguồn phát xạ (cm)
        """
        # Chuyển góc gantry sang radian
        gantry_rad = np.radians(self.gantry_angle)
        couch_rad = np.radians(self.couch_angle)

        # Tính toán vị trí nguồn theo công thức
        x = -self.sad * np.sin(gantry_rad) * np.cos(couch_rad)
        y = -self.sad * np.sin(gantry_rad) * np.sin(couch_rad)
        z = -self.sad * np.cos(gantry_rad)

        # Vị trí nguồn phát xạ tương đối với isocenter
        source_relative = np.array([x, y, z])

        # Vị trí nguồn phát xạ trong hệ tọa độ bệnh nhân
        source_position = self.isocenter + source_relative

        return source_position

    def transform_point(
        self, point: Union[Tuple[float, float, float], np.ndarray]
    ) -> np.ndarray:
        """
        Chuyển đổi một điểm từ tọa độ bệnh nhân sang tọa độ BEV.

        Parameters
        ----------
        point : Union[Tuple[float, float, float], np.ndarray]
            Điểm trong hệ tọa độ bệnh nhân (cm)

        Returns
        -------
        np.ndarray
            Điểm trong hệ tọa độ BEV (cm). Giá trị trả về là một mảng 2 chiều
            [x, y] trong mặt phẳng BEV, với:
            - x: Tọa độ dọc theo hướng collimator X
            - y: Tọa độ dọc theo hướng collimator Y
        """
        point_np = np.array(point, dtype=float)

        # Vector từ isocenter đến điểm
        v_point = point_np - self.isocenter

        # Áp dụng phép quay
        rotated_point = np.matmul(self.rotation_matrix, v_point)

        # Lấy tọa độ x, y trong mặt phẳng BEV (z là dọc theo trục chùm tia)
        return rotated_point[:2]

    def transform_points(
        self, points: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> np.ndarray:
        """
        Chuyển đổi nhiều điểm từ tọa độ bệnh nhân sang tọa độ BEV.

        Parameters
        ----------
        points : Union[List[Tuple[float, float, float]], np.ndarray]
            Danh sách các điểm trong hệ tọa độ bệnh nhân (cm)

        Returns
        -------
        np.ndarray
            Mảng các điểm trong hệ tọa độ BEV (cm), mỗi điểm là [x, y]
        """
        points_np = np.array(points, dtype=float)

        # Áp dụng phép dịch chuyển để đưa về tâm tại isocenter
        centered_points = points_np - self.isocenter

        # Áp dụng phép quay cho mỗi điểm
        rotated_points = np.zeros((len(points_np), 3))
        for i, point in enumerate(centered_points):
            rotated_points[i] = np.matmul(self.rotation_matrix, point)

        # Trả về các tọa độ x, y trong mặt phẳng BEV
        return rotated_points[:, :2]

    def inverse_transform_point(
        self, bev_point: Union[Tuple[float, float], np.ndarray], depth: float = 0.0
    ) -> np.ndarray:
        """
        Chuyển đổi ngược từ tọa độ BEV sang tọa độ bệnh nhân.

        Parameters
        ----------
        bev_point : Union[Tuple[float, float], np.ndarray]
            Điểm trong hệ tọa độ BEV (cm)
        depth : float, optional
            Độ sâu dọc theo trục chùm tia, tính từ isocenter (cm),
            giá trị mặc định là 0.0 (mặt phẳng isocentric)

        Returns
        -------
        np.ndarray
            Điểm trong hệ tọa độ bệnh nhân (cm)
        """
        bev_point_np = np.array(bev_point, dtype=float)

        # Tạo điểm 3D bằng cách thêm tọa độ z (độ sâu)
        bev_point_3d = np.array([bev_point_np[0], bev_point_np[1], depth])

        # Áp dụng phép quay nghịch đảo
        rotated_point = np.matmul(self.inverse_rotation_matrix, bev_point_3d)

        # Dịch chuyển về hệ tọa độ bệnh nhân
        patient_point = rotated_point + self.isocenter

        return patient_point

    def inverse_transform_points(
        self,
        bev_points: Union[List[Tuple[float, float]], np.ndarray],
        depths: Union[float, List[float], np.ndarray] = 0.0,
    ) -> np.ndarray:
        """
        Chuyển đổi ngược nhiều điểm từ tọa độ BEV sang tọa độ bệnh nhân.

        Parameters
        ----------
        bev_points : Union[List[Tuple[float, float]], np.ndarray]
            Danh sách các điểm trong hệ tọa độ BEV (cm)
        depths : Union[float, List[float], np.ndarray], optional
            Độ sâu hoặc danh sách độ sâu dọc theo trục chùm tia cho mỗi điểm (cm),
            giá trị mặc định là 0.0 (mặt phẳng isocentric)

        Returns
        -------
        np.ndarray
            Mảng các điểm trong hệ tọa độ bệnh nhân (cm), mỗi điểm là [x, y, z]
        """
        bev_points_np = np.array(bev_points, dtype=float)

        # Nếu depths là một số, tạo mảng có cùng kích thước với số điểm BEV
        if isinstance(depths, (int, float)):
            depths = np.full(len(bev_points_np), depths)
        else:
            depths = np.array(depths, dtype=float)

        # Đảm bảo depths có đúng kích thước
        if len(depths) != len(bev_points_np):
            raise ValueError("Số lượng độ sâu phải bằng số lượng điểm BEV")

        # Áp dụng phép chuyển đổi nghịch đảo cho mỗi điểm
        patient_points = np.zeros((len(bev_points_np), 3))
        for i, (bev_point, depth) in enumerate(zip(bev_points_np, depths)):
            patient_points[i] = self.inverse_transform_point(bev_point, depth)

        return patient_points

    def calculate_magnification(self, z_distance: float) -> float:
        """
        Tính hệ số phóng đại tại một khoảng cách z dọc theo trục chùm tia.

        Parameters
        ----------
        z_distance : float
            Khoảng cách dọc theo trục chùm tia từ isocenter (cm)

        Returns
        -------
        float
            Hệ số phóng đại
        """
        # SAD là khoảng cách từ nguồn đến isocenter
        # Hệ số phóng đại = SAD / (SAD - z)
        return self.sad / (self.sad - z_distance)

    def project_to_isocentric_plane(
        self, point: Union[Tuple[float, float, float], np.ndarray]
    ) -> np.ndarray:
        """
        Chiếu một điểm từ không gian 3D lên mặt phẳng isocentric trong tọa độ BEV.

        Parameters
        ----------
        point : Union[Tuple[float, float, float], np.ndarray]
            Điểm trong hệ tọa độ bệnh nhân (cm)

        Returns
        -------
        np.ndarray
            Điểm đã chiếu trong hệ tọa độ BEV (cm)
        """
        point_np = np.array(point, dtype=float)

        # Vector từ nguồn đến điểm
        v_source_to_point = point_np - self.source_position

        # Vector từ nguồn đến isocenter
        v_source_to_iso = self.isocenter - self.source_position

        # Tỷ lệ để chiếu lên mặt phẳng isocentric
        # Mặt phẳng isocentric vuông góc với v_source_to_iso
        iso_direction = v_source_to_iso / np.linalg.norm(v_source_to_iso)

        # Tính tỷ lệ dựa trên công thức chiếu
        ratio = np.dot(v_source_to_iso, iso_direction) / np.dot(
            v_source_to_point, iso_direction
        )

        # Điểm được chiếu trong hệ tọa độ bệnh nhân
        projected_point = self.source_position + ratio * v_source_to_point

        # Chuyển đổi sang tọa độ BEV
        return self.transform_point(projected_point)

    def ray_trace_to_depth(
        self,
        bev_point: Union[Tuple[float, float], np.ndarray],
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> float:
        """
        Thực hiện ray tracing từ nguồn qua một điểm BEV để tìm độ sâu đầu tiên gặp cấu trúc.

        Parameters
        ----------
        bev_point : Union[Tuple[float, float], np.ndarray]
            Điểm trong hệ tọa độ BEV (cm)
        volume : np.ndarray
            Mảng 3D biểu diễn thể tích cần dò tia
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các voxel (cm)
        origin : Tuple[float, float, float]
            Tọa độ gốc của thể tích trong hệ tọa độ bệnh nhân (cm)

        Returns
        -------
        float
            Độ sâu từ isocenter đến điểm đầu tiên gặp cấu trúc (cm)
            Nếu không gặp cấu trúc nào, trả về np.inf
        """
        # Tìm tia từ nguồn đi qua điểm BEV
        # Đầu tiên chuyển điểm BEV thành điểm trên mặt phẳng isocentric
        isocentric_point = self.inverse_transform_point(bev_point)

        # Tính vector hướng của tia (từ nguồn đến điểm isocentric)
        ray_direction = isocentric_point - self.source_position
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # Thực hiện ray tracing với bước nhỏ (ví dụ 1mm)
        step_size = min(spacing) / 2.0  # Nửa khoảng cách nhỏ nhất giữa các voxel
        max_distance = 50.0  # Khoảng cách tối đa để tìm kiếm (cm)

        current_point = self.source_position
        current_distance = 0.0

        while current_distance < max_distance:
            # Di chuyển dọc theo tia
            current_point = current_point + step_size * ray_direction
            current_distance += step_size

            # Chuyển từ tọa độ bệnh nhân sang chỉ số voxel
            voxel_index = [
                int((current_point[i] - origin[i]) / spacing[i]) for i in range(3)
            ]

            # Kiểm tra xem voxel có trong volume không
            if (
                0 <= voxel_index[0] < volume.shape[0]
                and 0 <= voxel_index[1] < volume.shape[1]
                and 0 <= voxel_index[2] < volume.shape[2]
            ):
                # Kiểm tra xem voxel có giá trị > 0 không (đại diện cho cấu trúc)
                if volume[voxel_index[0], voxel_index[1], voxel_index[2]] > 0:
                    # Tính độ sâu từ isocenter
                    # Vector từ isocenter đến điểm gặp
                    v_iso_to_point = current_point - self.isocenter

                    # Chiếu lên trục chùm tia (z-axis trong hệ tọa độ BEV)
                    # Lấy trục z từ ma trận quay
                    beam_axis = self.rotation_matrix[2, :]

                    # Độ sâu là độ dài chiếu của vector lên trục chùm tia
                    depth = np.dot(v_iso_to_point, beam_axis)

                    return depth

        # Không tìm thấy giao điểm
        return np.inf


def get_bev_transform(beam) -> BEVTransform:
    """
    Tạo đối tượng BEVTransform từ đối tượng chùm tia.

    Parameters
    ----------
    beam : object
        Đối tượng chùm tia có chứa thông tin về góc và isocenter

    Returns
    -------
    BEVTransform
        Đối tượng thực hiện chuyển đổi tọa độ sang BEV
    """
    # Lấy các thông số từ chùm tia
    gantry_angle = getattr(beam, "gantry_angle", 0.0)
    collimator_angle = getattr(beam, "collimator_angle", 0.0)
    couch_angle = getattr(beam, "couch_angle", 0.0)

    # Lấy isocenter
    if hasattr(beam, "isocenter"):
        isocenter = beam.isocenter
    else:
        isocenter = (0.0, 0.0, 0.0)

    # Lấy SAD
    sad = getattr(beam, "sad", 100.0)

    # Tạo và trả về đối tượng BEVTransform
    return BEVTransform(
        gantry_angle=gantry_angle,
        collimator_angle=collimator_angle,
        couch_angle=couch_angle,
        isocenter=isocenter,
        sad=sad,
    )
