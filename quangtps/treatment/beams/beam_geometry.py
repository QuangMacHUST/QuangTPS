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

        # Tính toán vị trí nguồn (để phục vụ ray tracing)
        self._calculate_source_position()

    def _calculate_transformation_matrix(self):
        """
        Tính toán ma trận chuyển đổi từ tọa độ bệnh nhân sang tọa độ BEV.

        Quy trình chuyển đổi tọa độ bao gồm các bước:
        1. Dịch chuyển hệ tọa độ để isocenter trở thành tâm
        2. Quay theo góc couch
        3. Quay theo góc gantry
        4. Quay theo góc collimator
        """
        # Chuyển các góc sang radian
        gantry_rad = np.radians(self.gantry_angle)
        collimator_rad = np.radians(self.collimator_angle)
        couch_rad = np.radians(self.couch_angle)

        # Tính ma trận quay cho gantry
        # Quay quanh trục Y (IEC 61217)
        cos_g = np.cos(gantry_rad)
        sin_g = np.sin(gantry_rad)
        R_gantry = np.array([[cos_g, 0, sin_g], [0, 1, 0], [-sin_g, 0, cos_g]])

        # Tính ma trận quay cho bàn điều trị
        # Quay quanh trục Z
        cos_c = np.cos(couch_rad)
        sin_c = np.sin(couch_rad)
        R_couch = np.array([[cos_c, -sin_c, 0], [sin_c, cos_c, 0], [0, 0, 1]])

        # Tính ma trận quay cho collimator
        # Quay quanh trục Z trong hệ tọa độ sau khi quay gantry
        cos_col = np.cos(collimator_rad)
        sin_col = np.sin(collimator_rad)
        R_collimator = np.array(
            [[cos_col, -sin_col, 0], [sin_col, cos_col, 0], [0, 0, 1]]
        )

        # Kết hợp các ma trận quay theo thứ tự: couch -> gantry -> collimator
        # Thứ tự nhân ma trận: R_final = R_collimator * R_gantry * R_couch
        self.rotation_matrix = np.matmul(np.matmul(R_collimator, R_gantry), R_couch)

        # Tính ma trận nghịch đảo cho phép biến đổi ngược
        self.inverse_rotation_matrix = np.linalg.inv(self.rotation_matrix)

        # Tính vị trí nguồn phát xạ trong hệ tọa độ bệnh nhân
        self.source_position = self._calculate_source_position()

        # Tính các ma trận riêng biệt cho ray tracing
        self.R_couch = R_couch
        self.R_gantry = R_gantry
        self.R_collimator = R_collimator

    def _calculate_source_position(self):
        """
        Tính toán vị trí nguồn chùm tia dựa trên các thông số góc và SAD.
        Cải thiện độ chính xác bằng cách xét cả ảnh hưởng của góc gantry, góc collimator và góc bàn.
        """
        # Chuyển góc sang radian
        gantry_rad = np.radians(self.gantry_angle)
        couch_rad = np.radians(self.couch_angle)

        # Tính tọa độ nguồn trong không gian IEC
        # Theo chuẩn IEC 61217, nguồn nằm ở phía trên trục Z khi gantry = 0
        # Khi gantry quay, nguồn di chuyển trong mặt phẳng X-Z
        x = -self.sad * np.sin(gantry_rad) * np.cos(couch_rad)
        y = -self.sad * np.sin(gantry_rad) * np.sin(couch_rad)
        z = -self.sad * np.cos(gantry_rad)

        # Vị trí nguồn (tương đối với isocenter)
        source_rel = np.array([x, y, z])

        # Vị trí nguồn tuyệt đối
        source_position = self.isocenter + source_rel

        # Đơn vị vector hướng chùm tia (từ nguồn đến isocenter)
        # Chùm tia luôn hướng từ nguồn đến isocenter
        beam_direction = self.isocenter - source_position
        beam_direction = beam_direction / np.linalg.norm(beam_direction)

        # Lưu thông tin để sử dụng sau này
        self.source_position = source_position
        self.beam_direction = beam_direction

        # Tính các trục chính của hệ tọa độ chùm tia
        # Trục Z là hướng chùm tia
        self.beam_z_axis = beam_direction

        # Tính trục X (ngang) và Y (dọc) của chùm tia
        # Trục X ban đầu nằm trong mặt phẳng X-Y của bệnh nhân
        # Sau đó quay theo góc collimator
        collimator_rad = np.radians(self.collimator_angle)

        # Tính trục X và Y cơ bản (chưa xoay collimator)
        if abs(abs(gantry_rad) - np.pi / 2) < 1e-6:  # Gantry = ±90°
            beam_x_base = np.array([0, 0, np.sign(np.sin(gantry_rad))])
        else:
            beam_x_base = np.array([1, 0, 0])

        # Đảm bảo beam_x_base vuông góc với beam_direction
        beam_x_base = beam_x_base - np.dot(beam_x_base, beam_direction) * beam_direction
        beam_x_base = beam_x_base / np.linalg.norm(beam_x_base)

        # Tính trục Y bằng cách lấy tích có hướng của Z và X
        beam_y_base = np.cross(beam_direction, beam_x_base)

        # Xoay trục X và Y theo góc collimator
        self.beam_x_axis = beam_x_base * np.cos(collimator_rad) + beam_y_base * np.sin(
            collimator_rad
        )
        self.beam_y_axis = -beam_x_base * np.sin(collimator_rad) + beam_y_base * np.cos(
            collimator_rad
        )

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

        # Kiểm tra nếu không có điểm
        if len(points_np) == 0:
            return np.zeros((0, 2))

        # Áp dụng phép dịch chuyển để đưa về tâm tại isocenter
        centered_points = points_np - self.isocenter

        # Áp dụng phép quay cho mỗi điểm
        # Sử dụng phép nhân ma trận cho vectorized performance
        rotated_points = np.dot(centered_points, self.rotation_matrix.T)

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

    def ray_trace_to_depth(
        self,
        bev_point: Union[Tuple[float, float], np.ndarray],
        structure: Any,
        max_distance: float = 50.0,
        step_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Phương thức truy vết tia từ nguồn phát xạ qua một điểm trong mặt phẳng BEV
        để xác định điểm vào, điểm ra và độ dày của cấu trúc.

        Parameters
        ----------
        bev_point : Union[Tuple[float, float], np.ndarray]
            Tọa độ điểm trong hệ tọa độ BEV (cm)
        structure : Any
            Cấu trúc cần xác định độ sâu
        max_distance : float, optional
            Khoảng cách tối đa để truy vết (cm), mặc định là 50.0 cm
        step_size : float, optional
            Kích thước bước dò tia (cm), mặc định là 0.2 cm

        Returns
        -------
        Dict[str, Any]
            Kết quả truy vết bao gồm:
            - has_intersection: bool - Có giao cắt với cấu trúc hay không
            - entry_point: np.ndarray - Tọa độ điểm vào cấu trúc
            - exit_point: np.ndarray - Tọa độ điểm ra cấu trúc
            - entry_depth: float - Độ sâu của điểm vào so với isocenter (cm)
            - exit_depth: float - Độ sâu của điểm ra so với isocenter (cm)
            - thickness: float - Độ dày của cấu trúc dọc theo tia (cm)
            - path_points: List[np.ndarray] - Danh sách các điểm trong đường đi của tia
        """
        # Kiểm tra cấu trúc có hợp lệ không
        if structure is None or not hasattr(structure, "contains_point"):
            return {
                "has_intersection": False,
                "entry_point": None,
                "exit_point": None,
                "entry_depth": np.nan,
                "exit_depth": np.nan,
                "thickness": 0.0,
                "path_points": [],
            }

        # Chuyển tọa độ BEV sang mặt phẳng isocentric trong hệ tọa độ bệnh nhân
        plane_point = self.inverse_transform_point(bev_point, depth=0.0)

        # Tính hướng của tia (từ nguồn đến điểm trong mặt phẳng isocentric)
        ray_direction = plane_point - self.source_position
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # Vị trí hiện tại (bắt đầu từ điểm nguồn phát xạ)
        current_position = self.source_position.copy()

        # Các biến để theo dõi quá trình dò tia
        inside_structure = False
        entry_point = None
        exit_point = None
        entry_depth = np.nan
        exit_depth = np.nan
        path_points = []

        # Truy vết tia từ nguồn theo hướng xác định
        total_distance = 0.0

        while total_distance < max_distance:
            # Lưu vị trí hiện tại
            path_points.append(current_position.copy())

            # Kiểm tra điểm hiện tại có nằm trong cấu trúc không
            is_inside = structure.contains_point(current_position)

            # Phát hiện chuyển trạng thái (vào/ra cấu trúc)
            if is_inside and not inside_structure:
                # Chuyển từ ngoài vào trong: ghi nhận điểm vào
                entry_point = current_position.copy()

                # Tính độ sâu của điểm vào
                vec_to_isocenter = self.isocenter - self.source_position
                vec_to_entry = entry_point - self.source_position

                # Chiếu vec_to_entry lên vec_to_isocenter để tính độ sâu so với isocenter
                sad_length = np.linalg.norm(vec_to_isocenter)
                projection = np.dot(vec_to_entry, vec_to_isocenter) / sad_length
                entry_depth = projection - sad_length

                inside_structure = True

            elif not is_inside and inside_structure:
                # Chuyển từ trong ra ngoài: ghi nhận điểm ra
                exit_point = current_position.copy()

                # Tính độ sâu của điểm ra
                vec_to_isocenter = self.isocenter - self.source_position
                vec_to_exit = exit_point - self.source_position

                # Chiếu vec_to_exit lên vec_to_isocenter để tính độ sâu so với isocenter
                sad_length = np.linalg.norm(vec_to_isocenter)
                projection = np.dot(vec_to_exit, vec_to_isocenter) / sad_length
                exit_depth = projection - sad_length

                # Đã tìm thấy cả điểm vào và điểm ra, có thể thoát vòng lặp
                break

            # Di chuyển dọc theo tia
            current_position += step_size * ray_direction
            total_distance += step_size

        # Tính độ dày của cấu trúc
        thickness = 0.0
        if inside_structure and exit_point is not None and entry_point is not None:
            thickness = np.linalg.norm(exit_point - entry_point)
        elif inside_structure and entry_point is not None:
            # Trường hợp tia đi vào cấu trúc nhưng chưa ra (đến giới hạn max_distance)
            thickness = np.linalg.norm(current_position - entry_point)
            exit_point = current_position.copy()

            # Tính độ sâu của điểm cuối
            vec_to_isocenter = self.isocenter - self.source_position
            vec_to_exit = exit_point - self.source_position
            sad_length = np.linalg.norm(vec_to_isocenter)
            projection = np.dot(vec_to_exit, vec_to_isocenter) / sad_length
            exit_depth = projection - sad_length

        return {
            "has_intersection": entry_point is not None,
            "entry_point": entry_point,
            "exit_point": exit_point,
            "entry_depth": entry_depth,
            "exit_depth": exit_depth,
            "thickness": thickness,
            "path_points": path_points,
        }

    def structure_to_bev_map(
        self,
        structure: Any,
        resolution: Tuple[int, int] = (256, 256),
        field_size: Tuple[float, float] = (20.0, 20.0),
        color_by_depth: bool = False,
        max_depth: float = 30.0,
    ) -> np.ndarray:
        """
        Chuyển đổi cấu trúc thành bản đồ BEV 2D.

        Parameters
        ----------
        structure : Any
            Cấu trúc cần chuyển đổi
        resolution : Tuple[int, int], optional
            Độ phân giải của bản đồ BEV (pixels), mặc định là (256, 256)
        field_size : Tuple[float, float], optional
            Kích thước trường chiếu (cm), mặc định là (20.0, 20.0)
        color_by_depth : bool, optional
            Nếu True, màu sắc sẽ thay đổi theo độ sâu của cấu trúc
        max_depth : float, optional
            Độ sâu tối đa (cm) để chuẩn hóa màu sắc, mặc định là 30.0

        Returns
        -------
        np.ndarray
            Nếu color_by_depth=False: Bản đồ BEV của cấu trúc (0-1)
            Nếu color_by_depth=True: Bản đồ BEV màu RGB (shape: [height, width, 3])
        """
        width, height = resolution
        x_field, y_field = field_size

        # Tạo bản đồ rỗng
        if color_by_depth:
            # Tạo bản đồ màu RGB
            bev_map = np.zeros((height, width, 3), dtype=float)
        else:
            # Tạo bản đồ đơn sắc
            bev_map = np.zeros(resolution, dtype=float)

        # Tính kích thước một pixel
        x_step = x_field / width
        y_step = y_field / height

        # Tính tọa độ BEV cho mỗi pixel
        for i in range(width):
            for j in range(height):
                # Tọa độ BEV của trung tâm pixel
                x_bev = (i - width / 2) * x_step + x_step / 2
                y_bev = (j - height / 2) * y_step + y_step / 2

                # Ray trace qua cấu trúc
                ray_result = self.ray_trace_to_depth((x_bev, y_bev), structure)

                # Nếu tia đi qua cấu trúc, đánh dấu pixel
                if ray_result["has_intersection"]:
                    if color_by_depth:
                        # Màu sắc dựa trên độ sâu
                        entry_depth = ray_result["entry_depth"]
                        thickness = ray_result["thickness"]

                        # Chuẩn hóa độ sâu vào khoảng [0, 1]
                        norm_depth = min(abs(entry_depth), max_depth) / max_depth
                        norm_thickness = min(thickness, max_depth / 2) / (max_depth / 2)

                        # Tạo màu dựa trên độ sâu: Đỏ -> Vàng -> Xanh lá
                        if entry_depth < 0:
                            # Phía trước isocenter: Đỏ -> Vàng
                            bev_map[j, i, 0] = 1.0  # R
                            bev_map[j, i, 1] = 1.0 - norm_depth  # G
                            bev_map[j, i, 2] = 0.0  # B
                        else:
                            # Phía sau isocenter: Vàng -> Xanh lá
                            bev_map[j, i, 0] = 1.0 - norm_depth  # R
                            bev_map[j, i, 1] = 1.0  # G
                            bev_map[j, i, 2] = 0.0  # B

                        # Điều chỉnh độ đậm dựa trên độ dày
                        alpha = min(0.3 + 0.7 * norm_thickness, 1.0)
                        bev_map[j, i] *= alpha
                    else:
                        # Đánh dấu 1 cho mọi điểm giao
                        bev_map[j, i] = 1.0

        return bev_map

    def structure_to_bev_depth_map(
        self,
        structure: Any,
        resolution: Tuple[int, int] = (256, 256),
        field_size: Tuple[float, float] = (20.0, 20.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo bản đồ độ sâu cho cấu trúc trong BEV.

        Parameters
        ----------
        structure : Any
            Cấu trúc cần chuyển đổi
        resolution : Tuple[int, int], optional
            Độ phân giải của bản đồ BEV (pixels), mặc định là (256, 256)
        field_size : Tuple[float, float], optional
            Kích thước trường chiếu (cm), mặc định là (20.0, 20.0)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Trả về một tuple gồm:
            - depth_map: Bản đồ độ sâu của điểm vào (cm)
            - thickness_map: Bản đồ độ dày của cấu trúc (cm)
        """
        width, height = resolution
        x_field, y_field = field_size

        # Tạo bản đồ rỗng
        depth_map = np.full(resolution, np.nan, dtype=float)
        thickness_map = np.zeros(resolution, dtype=float)

        # Tính kích thước một pixel
        x_step = x_field / width
        y_step = y_field / height

        # Tính tọa độ BEV cho mỗi pixel
        for i in range(width):
            for j in range(height):
                # Tọa độ BEV của trung tâm pixel
                x_bev = (i - width / 2) * x_step + x_step / 2
                y_bev = (j - height / 2) * y_step + y_step / 2

                # Ray trace qua cấu trúc
                ray_result = self.ray_trace_to_depth((x_bev, y_bev), structure)

                # Nếu tia đi qua cấu trúc, lưu thông tin độ sâu
                if ray_result["has_intersection"]:
                    depth_map[j, i] = ray_result["entry_depth"]
                    thickness_map[j, i] = ray_result["thickness"]

        return depth_map, thickness_map


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
