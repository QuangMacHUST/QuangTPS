#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module mô hình chùm tia.

Module này cung cấp các lớp và phương thức để mô tả và xử lý
các mô hình chùm tia trong QuangTPS.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BeamEnergyType(Enum):
    """Các loại năng lượng chùm tia."""

    PHOTON = auto()  # Photon (X-rays)
    ELECTRON = auto()  # Electron
    PROTON = auto()  # Proton
    CARBON = auto()  # Carbon ion
    FFF = auto()  # Flattening Filter Free


class BeamModelType(Enum):
    """Các loại mô hình chùm tia."""

    VARIAN = auto()  # Mô hình Varian
    ELEKTA = auto()  # Mô hình Elekta
    SIEMENS = auto()  # Mô hình Siemens
    ACCURAY = auto()  # Mô hình Accuray (TomoTherapy, CyberKnife)
    GENERIC = auto()  # Mô hình chung
    CUSTOM = auto()  # Mô hình tùy chỉnh


@dataclass
class BeamModelParameters:
    """Các tham số cho mô hình chùm tia."""

    # Thông tin cơ bản
    name: str = "Generic Beam Model"
    model_type: BeamModelType = BeamModelType.GENERIC
    energy_type: BeamEnergyType = BeamEnergyType.PHOTON
    energy_value: float = 6.0  # MV hoặc MeV

    # Tham số vật lý
    dose_rate: float = 600.0  # MU/min
    sad: float = 100.0  # Source-to-axis distance (cm)
    max_field_size: Tuple[float, float] = (40.0, 40.0)  # cm
    penumbra_width: float = 0.7  # cm

    # Tham số hiệu suất
    output_factor: float = 1.0
    transmission_factor: float = 0.015  # For MLC

    # PDD/TMR dữ liệu
    pdd_data: Dict[float, List[float]] = field(default_factory=dict)
    tmr_data: Dict[float, List[float]] = field(default_factory=dict)

    # Off-axis ratio
    oar_data: Dict[Tuple[float, float], List[float]] = field(default_factory=dict)

    # Thông tin bổ sung
    description: str = ""
    source_file: str = ""
    is_commissioned: bool = False
    commission_date: str = ""


class BeamModel:
    """
    Lớp mô tả mô hình chùm tia.

    Lớp này chứa tất cả thông tin cần thiết để mô tả và mô phỏng
    một chùm tia từ máy xạ trị, bao gồm các tham số vật lý và
    dữ liệu đo lường.
    """

    def __init__(
        self,
        parameters: Optional[BeamModelParameters] = None,
        data_file: Optional[str] = None,
    ):
        """
        Khởi tạo mô hình chùm tia.

        Args:
            parameters: Các tham số mô hình chùm tia
            data_file: Đường dẫn đến file dữ liệu chùm tia
        """
        self.parameters = parameters or BeamModelParameters()

        # Khởi tạo dữ liệu
        self.profiles = {}  # Beam profiles
        self.wedge_factors = {}  # Wedge factors
        self.scatter_factors = {}  # Scatter factors

        # Tải dữ liệu nếu được chỉ định
        if data_file:
            self.load_from_file(data_file)

    @property
    def name(self) -> str:
        """Tên của mô hình chùm tia."""
        return self.parameters.name

    @property
    def energy(self) -> str:
        """Năng lượng chùm tia dưới dạng chuỗi."""
        if self.parameters.energy_type == BeamEnergyType.FFF:
            return f"{self.parameters.energy_value}FFF"
        else:
            unit = (
                "MV" if self.parameters.energy_type == BeamEnergyType.PHOTON else "MeV"
            )
            return f"{self.parameters.energy_value}{unit}"

    def load_from_file(self, file_path: str) -> bool:
        """
        Tải dữ liệu chùm tia từ file.

        Args:
            file_path: Đường dẫn đến file dữ liệu

        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            logger.info(f"Đang tải dữ liệu chùm tia từ {file_path}")
            # TODO: Triển khai việc đọc dữ liệu từ file
            # Đây sẽ phụ thuộc vào định dạng file
            self.parameters.source_file = file_path
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu chùm tia: {e}")
            return False

    def get_pdd(
        self, depth: float, field_size: Tuple[float, float] = (10.0, 10.0)
    ) -> float:
        """
        Lấy giá trị PDD (Percentage Depth Dose) cho độ sâu và kích thước trường xác định.

        Args:
            depth: Độ sâu (cm)
            field_size: Kích thước trường (cm)

        Returns:
            Giá trị PDD (%)
        """
        # Đơn giản hóa bằng cách chỉ sử dụng kích thước trường trung bình
        avg_field_size = (field_size[0] + field_size[1]) / 2

        # Tìm kích thước trường gần nhất trong dữ liệu
        available_sizes = list(self.parameters.pdd_data.keys())
        if not available_sizes:
            return 100.0 * np.exp(-0.04 * depth)  # Mô hình mặc định đơn giản

        nearest_size = min(available_sizes, key=lambda x: abs(x - avg_field_size))

        # Lấy dữ liệu PDD cho kích thước trường này
        pdd_curve = self.parameters.pdd_data.get(
            nearest_size, [100.0, 80.0, 60.0, 40.0, 20.0]
        )

        # Tính nội suy PDD
        # Giả định rằng pdd_curve chứa các giá trị tại các độ sâu chuẩn (0, 5, 10, 15, 20) cm
        std_depths = np.linspace(0, 20, len(pdd_curve))

        if depth > max(std_depths):
            # Ngoại suy đơn giản cho độ sâu lớn hơn
            return pdd_curve[-1] * np.exp(-0.04 * (depth - max(std_depths)))

        # Nội suy tuyến tính
        return np.interp(depth, std_depths, pdd_curve)

    def get_output_factor(self, field_size: Tuple[float, float]) -> float:
        """
        Lấy hệ số đầu ra cho kích thước trường xác định.

        Args:
            field_size: Kích thước trường (cm)

        Returns:
            Hệ số đầu ra
        """
        # TODO: Triển khai tính toán hệ số đầu ra dựa trên dữ liệu đo lường
        avg_field_size = (field_size[0] + field_size[1]) / 2

        # Mô hình đơn giản: Hệ số đầu ra tăng với kích thước trường, bão hòa ở kích thước lớn
        max_of = 1.05  # Giá trị hệ số đầu ra tối đa
        min_of = 0.85  # Giá trị hệ số đầu ra tối thiểu

        # Sử dụng hàm sigmoid để mô phỏng hệ số đầu ra theo kích thước trường
        return min_of + (max_of - min_of) * (1 - np.exp(-0.1 * avg_field_size))

    def get_fluence_map(
        self, field_size: Tuple[float, float], resolution: float = 0.1
    ) -> np.ndarray:
        """
        Tạo bản đồ fluence cho kích thước trường và độ phân giải xác định.

        Args:
            field_size: Kích thước trường (cm)
            resolution: Độ phân giải (cm/pixel)

        Returns:
            2D numpy array biểu diễn bản đồ fluence
        """
        # Tính toán kích thước bản đồ
        width_pixels = int(field_size[0] / resolution)
        height_pixels = int(field_size[1] / resolution)

        # Tạo bản đồ fluence cơ bản
        fluence_map = np.ones((height_pixels, width_pixels))

        # Mô phỏng hiệu ứng penumbra và horns (đối với chùm tia có bộ lọc phẳng)
        if self.parameters.energy_type != BeamEnergyType.FFF:
            # Thêm horns
            x = np.linspace(-1, 1, width_pixels)
            y = np.linspace(-1, 1, height_pixels)
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)

            # Mô phỏng horn như một hàm của khoảng cách tính từ trung tâm
            horn_factor = 1 + 0.1 * (r - 0.5) * (r < 0.7)
            fluence_map *= horn_factor

        # Áp dụng penumbra ở rìa trường
        penumbra_pixels = int(self.parameters.penumbra_width / resolution)
        for i in range(penumbra_pixels):
            edge_value = 0.5 * (1 + np.cos(np.pi * i / penumbra_pixels))
            # Áp dụng cho 4 cạnh
            if i < height_pixels:
                fluence_map[i, :] *= edge_value  # Top edge
                if height_pixels - i - 1 >= 0:
                    fluence_map[height_pixels - i - 1, :] *= edge_value  # Bottom edge

            if i < width_pixels:
                fluence_map[:, i] *= edge_value  # Left edge
                if width_pixels - i - 1 >= 0:
                    fluence_map[:, width_pixels - i - 1] *= edge_value  # Right edge

        return fluence_map

    def __str__(self) -> str:
        """Biểu diễn chuỗi của mô hình chùm tia."""
        return (
            f"BeamModel({self.name}, {self.energy}, {self.parameters.model_type.name})"
        )
