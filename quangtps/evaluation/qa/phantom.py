#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý phantom cho kiểm tra chất lượng xạ trị.

Module này cung cấp các lớp và chức năng để quản lý phantom QA,
bao gồm thư viện phantom sẵn có và tính năng tạo phantom tùy chỉnh.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import json
import copy

from quangtps.core.patient.patient import Patient
from quangtps.structures.roi import ROI
from quangtps.structures.structure_set import StructureSet
from quangtps.imaging.image_set import ImageSet

logger = logging.getLogger(__name__)


class PhantomType(Enum):
    """Loại phantom QA."""

    CUBIC = "cubic"
    CYLINDRICAL = "cylindrical"
    ANTHROPOMORPHIC = "anthropomorphic"
    SPECIALIZED = "specialized"
    CUSTOM = "custom"


class Phantom:
    """
    Lớp biểu diễn phantom QA.

    Lớp này cung cấp cấu trúc dữ liệu và phương thức để làm việc với phantom QA,
    bao gồm hình dạng, thông số vật lý, và cấu trúc bên trong.
    """

    def __init__(
        self,
        name: str,
        phantom_type: PhantomType,
        dimensions: Tuple[float, float, float],
        material: str = "Water",
        description: str = "",
    ):
        """
        Khởi tạo phantom QA.

        Parameters
        ----------
        name : str
            Tên phantom
        phantom_type : PhantomType
            Loại phantom
        dimensions : Tuple[float, float, float]
            Kích thước phantom (mm)
        material : str, optional
            Vật liệu chính của phantom, mặc định là "Water"
        description : str, optional
            Mô tả về phantom, mặc định là ""
        """
        self.name = name
        self.phantom_type = phantom_type
        self.dimensions = dimensions  # (width, height, depth) in mm
        self.material = material
        self.description = description

        # Vị trí và hướng
        self.position = (0.0, 0.0, 0.0)  # (x, y, z) in mm
        self.orientation = (0.0, 0.0, 0.0)  # (roll, pitch, yaw) in degrees

        # Thông tin chi tiết
        self.center = (
            dimensions[0] / 2,
            dimensions[1] / 2,
            dimensions[2] / 2,
        )  # Trung tâm phantom
        self.vendor = ""
        self.model = ""
        self.serial_number = ""

        # Tài liệu/manual và thông tin tham khảo
        self.documentation = {}

        # Cấu trúc trong phantom (detector, film, chamber, ...)
        self.structures = StructureSet([])

        # Bộ ảnh CT của phantom (nếu có)
        self.image_set = None

        # Bệnh nhân ảo chứa phantom (nếu cần)
        self.patient = None

        # Giá trị chuẩn và tham số hiệu chuẩn
        self.calibration_data = {}

        # Lịch sử sử dụng
        self.usage_history = []

    def add_structure(self, structure: ROI) -> None:
        """
        Thêm cấu trúc vào phantom.

        Parameters
        ----------
        structure : ROI
            Cấu trúc cần thêm
        """
        self.structures.add_structure(structure)
        logger.info(f"Đã thêm cấu trúc {structure.name} vào phantom {self.name}")

    def set_image_set(self, image_set: ImageSet) -> None:
        """
        Thiết lập bộ ảnh CT cho phantom.

        Parameters
        ----------
        image_set : ImageSet
            Bộ ảnh CT của phantom
        """
        self.image_set = image_set
        logger.info(f"Đã thiết lập bộ ảnh CT cho phantom {self.name}")

    def create_virtual_patient(self) -> Patient:
        """
        Tạo bệnh nhân ảo chứa phantom cho mục đích QA.

        Returns
        -------
        Patient
            Bệnh nhân ảo chứa phantom
        """
        if self.patient:
            return self.patient

        # Tạo bệnh nhân ảo
        patient = Patient()
        patient.id = f"PHANTOM_{self.name}"
        patient.name = f"QA Phantom - {self.name}"

        # Thiết lập thông tin cơ bản cho bệnh nhân ảo
        patient.sex = "O"  # Other
        patient.birth_date = None
        patient.weight = None
        patient.height = None

        # Gán bộ ảnh CT nếu có
        if self.image_set:
            patient.add_image_set(self.image_set)

        # Gán cấu trúc nếu có
        if self.structures and len(self.structures) > 0:
            patient.add_structure_set(self.structures)

        # Lưu tham chiếu đến bệnh nhân ảo
        self.patient = patient

        logger.info(f"Đã tạo bệnh nhân ảo cho phantom {self.name}")
        return patient

    def get_calibration_factor(
        self, energy: Union[str, float], field_size: Tuple[float, float]
    ) -> float:
        """
        Lấy hệ số hiệu chuẩn cho năng lượng và kích thước trường xạ.

        Parameters
        ----------
        energy : Union[str, float]
            Năng lượng chùm tia (MV hoặc MeV)
        field_size : Tuple[float, float]
            Kích thước trường xạ (cm)

        Returns
        -------
        float
            Hệ số hiệu chuẩn
        """
        # Kiểm tra nếu có dữ liệu hiệu chuẩn
        if not self.calibration_data:
            logger.warning(f"Không có dữ liệu hiệu chuẩn cho phantom {self.name}")
            return 1.0

        # Chuyển đổi năng lượng sang chuỗi để sử dụng làm khóa
        energy_key = str(energy)

        # Tìm khóa phù hợp cho kích thước trường
        field_size_key = None
        for key in self.calibration_data.get(energy_key, {}):
            if key == "default":
                field_size_key = key
                continue

            # Phân tích khóa kích thước trường
            try:
                fs = eval(key)  # Ví dụ: "(10, 10)" -> (10, 10)
                if abs(fs[0] - field_size[0]) <= 1 and abs(fs[1] - field_size[1]) <= 1:
                    field_size_key = key
                    break
            except:
                continue

        # Sử dụng giá trị mặc định nếu không tìm thấy khớp
        if field_size_key is None:
            if "default" in self.calibration_data.get(energy_key, {}):
                field_size_key = "default"
            else:
                logger.warning(
                    f"Không tìm thấy hệ số hiệu chuẩn phù hợp cho {energy} MV, trường {field_size} cm"
                )
                return 1.0

        return self.calibration_data[energy_key][field_size_key]

    def add_calibration_data(
        self,
        energy: Union[str, float],
        field_size: Optional[Tuple[float, float]],
        factor: float,
    ) -> None:
        """
        Thêm dữ liệu hiệu chuẩn cho phantom.

        Parameters
        ----------
        energy : Union[str, float]
            Năng lượng chùm tia (MV hoặc MeV)
        field_size : Optional[Tuple[float, float]]
            Kích thước trường xạ (cm), None cho mặc định
        factor : float
            Hệ số hiệu chuẩn
        """
        # Chuyển đổi năng lượng sang chuỗi để sử dụng làm khóa
        energy_key = str(energy)

        # Tạo cấu trúc dữ liệu nếu chưa có
        if energy_key not in self.calibration_data:
            self.calibration_data[energy_key] = {}

        # Thêm hệ số hiệu chuẩn
        if field_size is None:
            self.calibration_data[energy_key]["default"] = factor
        else:
            self.calibration_data[energy_key][str(field_size)] = factor

        logger.info(
            f"Đã thêm hệ số hiệu chuẩn {factor} cho {energy} MV, trường {field_size if field_size else 'mặc định'}"
        )

    def record_usage(self, purpose: str, operator: str, notes: str = "") -> None:
        """
        Ghi lại việc sử dụng phantom.

        Parameters
        ----------
        purpose : str
            Mục đích sử dụng
        operator : str
            Người thực hiện
        notes : str, optional
            Ghi chú, mặc định là ""
        """
        import datetime

        usage_entry = {
            "date": datetime.datetime.now(),
            "purpose": purpose,
            "operator": operator,
            "notes": notes,
        }

        self.usage_history.append(usage_entry)
        logger.info(f"Đã ghi lại việc sử dụng phantom {self.name} bởi {operator}")

    def save(self, file_path: str) -> bool:
        """
        Lưu thông tin phantom vào file.

        Parameters
        ----------
        file_path : str
            Đường dẫn file lưu

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không
        """
        try:
            # Tạo dữ liệu để lưu
            data = {
                "name": self.name,
                "phantom_type": self.phantom_type.value,
                "dimensions": self.dimensions,
                "material": self.material,
                "description": self.description,
                "position": self.position,
                "orientation": self.orientation,
                "center": self.center,
                "vendor": self.vendor,
                "model": self.model,
                "serial_number": self.serial_number,
                "documentation": self.documentation,
                "calibration_data": self.calibration_data,
            }

            # Lưu vào file JSON
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Đã lưu thông tin phantom {self.name} vào {file_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi lưu thông tin phantom: {e}")
            return False

    @classmethod
    def load(cls, file_path: str) -> Optional["Phantom"]:
        """
        Tải thông tin phantom từ file.

        Parameters
        ----------
        file_path : str
            Đường dẫn file

        Returns
        -------
        Optional[Phantom]
            Đối tượng Phantom nếu tải thành công, None nếu không
        """
        try:
            # Đọc dữ liệu từ file JSON
            with open(file_path, "r") as f:
                data = json.load(f)

            # Tạo đối tượng Phantom
            phantom = cls(
                name=data["name"],
                phantom_type=PhantomType(data["phantom_type"]),
                dimensions=tuple(data["dimensions"]),
                material=data["material"],
                description=data["description"],
            )

            # Thiết lập các thuộc tính khác
            phantom.position = tuple(data["position"])
            phantom.orientation = tuple(data["orientation"])
            phantom.center = tuple(data["center"])
            phantom.vendor = data["vendor"]
            phantom.model = data["model"]
            phantom.serial_number = data["serial_number"]
            phantom.documentation = data["documentation"]
            phantom.calibration_data = data["calibration_data"]

            logger.info(f"Đã tải thông tin phantom {phantom.name} từ {file_path}")
            return phantom

        except Exception as e:
            logger.error(f"Lỗi khi tải thông tin phantom: {e}")
            return None


class PhantomLibrary:
    """
    Thư viện quản lý các phantom QA.

    Lớp này cung cấp các phương thức để quản lý bộ sưu tập phantom QA,
    bao gồm phantom chuẩn và phantom tùy chỉnh.
    """

    # Đường dẫn thư mục lưu trữ phantom
    PHANTOM_DIR = os.path.join(os.path.expanduser("~"), ".quangtps", "phantoms")

    # Danh sách phantom được tải
    _phantoms = {}

    # Phantom mặc định
    _default_phantom = None

    @classmethod
    def initialize(cls) -> None:
        """Khởi tạo thư viện phantom."""
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        if not os.path.exists(cls.PHANTOM_DIR):
            os.makedirs(cls.PHANTOM_DIR)
            logger.info(f"Đã tạo thư mục lưu trữ phantom: {cls.PHANTOM_DIR}")

        # Tạo các phantom chuẩn nếu chưa có
        cls._create_standard_phantoms()

        # Tải tất cả phantom từ thư mục lưu trữ
        cls._load_all_phantoms()

    @classmethod
    def _create_standard_phantoms(cls) -> None:
        """Tạo các phantom chuẩn."""
        # Kiểm tra xem đã có phantom chuẩn chưa
        standard_phantom_file = os.path.join(
            cls.PHANTOM_DIR, "standard_water_phantom.json"
        )
        if os.path.exists(standard_phantom_file):
            return

        # Tạo phantom nước chuẩn
        water_phantom = Phantom(
            name="Standard Water Phantom",
            phantom_type=PhantomType.CUBIC,
            dimensions=(300.0, 300.0, 300.0),  # 30x30x30 cm
            material="Water",
            description="Phantom nước chuẩn 30x30x30 cm cho kiểm tra chất lượng cơ bản",
        )
        water_phantom.vendor = "QuangTPS"
        water_phantom.model = "Standard Water Cube"

        # Thêm dữ liệu hiệu chuẩn cơ bản
        water_phantom.add_calibration_data("6", (10, 10), 1.0)
        water_phantom.add_calibration_data("10", (10, 10), 1.0)
        water_phantom.add_calibration_data("15", (10, 10), 1.0)

        # Lưu phantom
        water_phantom.save(standard_phantom_file)
        logger.info(f"Đã tạo phantom nước chuẩn: {standard_phantom_file}")

        # Tạo phantom ArcCHECK
        arc_check = Phantom(
            name="ArcCHECK",
            phantom_type=PhantomType.CYLINDRICAL,
            dimensions=(264.0, 264.0, 320.0),  # 26.4 cm đường kính, 32 cm chiều dài
            material="PMMA",
            description="ArcCHECK phantom for VMAT và IMRT QA",
        )
        arc_check.vendor = "Sun Nuclear"
        arc_check.model = "ArcCHECK"

        # Lưu phantom
        arc_check_file = os.path.join(cls.PHANTOM_DIR, "arccheck_phantom.json")
        arc_check.save(arc_check_file)
        logger.info(f"Đã tạo phantom ArcCHECK: {arc_check_file}")

        # Tạo phantom solid water
        solid_water = Phantom(
            name="Solid Water Phantom",
            phantom_type=PhantomType.CUBIC,
            dimensions=(300.0, 300.0, 200.0),  # 30x30x20 cm
            material="Solid Water",
            description="Solid water phantom cho kiểm tra liều tuyệt đối và film dosimetry",
        )
        solid_water.vendor = "QuangTPS"
        solid_water.model = "Solid Water"

        # Lưu phantom
        solid_water_file = os.path.join(cls.PHANTOM_DIR, "solid_water_phantom.json")
        solid_water.save(solid_water_file)
        logger.info(f"Đã tạo phantom solid water: {solid_water_file}")

    @classmethod
    def _load_all_phantoms(cls) -> None:
        """Tải tất cả phantom từ thư mục lưu trữ."""
        if not os.path.exists(cls.PHANTOM_DIR):
            logger.warning(f"Thư mục phantom không tồn tại: {cls.PHANTOM_DIR}")
            return

        # Xóa danh sách hiện tại
        cls._phantoms = {}

        # Tìm tất cả file JSON trong thư mục
        for file_name in os.listdir(cls.PHANTOM_DIR):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(cls.PHANTOM_DIR, file_name)

            # Tải phantom
            phantom = Phantom.load(file_path)
            if phantom:
                cls._phantoms[phantom.name] = phantom

                # Đặt phantom mặc định nếu chưa có
                if (
                    cls._default_phantom is None
                    and phantom.name == "Standard Water Phantom"
                ):
                    cls._default_phantom = phantom

        logger.info(f"Đã tải {len(cls._phantoms)} phantom từ thư mục {cls.PHANTOM_DIR}")

    @classmethod
    def get_phantom(cls, name: str) -> Optional[Phantom]:
        """
        Lấy phantom theo tên.

        Parameters
        ----------
        name : str
            Tên phantom

        Returns
        -------
        Optional[Phantom]
            Đối tượng Phantom nếu tìm thấy, None nếu không
        """
        return cls._phantoms.get(name)

    @classmethod
    def get_all_phantoms(cls) -> Dict[str, Phantom]:
        """
        Lấy tất cả phantom có sẵn.

        Returns
        -------
        Dict[str, Phantom]
            Từ điển các phantom theo tên
        """
        return cls._phantoms

    @classmethod
    def get_default_phantom(cls) -> Optional[Phantom]:
        """
        Lấy phantom mặc định.

        Returns
        -------
        Optional[Phantom]
            Phantom mặc định nếu có, None nếu không
        """
        if cls._default_phantom is None and cls._phantoms:
            # Lấy phantom đầu tiên nếu không có phantom mặc định
            cls._default_phantom = next(iter(cls._phantoms.values()))

        return cls._default_phantom

    @classmethod
    def set_default_phantom(cls, name: str) -> bool:
        """
        Đặt phantom mặc định.

        Parameters
        ----------
        name : str
            Tên phantom

        Returns
        -------
        bool
            True nếu thành công, False nếu không
        """
        phantom = cls.get_phantom(name)
        if phantom:
            cls._default_phantom = phantom
            logger.info(f"Đã đặt {name} làm phantom mặc định")
            return True
        else:
            logger.warning(f"Không tìm thấy phantom: {name}")
            return False

    @classmethod
    def add_phantom(cls, phantom: Phantom) -> bool:
        """
        Thêm phantom vào thư viện.

        Parameters
        ----------
        phantom : Phantom
            Phantom cần thêm

        Returns
        -------
        bool
            True nếu thành công, False nếu không
        """
        if phantom.name in cls._phantoms:
            logger.warning(f"Phantom {phantom.name} đã tồn tại trong thư viện")
            return False

        # Thêm vào danh sách
        cls._phantoms[phantom.name] = phantom

        # Lưu vào file
        file_path = os.path.join(
            cls.PHANTOM_DIR, f"{phantom.name.replace(' ', '_').lower()}_phantom.json"
        )
        phantom.save(file_path)

        logger.info(f"Đã thêm phantom {phantom.name} vào thư viện")
        return True

    @classmethod
    def remove_phantom(cls, name: str) -> bool:
        """
        Xóa phantom khỏi thư viện.

        Parameters
        ----------
        name : str
            Tên phantom

        Returns
        -------
        bool
            True nếu thành công, False nếu không
        """
        if name not in cls._phantoms:
            logger.warning(f"Không tìm thấy phantom: {name}")
            return False

        # Xóa file
        file_path = os.path.join(
            cls.PHANTOM_DIR, f"{name.replace(' ', '_').lower()}_phantom.json"
        )
        if os.path.exists(file_path):
            os.remove(file_path)

        # Xóa khỏi danh sách
        phantom = cls._phantoms.pop(name)

        # Cập nhật phantom mặc định nếu cần
        if cls._default_phantom is phantom:
            cls._default_phantom = (
                next(iter(cls._phantoms.values())) if cls._phantoms else None
            )

        logger.info(f"Đã xóa phantom {name} khỏi thư viện")
        return True

    @classmethod
    def create_custom_phantom(
        cls,
        name: str,
        dimensions: Tuple[float, float, float],
        material: str = "Water",
        description: str = "",
    ) -> Phantom:
        """
        Tạo phantom tùy chỉnh mới.

        Parameters
        ----------
        name : str
            Tên phantom
        dimensions : Tuple[float, float, float]
            Kích thước phantom (mm)
        material : str, optional
            Vật liệu chính của phantom, mặc định là "Water"
        description : str, optional
            Mô tả về phantom, mặc định là ""

        Returns
        -------
        Phantom
            Phantom tùy chỉnh mới tạo
        """
        phantom = Phantom(
            name=name,
            phantom_type=PhantomType.CUSTOM,
            dimensions=dimensions,
            material=material,
            description=description,
        )

        # Thêm vào thư viện
        cls.add_phantom(phantom)

        return phantom
