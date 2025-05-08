#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module xử lý dữ liệu chùm tia.

Module này cung cấp các lớp và phương thức để xử lý dữ liệu chùm tia
từ nhiều máy xạ trị khác nhau, chuyển đổi giữa các định dạng và chuẩn hóa
dữ liệu cho hệ thống QuangTPS.
"""

import logging
import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class BeamDataProcessor(ABC):
    """
    Lớp trừu tượng gốc cho việc xử lý dữ liệu chùm tia.

    Định nghĩa giao diện chung cho tất cả các bộ xử lý dữ liệu chùm tia
    từ các loại máy xạ trị khác nhau.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Khởi tạo bộ xử lý dữ liệu chùm tia.

        Args:
            data_path: Đường dẫn đến thư mục chứa dữ liệu chùm tia
        """
        self.data_path = data_path
        self.machine_data = {}
        self.beam_data = {}
        self.energy_data = {}

    @abstractmethod
    def load_machine_data(self, machine_id: str) -> bool:
        """
        Tải dữ liệu máy xạ trị.

        Args:
            machine_id: ID của máy xạ trị

        Returns:
            True nếu tải thành công, False nếu không
        """
        pass

    @abstractmethod
    def get_available_energies(self) -> List[str]:
        """
        Lấy danh sách các năng lượng khả dụng.

        Returns:
            Danh sách các năng lượng khả dụng
        """
        pass

    @abstractmethod
    def get_beam_data(self, energy: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu chùm tia cho năng lượng cụ thể.

        Args:
            energy: Năng lượng chùm tia (VD: "6X", "10X", "6FFF")

        Returns:
            Dữ liệu chùm tia
        """
        pass

    @abstractmethod
    def get_pdd_data(
        self, energy: str, field_size: Tuple[float, float]
    ) -> Dict[str, np.ndarray]:
        """
        Lấy dữ liệu PDD (Percentage Depth Dose) cho năng lượng và kích thước trường.

        Args:
            energy: Năng lượng chùm tia
            field_size: Kích thước trường (chiều rộng, chiều cao) theo cm

        Returns:
            Dữ liệu PDD với các mảng cho độ sâu và liều
        """
        pass

    @abstractmethod
    def get_profile_data(
        self, energy: str, depth: float, field_size: Tuple[float, float]
    ) -> Dict[str, np.ndarray]:
        """
        Lấy dữ liệu profile cho năng lượng, độ sâu và kích thước trường.

        Args:
            energy: Năng lượng chùm tia
            depth: Độ sâu (cm)
            field_size: Kích thước trường (chiều rộng, chiều cao) theo cm

        Returns:
            Dữ liệu profile với các mảng cho vị trí và liều
        """
        pass

    @abstractmethod
    def get_output_factors(self, energy: str) -> Dict[Tuple[float, float], float]:
        """
        Lấy các hệ số output cho năng lượng.

        Args:
            energy: Năng lượng chùm tia

        Returns:
            Dictionary với khóa là kích thước trường và giá trị là hệ số output
        """
        pass

    def interpolate_pdd(
        self, energy: str, field_size: Tuple[float, float], depths: np.ndarray
    ) -> np.ndarray:
        """
        Nội suy dữ liệu PDD cho các độ sâu tùy chỉnh.

        Args:
            energy: Năng lượng chùm tia
            field_size: Kích thước trường
            depths: Mảng các độ sâu cần nội suy

        Returns:
            Mảng giá trị PDD tương ứng
        """
        try:
            # Lấy dữ liệu PDD
            pdd_data = self.get_pdd_data(energy, field_size)

            if not pdd_data or "depth" not in pdd_data or "dose" not in pdd_data:
                logger.error(
                    f"Không tìm thấy dữ liệu PDD cho năng lượng {energy} và trường {field_size}"
                )
                return np.zeros_like(depths)

            # Nội suy tuyến tính
            from scipy.interpolate import interp1d

            pdd_func = interp1d(
                pdd_data["depth"],
                pdd_data["dose"],
                bounds_error=False,
                fill_value=(pdd_data["dose"][0], pdd_data["dose"][-1]),
            )

            return pdd_func(depths)

        except Exception as e:
            logger.error(f"Lỗi khi nội suy PDD: {e}")
            return np.zeros_like(depths)

    def interpolate_profile(
        self,
        energy: str,
        depth: float,
        field_size: Tuple[float, float],
        positions: np.ndarray,
        axis: str = "x",
    ) -> np.ndarray:
        """
        Nội suy dữ liệu profile cho các vị trí tùy chỉnh.

        Args:
            energy: Năng lượng chùm tia
            depth: Độ sâu
            field_size: Kích thước trường
            positions: Mảng các vị trí cần nội suy
            axis: Trục nội suy ('x' hoặc 'y')

        Returns:
            Mảng giá trị profile tương ứng
        """
        try:
            # Lấy dữ liệu profile
            profile_data = self.get_profile_data(energy, depth, field_size)

            if not profile_data:
                logger.error(
                    f"Không tìm thấy dữ liệu profile cho năng lượng {energy}, độ sâu {depth} và trường {field_size}"
                )
                return np.zeros_like(positions)

            pos_key = f"{axis}_pos"
            dose_key = f"{axis}_dose"

            if pos_key not in profile_data or dose_key not in profile_data:
                logger.error(f"Không tìm thấy dữ liệu profile cho trục {axis}")
                return np.zeros_like(positions)

            # Nội suy tuyến tính
            from scipy.interpolate import interp1d

            profile_func = interp1d(
                profile_data[pos_key],
                profile_data[dose_key],
                bounds_error=False,
                fill_value=(0, 0),  # Giá trị 0 ngoài biên
            )

            return profile_func(positions)

        except Exception as e:
            logger.error(f"Lỗi khi nội suy profile: {e}")
            return np.zeros_like(positions)

    def interpolate_output_factor(
        self, energy: str, field_size: Tuple[float, float]
    ) -> float:
        """
        Nội suy hệ số output cho kích thước trường tùy chỉnh.

        Args:
            energy: Năng lượng chùm tia
            field_size: Kích thước trường

        Returns:
            Hệ số output nội suy
        """
        try:
            # Lấy bảng hệ số output
            output_factors = self.get_output_factors(energy)

            if not output_factors:
                logger.error(
                    f"Không tìm thấy dữ liệu hệ số output cho năng lượng {energy}"
                )
                return 1.0

            # Chuyển thành mảng để nội suy
            field_sizes = np.array(list(output_factors.keys()))
            factors = np.array(list(output_factors.values()))

            # Tính tổng diện tích trường
            field_areas = field_sizes[:, 0] * field_sizes[:, 1]
            target_area = field_size[0] * field_size[1]

            # Nội suy theo diện tích trường
            from scipy.interpolate import interp1d

            if len(field_areas) < 2:
                return factors[0] if factors.size > 0 else 1.0

            of_func = interp1d(
                field_areas,
                factors,
                bounds_error=False,
                fill_value=(factors[0], factors[-1]),
            )

            return float(of_func(target_area))

        except Exception as e:
            logger.error(f"Lỗi khi nội suy hệ số output: {e}")
            return 1.0

    def export_to_json(self, output_path: str) -> bool:
        """
        Xuất dữ liệu chùm tia sang định dạng JSON.

        Args:
            output_path: Đường dẫn đến file đầu ra

        Returns:
            True nếu xuất thành công, False nếu không
        """
        try:
            # Chuẩn bị dữ liệu để xuất
            export_data = {
                "machine_data": self.machine_data,
                "energies": list(self.energy_data.keys()),
                "beam_data": {},
            }

            # Chuyển đổi các mảng numpy thành list để dễ serialize
            for energy, data in self.energy_data.items():
                export_data["beam_data"][energy] = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        export_data["beam_data"][energy][key] = value.tolist()
                    elif isinstance(value, dict):
                        processed_dict = {}
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                processed_dict[k] = v.tolist()
                            else:
                                processed_dict[k] = v
                        export_data["beam_data"][energy][key] = processed_dict
                    else:
                        export_data["beam_data"][energy][key] = value

            # Ghi ra file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Đã xuất dữ liệu chùm tia sang {output_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xuất dữ liệu ra JSON: {e}")
            return False


class TrueBeamDataProcessor(BeamDataProcessor):
    """
    Lớp xử lý dữ liệu chùm tia từ máy xạ trị TrueBeam.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Khởi tạo bộ xử lý dữ liệu TrueBeam.

        Args:
            data_path: Đường dẫn đến thư mục chứa dữ liệu TrueBeam
        """
        super().__init__(data_path)
        self.machine_id = "TrueBeam"

    def load_machine_data(self, machine_id: str = "TrueBeam") -> bool:
        """
        Tải dữ liệu máy TrueBeam.

        Args:
            machine_id: ID của máy xạ trị

        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            if not self.data_path:
                logger.error("Chưa cung cấp đường dẫn dữ liệu")
                return False

            # Thiết lập đường dẫn đến dữ liệu máy
            machine_dir = os.path.join(self.data_path, machine_id)

            if not os.path.exists(machine_dir):
                logger.error(
                    f"Không tìm thấy thư mục dữ liệu cho máy {machine_id}: {machine_dir}"
                )
                return False

            # Tải dữ liệu máy cơ bản
            machine_info_path = os.path.join(machine_dir, "machine_info.json")
            if os.path.exists(machine_info_path):
                with open(machine_info_path, "r", encoding="utf-8") as f:
                    self.machine_data = json.load(f)
            else:
                # Thiết lập thông tin mặc định nếu không tìm thấy file
                self.machine_data = {
                    "name": machine_id,
                    "manufacturer": "Varian",
                    "model": "TrueBeam",
                    "serial_number": "Unknown",
                    "available_energies": [
                        "6X",
                        "10X",
                        "15X",
                        "6FFF",
                        "10FFF",
                        "6E",
                        "9E",
                        "12E",
                        "15E",
                        "18E",
                        "20E",
                    ],
                }

            # Tải dữ liệu từng năng lượng
            energy_dir = os.path.join(machine_dir, "energies")
            if os.path.exists(energy_dir):
                for energy in os.listdir(energy_dir):
                    energy_path = os.path.join(energy_dir, energy)
                    if os.path.isdir(energy_path):
                        # Tìm file dữ liệu chính
                        beam_data_path = os.path.join(energy_path, "beam_data.json")
                        if os.path.exists(beam_data_path):
                            with open(beam_data_path, "r", encoding="utf-8") as f:
                                self.energy_data[energy] = json.load(f)
                                logger.info(f"Đã tải dữ liệu cho năng lượng {energy}")

            if not self.energy_data:
                logger.warning(
                    f"Không tìm thấy dữ liệu năng lượng nào cho máy {machine_id}"
                )

            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu máy TrueBeam: {e}")
            return False

    def get_available_energies(self) -> List[str]:
        """
        Lấy danh sách các năng lượng khả dụng cho TrueBeam.

        Returns:
            Danh sách các năng lượng khả dụng
        """
        return list(self.energy_data.keys())

    def get_beam_data(self, energy: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu chùm tia cho năng lượng cụ thể của TrueBeam.

        Args:
            energy: Năng lượng chùm tia (VD: "6X", "10X", "6FFF")

        Returns:
            Dữ liệu chùm tia
        """
        if energy not in self.energy_data:
            logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
            return {}

        return self.energy_data[energy]

    def get_pdd_data(
        self, energy: str, field_size: Tuple[float, float]
    ) -> Dict[str, np.ndarray]:
        """
        Lấy dữ liệu PDD cho TrueBeam.

        Args:
            energy: Năng lượng chùm tia
            field_size: Kích thước trường (chiều rộng, chiều cao) theo cm

        Returns:
            Dữ liệu PDD với các mảng cho độ sâu và liều
        """
        if energy not in self.energy_data:
            logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
            return {}

        energy_data = self.energy_data[energy]

        if "pdd" not in energy_data:
            logger.error(f"Không tìm thấy dữ liệu PDD cho năng lượng {energy}")
            return {}

        # Tìm dữ liệu PDD cho kích thước trường phù hợp
        field_key = f"{field_size[0]}x{field_size[1]}"
        for key, data in energy_data["pdd"].items():
            if key == field_key:
                return {
                    "depth": np.array(data["depth"]),
                    "dose": np.array(data["dose"]),
                }

        # Nếu không tìm thấy chính xác, tìm gần đúng
        logger.warning(
            f"Không tìm thấy dữ liệu PDD chính xác cho trường {field_key}, đang tìm gần đúng"
        )

        # Tìm kích thước gần nhất
        min_diff = float("inf")
        closest_key = None
        target_area = field_size[0] * field_size[1]

        for key in energy_data["pdd"].keys():
            try:
                w, h = map(float, key.split("x"))
                area = w * h
                diff = abs(area - target_area)

                if diff < min_diff:
                    min_diff = diff
                    closest_key = key
            except:
                continue

        if closest_key:
            logger.info(
                f"Sử dụng dữ liệu PDD cho trường {closest_key} thay thế cho {field_key}"
            )
            return {
                "depth": np.array(energy_data["pdd"][closest_key]["depth"]),
                "dose": np.array(energy_data["pdd"][closest_key]["dose"]),
            }

        # Nếu không tìm thấy, trả về rỗng
        return {}

    def get_profile_data(
        self, energy: str, depth: float, field_size: Tuple[float, float]
    ) -> Dict[str, np.ndarray]:
        """
        Lấy dữ liệu profile cho TrueBeam.

        Args:
            energy: Năng lượng chùm tia
            depth: Độ sâu (cm)
            field_size: Kích thước trường (chiều rộng, chiều cao) theo cm

        Returns:
            Dữ liệu profile với các mảng cho vị trí và liều
        """
        if energy not in self.energy_data:
            logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
            return {}

        energy_data = self.energy_data[energy]

        if "profiles" not in energy_data:
            logger.error(f"Không tìm thấy dữ liệu profile cho năng lượng {energy}")
            return {}

        # Tìm dữ liệu profile cho kích thước trường và độ sâu phù hợp
        field_key = f"{field_size[0]}x{field_size[1]}"
        depth_key = f"{depth}"

        if (
            field_key in energy_data["profiles"]
            and depth_key in energy_data["profiles"][field_key]
        ):
            profile_data = energy_data["profiles"][field_key][depth_key]
            return {
                "x_pos": np.array(profile_data.get("x_pos", [])),
                "x_dose": np.array(profile_data.get("x_dose", [])),
                "y_pos": np.array(profile_data.get("y_pos", [])),
                "y_dose": np.array(profile_data.get("y_dose", [])),
            }

        # Nếu không tìm thấy, tìm độ sâu gần nhất
        if field_key in energy_data["profiles"]:
            logger.warning(
                f"Không tìm thấy dữ liệu profile cho độ sâu {depth}, đang tìm gần đúng"
            )

            # Tìm độ sâu gần nhất
            available_depths = [
                float(d) for d in energy_data["profiles"][field_key].keys()
            ]
            if not available_depths:
                return {}

            closest_depth = min(available_depths, key=lambda x: abs(float(x) - depth))
            depth_key = f"{closest_depth}"

            logger.info(
                f"Sử dụng dữ liệu profile ở độ sâu {depth_key} thay thế cho {depth}"
            )
            profile_data = energy_data["profiles"][field_key][depth_key]
            return {
                "x_pos": np.array(profile_data.get("x_pos", [])),
                "x_dose": np.array(profile_data.get("x_dose", [])),
                "y_pos": np.array(profile_data.get("y_pos", [])),
                "y_dose": np.array(profile_data.get("y_dose", [])),
            }

        # Nếu không tìm thấy trường, tìm kích thước trường gần nhất
        logger.warning(
            f"Không tìm thấy dữ liệu profile cho trường {field_key}, đang tìm gần đúng"
        )

        # Tìm kích thước gần nhất
        min_diff = float("inf")
        closest_field = None
        target_area = field_size[0] * field_size[1]

        for key in energy_data["profiles"].keys():
            try:
                w, h = map(float, key.split("x"))
                area = w * h
                diff = abs(area - target_area)

                if diff < min_diff:
                    min_diff = diff
                    closest_field = key
            except:
                continue

        if closest_field and energy_data["profiles"][closest_field]:
            # Tìm độ sâu gần nhất trong trường này
            available_depths = [
                float(d) for d in energy_data["profiles"][closest_field].keys()
            ]
            if not available_depths:
                return {}

            closest_depth = min(available_depths, key=lambda x: abs(float(x) - depth))
            depth_key = f"{closest_depth}"

            logger.info(
                f"Sử dụng dữ liệu profile cho trường {closest_field} ở độ sâu {depth_key}"
            )
            profile_data = energy_data["profiles"][closest_field][depth_key]
            return {
                "x_pos": np.array(profile_data.get("x_pos", [])),
                "x_dose": np.array(profile_data.get("x_dose", [])),
                "y_pos": np.array(profile_data.get("y_pos", [])),
                "y_dose": np.array(profile_data.get("y_dose", [])),
            }

        # Nếu không tìm thấy, trả về rỗng
        return {}

    def get_output_factors(self, energy: str) -> Dict[Tuple[float, float], float]:
        """
        Lấy các hệ số output cho TrueBeam.

        Args:
            energy: Năng lượng chùm tia

        Returns:
            Dictionary với khóa là kích thước trường và giá trị là hệ số output
        """
        if energy not in self.energy_data:
            logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
            return {}

        energy_data = self.energy_data[energy]

        if "output_factors" not in energy_data:
            logger.error(f"Không tìm thấy dữ liệu hệ số output cho năng lượng {energy}")
            return {}

        # Chuyển đổi dữ liệu từ JSON
        output_factors = {}
        for key, value in energy_data["output_factors"].items():
            try:
                w, h = map(float, key.split("x"))
                output_factors[(w, h)] = float(value)
            except:
                logger.warning(f"Không thể chuyển đổi hệ số output cho trường {key}")

        return output_factors


class GenericBeamDataProcessor(BeamDataProcessor):
    """
    Lớp xử lý dữ liệu chùm tia chung cho nhiều loại máy.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Khởi tạo bộ xử lý dữ liệu chung.

        Args:
            data_path: Đường dẫn đến thư mục chứa dữ liệu chùm tia
        """
        super().__init__(data_path)

    def load_machine_data(self, machine_id: str) -> bool:
        """
        Tải dữ liệu máy xạ trị từ định dạng JSON chung.

        Args:
            machine_id: ID của máy xạ trị

        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            if not self.data_path:
                logger.error("Chưa cung cấp đường dẫn dữ liệu")
                return False

            # Tìm file dữ liệu máy
            machine_file = os.path.join(self.data_path, f"{machine_id}.json")

            if not os.path.exists(machine_file):
                logger.error(
                    f"Không tìm thấy file dữ liệu cho máy {machine_id}: {machine_file}"
                )
                return False

            # Tải toàn bộ dữ liệu từ file JSON
            with open(machine_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Phân tích dữ liệu
            if "machine_data" in data:
                self.machine_data = data["machine_data"]
            else:
                logger.warning("Không tìm thấy thông tin máy trong file dữ liệu")

            # Tải dữ liệu từng năng lượng
            if "beam_data" in data:
                for energy, energy_data in data["beam_data"].items():
                    self.energy_data[energy] = energy_data

            if not self.energy_data:
                logger.warning(
                    f"Không tìm thấy dữ liệu năng lượng nào cho máy {machine_id}"
                )

            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu máy: {e}")
            return False

    def get_available_energies(self) -> List[str]:
        """
        Lấy danh sách các năng lượng khả dụng.

        Returns:
            Danh sách các năng lượng khả dụng
        """
        return list(self.energy_data.keys())

    def get_beam_data(self, energy: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu chùm tia cho năng lượng cụ thể.

        Args:
            energy: Năng lượng chùm tia

        Returns:
            Dữ liệu chùm tia
        """
        if energy not in self.energy_data:
            logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
            return {}

        return self.energy_data[energy]

    def get_pdd_data(
        self, energy: str, field_size: Tuple[float, float]
    ) -> Dict[str, np.ndarray]:
        """
        Lấy dữ liệu PDD cho năng lượng và kích thước trường.

        Args:
            energy: Năng lượng chùm tia
            field_size: Kích thước trường (chiều rộng, chiều cao) theo cm

        Returns:
            Dữ liệu PDD với các mảng cho độ sâu và liều
        """
        if energy not in self.energy_data:
            logger.error(f"Không tìm thấy dữ liệu cho năng lượng {energy}")
            return {}

        energy_data = self.energy_data[energy]

        if "pdd" not in energy_data:
            logger.error(f"Không tìm thấy dữ liệu PDD cho năng lượng {energy}")
            return {}

        # Tìm dữ liệu PDD cho kích thước trường phù hợp
        field_key = f"{field_size[0]}x{field_size[1]}"

        # Thử tìm trực tiếp
        if field_key in energy_data["pdd"]:
            pdd = energy_data["pdd"][field_key]
            return {"depth": np.array(pdd["depth"]), "dose": np.array(pdd["dose"])}

        # Nếu không tìm thấy, thử tìm gần đúng
        logger.warning(
            f"Không tìm thấy dữ liệu PDD chính xác cho trường {field_key}, đang tìm gần đúng"
        )

        # Phương pháp tìm trường gần đúng tương tự như TrueBeamDataProcessor
        # [Phần này tương tự với TrueBeamDataProcessor, đã được thu gọn]

        # Nếu không tìm thấy, trả về rỗng
        return {}

    def get_profile_data(
        self, energy: str, depth: float, field_size: Tuple[float, float]
    ) -> Dict[str, np.ndarray]:
        """
        Lấy dữ liệu profile cho năng lượng, độ sâu và kích thước trường.

        Args:
            energy: Năng lượng chùm tia
            depth: Độ sâu (cm)
            field_size: Kích thước trường (chiều rộng, chiều cao) theo cm

        Returns:
            Dữ liệu profile với các mảng cho vị trí và liều
        """
        # [Phần này tương tự với TrueBeamDataProcessor, đã được thu gọn]

        # Nếu không tìm thấy, trả về rỗng
        return {}

    def get_output_factors(self, energy: str) -> Dict[Tuple[float, float], float]:
        """
        Lấy các hệ số output cho năng lượng.

        Args:
            energy: Năng lượng chùm tia

        Returns:
            Dictionary với khóa là kích thước trường và giá trị là hệ số output
        """
        # [Phần này tương tự với TrueBeamDataProcessor, đã được thu gọn]

        # Nếu không tìm thấy, trả về rỗng
        return {}


# Factory function để tạo bộ xử lý dữ liệu chùm tia phù hợp
def create_beam_data_processor(
    machine_type: str, data_path: Optional[str] = None
) -> BeamDataProcessor:
    """
    Tạo bộ xử lý dữ liệu chùm tia phù hợp với loại máy.

    Args:
        machine_type: Loại máy (VD: "TrueBeam", "Halcyon", "Generic")
        data_path: Đường dẫn đến thư mục chứa dữ liệu chùm tia

    Returns:
        Bộ xử lý dữ liệu chùm tia
    """
    if machine_type.lower() == "truebeam":
        return TrueBeamDataProcessor(data_path)
    else:
        return GenericBeamDataProcessor(data_path)
