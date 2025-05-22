#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý dữ liệu contour.

Module này cung cấp các lớp để lưu trữ và quản lý dữ liệu contour
cho các cấu trúc giải phẫu trong hệ thống lập kế hoạch xạ trị QuangTPS.
"""

import logging
import numpy as np
import uuid
import datetime
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)


class ContourData:
    """
    Lớp để lưu trữ và quản lý dữ liệu contour cho một cấu trúc.

    Attributes
    ----------
    name : str
        Tên của cấu trúc
    id : str
        ID duy nhất của cấu trúc
    color : Tuple[int, int, int]
        Màu RGB của cấu trúc
    contours : Dict[int, np.ndarray]
        Từ điển ánh xạ chỉ số lát cắt với dữ liệu contour
    creation_date : str
        Ngày giờ tạo cấu trúc
    last_modified : str
        Ngày giờ chỉnh sửa cấu trúc lần cuối
    metadata : Dict[str, Any]
        Metadata bổ sung cho cấu trúc
    """

    def __init__(
        self,
        name: str,
        color: Optional[Tuple[int, int, int]] = None,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo đối tượng ContourData.

        Parameters
        ----------
        name : str
            Tên của cấu trúc
        color : Tuple[int, int, int], optional
            Màu RGB của cấu trúc
        id : str, optional
            ID duy nhất của cấu trúc, mặc định tạo UUID mới
        metadata : Dict[str, Any], optional
            Metadata bổ sung cho cấu trúc
        """
        self.name = name
        self.id = id if id else str(uuid.uuid4())
        self.color = color if color else (255, 0, 0)  # Mặc định là màu đỏ
        self.contours = {}  # Dict[int, np.ndarray]
        self.creation_date = datetime.datetime.now().isoformat()
        self.last_modified = self.creation_date
        self.metadata = metadata if metadata else {}

    def add_contour(self, slice_idx: int, contour_points: np.ndarray) -> None:
        """
        Thêm contour cho một lát cắt.

        Parameters
        ----------
        slice_idx : int
            Chỉ số của lát cắt
        contour_points : np.ndarray
            Dữ liệu contour, mảng nx2 chứa tọa độ (x, y)
        """
        if contour_points is None or len(contour_points) < 3:
            logger.warning(f"Bỏ qua contour không hợp lệ cho lát cắt {slice_idx}")
            return

        # Đảm bảo contour_points là mảng numpy
        if not isinstance(contour_points, np.ndarray):
            try:
                contour_points = np.array(contour_points)
            except Exception as e:
                logger.error(
                    f"Không thể chuyển đổi contour_points sang numpy array: {e}"
                )
                return

        # Đảm bảo định dạng của contour_points là đúng (nx2)
        if contour_points.ndim != 2 or contour_points.shape[1] < 2:
            logger.warning(
                f"Contour_points phải có định dạng nx2, nhưng có shape {contour_points.shape}"
            )
            return

        # Lưu chỉ tọa độ (x, y)
        self.contours[slice_idx] = contour_points[:, :2].copy()
        self.last_modified = datetime.datetime.now().isoformat()

    def remove_contour(self, slice_idx: int) -> bool:
        """
        Xóa contour cho một lát cắt.

        Parameters
        ----------
        slice_idx : int
            Chỉ số của lát cắt cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if slice_idx in self.contours:
            del self.contours[slice_idx]
            self.last_modified = datetime.datetime.now().isoformat()
            return True
        return False

    def get_contour(self, slice_idx: int) -> Optional[np.ndarray]:
        """
        Lấy contour cho một lát cắt.

        Parameters
        ----------
        slice_idx : int
            Chỉ số của lát cắt

        Returns
        -------
        np.ndarray or None
            Dữ liệu contour hoặc None nếu không tìm thấy
        """
        return self.contours.get(slice_idx)

    def get_slices(self) -> List[int]:
        """
        Lấy danh sách các chỉ số lát cắt có contour.

        Returns
        -------
        List[int]
            Danh sách chỉ số lát cắt
        """
        return sorted(list(self.contours.keys()))

    def has_contours(self) -> bool:
        """
        Kiểm tra xem cấu trúc có contour nào không.

        Returns
        -------
        bool
            True nếu có contour, False nếu không có
        """
        return len(self.contours) > 0

    def clear_contours(self) -> None:
        """Xóa tất cả contours."""
        self.contours.clear()
        self.last_modified = datetime.datetime.now().isoformat()

    def update_metadata(self, key: str, value: Any) -> None:
        """
        Cập nhật một trường metadata.

        Parameters
        ----------
        key : str
            Khóa của trường metadata
        value : Any
            Giá trị mới
        """
        self.metadata[key] = value
        self.last_modified = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng thành từ điển để lưu trữ.

        Returns
        -------
        Dict[str, Any]
            Từ điển biểu diễn ContourData
        """
        # Chuyển đổi contours thành định dạng có thể JSON hóa
        serialized_contours = {}
        for slice_idx, contour in self.contours.items():
            serialized_contours[str(slice_idx)] = contour.tolist()

        return {
            "name": self.name,
            "id": self.id,
            "color": self.color,
            "contours": serialized_contours,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContourData":
        """
        Tạo đối tượng ContourData từ từ điển.

        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển biểu diễn ContourData

        Returns
        -------
        ContourData
            Đối tượng ContourData được tạo
        """
        try:
            contour_data = cls(
                name=data["name"],
                color=data.get("color", (255, 0, 0)),
                id=data.get("id"),
                metadata=data.get("metadata", {}),
            )

            # Cập nhật các trường thời gian
            contour_data.creation_date = data.get(
                "creation_date", contour_data.creation_date
            )
            contour_data.last_modified = data.get(
                "last_modified", contour_data.last_modified
            )

            # Khôi phục contours
            serialized_contours = data.get("contours", {})
            for slice_idx_str, contour_list in serialized_contours.items():
                try:
                    slice_idx = int(slice_idx_str)
                    contour_points = np.array(contour_list)
                    contour_data.contours[slice_idx] = contour_points
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Lỗi khi khôi phục contour cho lát cắt {slice_idx_str}: {e}"
                    )

            return contour_data
        except Exception as e:
            logger.error(f"Lỗi khi tạo ContourData từ dict: {e}")
            # Trả về một đối tượng trống
            return cls(name=data.get("name", "Unknown"))

    def save_to_json(self, filepath: str) -> bool:
        """
        Lưu ContourData vào file JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để lưu

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu có lỗi
        """
        try:
            data = self.to_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu ContourData vào {filepath}: {e}")
            return False

    @classmethod
    def load_from_json(cls, filepath: str) -> Optional["ContourData"]:
        """
        Tải ContourData từ file JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để tải

        Returns
        -------
        ContourData or None
            Đối tượng ContourData hoặc None nếu có lỗi
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Lỗi khi tải ContourData từ {filepath}: {e}")
            return None

    def get_volume(
        self,
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> float:
        """
        Tính thể tích của cấu trúc dựa trên các contour.

        Parameters
        ----------
        slice_thickness : float, optional
            Độ dày của lát cắt, mặc định là 1.0mm
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo (x, y), mặc định là (1.0, 1.0)mm

        Returns
        -------
        float
            Thể tích của cấu trúc tính bằng mm³
        """
        if not self.has_contours():
            return 0.0

        try:
            total_volume = 0.0
            slices = self.get_slices()

            for slice_idx in slices:
                contour = self.contours[slice_idx]
                if contour is None or len(contour) < 3:
                    continue

                # Tính diện tích contour bằng công thức Shoelace
                x = contour[:, 0] * pixel_spacing[0]
                y = contour[:, 1] * pixel_spacing[1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

                # Cộng thể tích của lát cắt vào tổng
                total_volume += area * slice_thickness

            return total_volume

        except Exception as e:
            logger.error(f"Lỗi khi tính thể tích cấu trúc: {str(e)}")
            return 0.0

    def get_centroid(
        self,
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Optional[np.ndarray]:
        """
        Tính tâm khối lượng của cấu trúc.

        Parameters
        ----------
        slice_thickness : float, optional
            Độ dày của lát cắt, mặc định là 1.0mm
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo (x, y), mặc định là (1.0, 1.0)mm

        Returns
        -------
        np.ndarray or None
            Tọa độ tâm khối lượng [x, y, z] hoặc None nếu không tính được
        """
        if not self.has_contours():
            return None

        try:
            total_volume = 0.0
            weighted_centroid = np.zeros(3)
            slices = self.get_slices()

            for slice_idx in slices:
                contour = self.contours[slice_idx]
                if contour is None or len(contour) < 3:
                    continue

                # Tính diện tích và tâm của contour
                x = contour[:, 0] * pixel_spacing[0]
                y = contour[:, 1] * pixel_spacing[1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

                # Tính tâm contour
                cx = np.mean(x)
                cy = np.mean(y)
                cz = slice_idx * slice_thickness

                # Cộng thể tích của lát cắt vào tổng
                slice_volume = area * slice_thickness
                total_volume += slice_volume

                # Cộng tọa độ tâm trọng lượng
                weighted_centroid += np.array([cx, cy, cz]) * slice_volume

            if total_volume > 0:
                return weighted_centroid / total_volume
            else:
                return None

        except Exception as e:
            logger.error(f"Lỗi khi tính tâm cấu trúc: {str(e)}")
            return None

    def simplify_contours(self, epsilon: float = 0.1) -> None:
        """
        Đơn giản hóa tất cả các contour để giảm số điểm.

        Parameters
        ----------
        epsilon : float, optional
            Tham số kiểm soát mức độ đơn giản hóa, mặc định là 0.1
        """
        if not self.has_contours():
            return

        try:
            from quangtps.segmentation.contour.contour_utils import ContourUtils

            simplified_contours = {}
            for slice_idx, contour in self.contours.items():
                if contour is None or len(contour) < 3:
                    simplified_contours[slice_idx] = contour
                    continue

                # Đơn giản hóa contour
                simplified = ContourUtils.simplify_contour(contour, epsilon)
                simplified_contours[slice_idx] = simplified

            # Cập nhật contours
            self.contours = simplified_contours
            self.last_modified = datetime.datetime.now().isoformat()
            self.update_metadata("simplified", True)
            self.update_metadata("simplify_epsilon", epsilon)

        except ImportError:
            logger.warning("Không thể import ContourUtils, bỏ qua đơn giản hóa contour")
        except Exception as e:
            logger.error(f"Lỗi khi đơn giản hóa contours: {str(e)}")

    def convert_to_mask(
        self,
        shape: Tuple[int, int, int],
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Optional[np.ndarray]:
        """
        Chuyển đổi contours thành mask 3D.

        Parameters
        ----------
        shape : Tuple[int, int, int]
            Kích thước của mask (z, y, x)
        slice_thickness : float, optional
            Độ dày của lát cắt, mặc định là 1.0mm
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo (x, y), mặc định là (1.0, 1.0)mm

        Returns
        -------
        np.ndarray or None
            Mask 3D hoặc None nếu không chuyển đổi được
        """
        if not self.has_contours():
            return None

        try:
            from quangtps.segmentation.contour.contour_utils import ContourUtils

            # Chuẩn bị contours ở định dạng phù hợp
            contours_list = []
            for slice_idx in range(shape[0]):
                contour = self.get_contour(slice_idx)
                if contour is not None and len(contour) >= 3:
                    contours_list.append(contour)
                else:
                    contours_list.append(np.zeros((0, 2)))

            # Chuyển đổi contours thành mask
            mask = ContourUtils.convert_contours_to_mask(
                contours_list, shape, slice_thickness, pixel_spacing
            )
            return mask

        except ImportError:
            logger.warning(
                "Không thể import ContourUtils, không thể chuyển đổi sang mask"
            )
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi contours sang mask: {str(e)}")

        return None

    def export_to_stl(
        self,
        filepath: str,
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> bool:
        """
        Xuất contours sang định dạng STL để sử dụng trong các phần mềm 3D.

        Parameters
        ----------
        filepath : str
            Đường dẫn tới file STL đầu ra
        slice_thickness : float, optional
            Độ dày của lát cắt, mặc định là 1.0mm
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo (x, y), mặc định là (1.0, 1.0)mm

        Returns
        -------
        bool
            True nếu xuất thành công, False nếu có lỗi
        """
        if not self.has_contours():
            logger.warning("Không có contour để xuất sang STL")
            return False

        try:
            import trimesh
            from skimage import measure

            # Đầu tiên chuyển contours thành mask 3D
            # Xác định kích thước hợp lý cho mask dựa trên contours
            slices = self.get_slices()
            max_slice = max(slices) if slices else 0

            # Tìm kích thước x, y tối đa
            x_max, y_max = 0, 0
            for contour in self.contours.values():
                if contour is not None and len(contour) > 0:
                    x_max = max(x_max, np.max(contour[:, 0]) if contour.size > 0 else 0)
                    y_max = max(y_max, np.max(contour[:, 1]) if contour.size > 0 else 0)

            # Thêm margin và làm tròn lên
            x_size = int(np.ceil(x_max + 10))
            y_size = int(np.ceil(y_max + 10))
            z_size = max_slice + 1

            # Chuyển đổi sang mask
            mask = self.convert_to_mask(
                (z_size, y_size, x_size), slice_thickness, pixel_spacing
            )
            if mask is None:
                logger.error("Không thể tạo mask từ contours")
                return False

            # Tạo mesh từ mask sử dụng marching cubes
            vertices, faces, normals, values = measure.marching_cubes(mask, 0.5)

            # Áp dụng pixel spacing và slice thickness
            vertices[:, 0] *= pixel_spacing[0]
            vertices[:, 1] *= pixel_spacing[1]
            vertices[:, 2] *= slice_thickness

            # Tạo mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)

            # Lưu mesh
            mesh.export(filepath)

            logger.info(f"Đã xuất contour sang STL: {filepath}")
            return True

        except ImportError as e:
            logger.error(
                f"Không thể import các thư viện cần thiết để xuất STL: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Lỗi khi xuất contour sang STL: {str(e)}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")

        return False

    def interpolate_missing_slices(self) -> bool:
        """
        Nội suy các contour cho các lát cắt bị thiếu.

        Returns
        -------
        bool
            True nếu nội suy thành công, False nếu có lỗi
        """
        if not self.has_contours():
            logger.warning("Không có contour để nội suy")
            return False

        try:
            slices = self.get_slices()
            if len(slices) < 2:
                logger.warning("Cần ít nhất 2 lát cắt để nội suy")
                return False

            # Tìm các lát cắt bị thiếu
            min_slice = min(slices)
            max_slice = max(slices)
            missing_slices = [
                i for i in range(min_slice, max_slice + 1) if i not in slices
            ]

            if not missing_slices:
                logger.info("Không có lát cắt nào bị thiếu")
                return True

            logger.info(f"Nội suy {len(missing_slices)} lát cắt bị thiếu")

            # Nội suy cho từng lát cắt bị thiếu
            for missing_slice in missing_slices:
                # Tìm lát cắt gần nhất bên trên và bên dưới
                lower_slices = [s for s in slices if s < missing_slice]
                upper_slices = [s for s in slices if s > missing_slice]

                if not lower_slices or not upper_slices:
                    continue  # Không thể nội suy nếu không có lát cắt trên hoặc dưới

                lower_slice = max(lower_slices)
                upper_slice = min(upper_slices)

                lower_contour = self.contours[lower_slice]
                upper_contour = self.contours[upper_slice]

                # Đảm bảo cả hai contour đều có cùng số điểm
                if len(lower_contour) != len(upper_contour):
                    # Resampling các contour để có cùng số điểm
                    num_points = max(len(lower_contour), len(upper_contour))

                    # Resampling lower_contour
                    lower_contour_resampled = np.zeros((num_points, 2))
                    for i in range(num_points):
                        idx = int(i * len(lower_contour) / num_points)
                        lower_contour_resampled[i] = lower_contour[idx]

                    # Resampling upper_contour
                    upper_contour_resampled = np.zeros((num_points, 2))
                    for i in range(num_points):
                        idx = int(i * len(upper_contour) / num_points)
                        upper_contour_resampled[i] = upper_contour[idx]

                    lower_contour = lower_contour_resampled
                    upper_contour = upper_contour_resampled

                # Tính trọng số dựa trên khoảng cách
                weight = (missing_slice - lower_slice) / (upper_slice - lower_slice)

                # Nội suy tuyến tính
                interpolated_contour = (
                    lower_contour * (1 - weight) + upper_contour * weight
                )

                # Thêm contour nội suy
                self.add_contour(missing_slice, interpolated_contour)

            self.update_metadata("interpolated", True)
            self.update_metadata(
                "interpolation_date", datetime.datetime.now().isoformat()
            )

            return True

        except Exception as e:
            logger.error(f"Lỗi khi nội suy contours: {str(e)}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")

        return False


class ContourSet:
    """
    Lớp quản lý tập hợp các contour cho nhiều cấu trúc.

    Attributes
    ----------
    name : str
        Tên của tập contour
    description : str
        Mô tả về tập contour
    id : str
        ID duy nhất của tập contour
    contours : Dict[str, Dict[int, np.ndarray]]
        Từ điển ánh xạ tên cấu trúc với từ điển contour
    structure_info : Dict[str, Dict[str, Any]]
        Thông tin về các cấu trúc (màu, metadata, v.v.)
    creation_date : str
        Ngày giờ tạo tập contour
    last_modified : str
        Ngày giờ chỉnh sửa tập contour lần cuối
    """

    def __init__(self, name: str = "ContourSet", description: str = ""):
        """
        Khởi tạo đối tượng ContourSet.

        Parameters
        ----------
        name : str, optional
            Tên của tập contour, mặc định "ContourSet"
        description : str, optional
            Mô tả về tập contour
        """
        self.name = name
        self.description = description
        self.id = str(uuid.uuid4())

        # Lưu trữ contour: {'cấu trúc_name': {slice_idx: contour_points}}
        self.contours = {}  # Dict[str, Dict[int, np.ndarray]]

        # Thông tin về cấu trúc: {'cấu trúc_name': {'color': (r,g,b), 'metadata': {...}}}
        self.structure_info = {}  # Dict[str, Dict[str, Any]]

        self.creation_date = datetime.datetime.now().isoformat()
        self.last_modified = self.creation_date

    def add_structure(
        self,
        structure_name: str,
        color: Optional[Tuple[int, int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Thêm một cấu trúc mới.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc
        color : Tuple[int, int, int], optional
            Màu RGB của cấu trúc
        metadata : Dict[str, Any], optional
            Metadata bổ sung cho cấu trúc
        """
        if structure_name in self.contours:
            logger.warning(
                f"Cấu trúc '{structure_name}' đã tồn tại. Cập nhật thông tin."
            )
        else:
            self.contours[structure_name] = {}

        # Cập nhật thông tin cấu trúc
        if structure_name not in self.structure_info:
            self.structure_info[structure_name] = {}

        # Cập nhật màu sắc
        if color:
            self.structure_info[structure_name]["color"] = color
        elif "color" not in self.structure_info[structure_name]:
            # Gán màu ngẫu nhiên nếu không có
            import random

            self.structure_info[structure_name]["color"] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        # Cập nhật metadata
        if not metadata:
            metadata = {}
        if "metadata" not in self.structure_info[structure_name]:
            self.structure_info[structure_name]["metadata"] = {}
        self.structure_info[structure_name]["metadata"].update(metadata)

        # Cập nhật thời gian
        self.last_modified = datetime.datetime.now().isoformat()
        self.structure_info[structure_name]["last_modified"] = self.last_modified

    def remove_structure(self, structure_name: str) -> bool:
        """
        Xóa một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if structure_name in self.contours:
            del self.contours[structure_name]
            if structure_name in self.structure_info:
                del self.structure_info[structure_name]
            self.last_modified = datetime.datetime.now().isoformat()
            return True
        return False

    def set_contour(
        self, structure_name: str, slice_idx: int, contour_points: np.ndarray
    ) -> None:
        """
        Thiết lập contour cho một cấu trúc và lát cắt cụ thể.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc
        slice_idx : int
            Chỉ số lát cắt
        contour_points : np.ndarray
            Dữ liệu contour, mảng nx2 chứa tọa độ (x, y)
        """
        # Đảm bảo cấu trúc tồn tại
        if structure_name not in self.contours:
            logger.warning(
                f"Cấu trúc '{structure_name}' không tồn tại. Tạo cấu trúc mới."
            )
            self.add_structure(structure_name)

        # Kiểm tra và chuyển đổi contour_points
        if contour_points is None or (
            isinstance(contour_points, np.ndarray) and len(contour_points) < 3
        ):
            logger.warning(
                f"Bỏ qua contour không hợp lệ cho {structure_name}, lát cắt {slice_idx}"
            )
            # Xóa contour hiện tại nếu có
            if slice_idx in self.contours[structure_name]:
                del self.contours[structure_name][slice_idx]
            return

        # Chuyển đổi sang numpy array nếu cần
        if not isinstance(contour_points, np.ndarray):
            try:
                contour_points = np.array(contour_points)
            except Exception as e:
                logger.error(
                    f"Không thể chuyển đổi contour_points sang numpy array: {e}"
                )
                return

        # Đảm bảo định dạng contour là nx2
        if contour_points.ndim != 2 or contour_points.shape[1] < 2:
            logger.warning(
                f"Contour_points phải có định dạng nx2, nhưng có shape {contour_points.shape}"
            )
            return

        # Lưu contour
        self.contours[structure_name][slice_idx] = contour_points[:, :2].copy()

        # Cập nhật thời gian
        now = datetime.datetime.now().isoformat()
        self.last_modified = now
        if structure_name in self.structure_info:
            self.structure_info[structure_name]["last_modified"] = now

    def get_contour(self, structure_name: str, slice_idx: int) -> Optional[np.ndarray]:
        """
        Lấy contour cho một cấu trúc và lát cắt cụ thể.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc
        slice_idx : int
            Chỉ số lát cắt

        Returns
        -------
        np.ndarray or None
            Dữ liệu contour hoặc None nếu không tìm thấy
        """
        if structure_name not in self.contours:
            logger.warning(
                f"Cấu trúc '{structure_name}' không tồn tại trong tập contour '{self.name}'"
            )
            return None

        return self.contours[structure_name].get(slice_idx)

    def get_structure_slices(self, structure_name: str) -> List[int]:
        """
        Lấy danh sách các lát cắt có contour cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc

        Returns
        -------
        List[int]
            Danh sách chỉ số lát cắt hoặc danh sách trống nếu không tìm thấy cấu trúc
        """
        if structure_name not in self.contours:
            return []

        return sorted(list(self.contours[structure_name].keys()))

    def get_structure_contours(self, structure_name: str) -> Dict[int, np.ndarray]:
        """
        Lấy tất cả contour cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc

        Returns
        -------
        Dict[int, np.ndarray]
            Từ điển ánh xạ chỉ số lát cắt với dữ liệu contour
        """
        if structure_name not in self.contours:
            logger.warning(
                f"Cấu trúc '{structure_name}' không tồn tại trong tập contour '{self.name}'"
            )
            return {}

        return self.contours[structure_name]

    def get_structure_info(self, structure_name: str) -> Dict[str, Any]:
        """
        Lấy thông tin về một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc

        Returns
        -------
        Dict[str, Any]
            Thông tin về cấu trúc
        """
        if structure_name not in self.structure_info:
            return {}

        return self.structure_info[structure_name]

    def get_structure_names(self) -> List[str]:
        """
        Lấy danh sách tên tất cả các cấu trúc.

        Returns
        -------
        List[str]
            Danh sách tên cấu trúc
        """
        return list(self.contours.keys())

    def has_structure(self, structure_name: str) -> bool:
        """
        Kiểm tra xem một cấu trúc có tồn tại không.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc

        Returns
        -------
        bool
            True nếu cấu trúc tồn tại, False nếu không
        """
        return structure_name in self.contours

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi ContourSet thành từ điển để lưu trữ.

        Returns
        -------
        Dict[str, Any]
            Từ điển biểu diễn ContourSet
        """
        try:
            # Chuyển đổi contours thành định dạng có thể JSON hóa
            serialized_contours = {}
            for structure_name, structure_contours in self.contours.items():
                serialized_contours[structure_name] = {}
                for slice_idx, contour in structure_contours.items():
                    serialized_contours[structure_name][str(slice_idx)] = (
                        contour.tolist()
                    )

            return {
                "name": self.name,
                "description": self.description,
                "id": self.id,
                "contours": serialized_contours,
                "structure_info": self.structure_info,
                "creation_date": self.creation_date,
                "last_modified": self.last_modified,
            }
        except Exception as e:
            logger.error(f"Lỗi khi xuất ContourSet sang dict: {e}")
            # Trả về một từ điển tối thiểu
            return {
                "name": self.name,
                "description": self.description,
                "id": self.id,
                "error": str(e),
            }

    @classmethod
    def import_from_dict(cls, data: Dict[str, Any]) -> "ContourSet":
        """
        Tạo ContourSet từ từ điển.

        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển biểu diễn ContourSet

        Returns
        -------
        ContourSet
            Đối tượng ContourSet được tạo
        """
        contour_set = cls(
            name=data.get("name", "Imported"), description=data.get("description", "")
        )

        try:
            # Thiết lập các thuộc tính từ từ điển
            contour_set.id = data.get("id", str(uuid.uuid4()))
            contour_set.creation_date = data.get(
                "creation_date", contour_set.creation_date
            )
            contour_set.last_modified = data.get(
                "last_modified", contour_set.last_modified
            )

            # Import thông tin cấu trúc
            contour_set.structure_info = data.get("structure_info", {})

            # Import contours (chuyển đổi từ list về numpy arrays)
            serialized_contours = data.get("contours", {})
            for structure_name, slices in serialized_contours.items():
                contour_set.contours[structure_name] = {}
                for slice_idx_str, contour_list in slices.items():
                    try:
                        slice_idx = int(slice_idx_str)
                        contour_set.contours[structure_name][slice_idx] = np.array(
                            contour_list
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Không thể import contour cho {structure_name}, lát cắt {slice_idx_str}: {e}"
                        )

        except Exception as e:
            logger.error(f"Lỗi khi import ContourSet từ dict: {e}")

        return contour_set

    def save_to_json(self, filepath: str) -> bool:
        """
        Lưu ContourSet vào file JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để lưu

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu có lỗi
        """
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Xuất dữ liệu và lưu
            data = self.export_to_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Đã lưu tập contour '{self.name}' vào {filepath}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu tập contour vào {filepath}: {e}")
            return False

    @classmethod
    def load_from_json(cls, filepath: str) -> Optional["ContourSet"]:
        """
        Tải ContourSet từ file JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để tải

        Returns
        -------
        ContourSet or None
            Đối tượng ContourSet hoặc None nếu có lỗi
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            contour_set = cls.import_from_dict(data)
            logger.info(f"Đã tải tập contour '{contour_set.name}' từ {filepath}")
            return contour_set
        except Exception as e:
            logger.error(f"Lỗi khi tải tập contour từ {filepath}: {e}")
            return None

    def merge(self, other: "ContourSet", override: bool = False) -> None:
        """
        Kết hợp với một ContourSet khác.

        Parameters
        ----------
        other : ContourSet
            Tập contour khác để kết hợp
        override : bool, optional
            Nếu True, ghi đè các cấu trúc trùng tên; nếu False, bỏ qua (giữ nguyên của self)
        """
        if not isinstance(other, ContourSet):
            logger.error("Đối tượng để merge không phải ContourSet")
            return

        # Kết hợp các cấu trúc
        for structure_name in other.get_structure_names():
            # Nếu cấu trúc đã tồn tại và không ghi đè, bỏ qua
            if structure_name in self.contours and not override:
                logger.info(f"Bỏ qua cấu trúc '{structure_name}' đã tồn tại")
                continue

            # Thêm cấu trúc mới hoặc ghi đè
            structure_info = other.get_structure_info(structure_name)
            self.add_structure(
                structure_name,
                color=structure_info.get("color"),
                metadata=structure_info.get("metadata", {}),
            )

            # Sao chép các contour
            for slice_idx, contour in other.get_structure_contours(
                structure_name
            ).items():
                self.set_contour(structure_name, slice_idx, contour)

        # Cập nhật thời gian
        self.last_modified = datetime.datetime.now().isoformat()
