#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý hình ảnh y tế của bệnh nhân trong QuangTPS.

Module này cung cấp các lớp và hàm để tải, xử lý và hiển thị
hình ảnh y tế như CT, MRI, PET, v.v. với đầy đủ metadata.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import os
import SimpleITK as sitk
from enum import Enum, auto
import datetime
import uuid

from quangtps.core.logging import get_logger
from quangtps.core.exceptions import ImageError

logger = get_logger(__name__)


class ImageModality(str, Enum):
    """Các loại hình thức hình ảnh y tế."""

    CT = "CT"
    MR = "MR"
    PT = "PT"  # PET
    RTDOSE = "RTDOSE"
    CBCT = "CBCT"
    RTPLAN = "RTPLAN"
    RTSTRUCT = "RTSTRUCT"
    RTIMAGE = "RTIMAGE"
    US = "US"
    UNKNOWN = "UNKNOWN"


class ImageOrientation:
    """Các hướng chuẩn của hình ảnh trong không gian."""

    AXIAL = "AXIAL"
    SAGITTAL = "SAGITTAL"
    CORONAL = "CORONAL"
    OBLIQUE = "OBLIQUE"


class Image:
    """
    Lớp biểu diễn một hình ảnh y tế 3D với đầy đủ metadata.

    Lớp này lưu trữ dữ liệu hình ảnh 3D và các thông tin liên quan
    như kích thước voxel, thông tin không gian, và các thông số
    hiển thị cần thiết để hiển thị và xử lý hình ảnh y tế.
    """

    def __init__(
        self,
        id: str = None,
        modality: Union[str, ImageModality] = ImageModality.CT,
        data: Optional[np.ndarray] = None,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        slice_thickness: float = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, ...] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    ):
        """
        Khởi tạo đối tượng Image.

        Parameters
        ----------
        id : str, optional
            ID của hình ảnh, mặc định là None (tự động tạo UUID)
        modality : Union[str, ImageModality], optional
            Loại hình ảnh, mặc định là CT
        data : Optional[np.ndarray], optional
            Mảng 3D chứa dữ liệu hình ảnh, mặc định là None
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách giữa các pixel theo trục x, y (mm), mặc định là (1.0, 1.0)
        slice_thickness : float, optional
            Độ dày lát cắt (mm), mặc định là 1.0
        origin : Tuple[float, float, float], optional
            Tọa độ gốc của hình ảnh trong không gian (mm), mặc định là (0.0, 0.0, 0.0)
        direction : Tuple[float, ...], optional
            Ma trận hướng 3x3 duỗi thành tuple, mặc định là (1, 0, 0, 0, 1, 0, 0, 0, 1)
        """
        # Tự động tạo ID nếu không được cung cấp
        self.id = id if id is not None else str(uuid.uuid4())

        # Xử lý modality
        if isinstance(modality, str):
            try:
                self.modality = ImageModality(modality)
            except ValueError:
                self.modality = ImageModality.UNKNOWN
                logger.warning(
                    f"Không nhận dạng được modality '{modality}', sử dụng UNKNOWN"
                )
        else:
            self.modality = modality

        # Khởi tạo dữ liệu hình ảnh
        self.data = data if data is not None else np.zeros((1, 1, 1), dtype=np.float32)

        # Thông tin không gian
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
        self.origin = origin
        self.direction = direction

        # Tính toán kích thước dựa trên dữ liệu
        if self.data is not None:
            self.dimensions = self.data.shape
        else:
            self.dimensions = (0, 0, 0)

        # Thông tin mô tả
        self.description = ""
        self.series_description = ""
        self.study_description = ""
        self.acquisition_date = None
        self.acquisition_time = None
        self.series_number = 0
        self.instance_number = 0
        self.accession_number = ""
        self.study_id = ""
        self.series_id = ""

        # Thông tin bệnh nhân
        self.patient_id = ""
        self.patient_name = ""
        self.patient_birth_date = None
        self.patient_sex = ""

        # Thông tin hiển thị
        self.window_center = 0
        self.window_width = 0
        self.rescale_intercept = 0
        self.rescale_slope = 1

        # Calculated Hounsfield units (for CT) or other intensity values
        self._hu_data = None

        # Metadata khác
        self.metadata = {}
        self.sitk_image = None  # Lưu trữ đối tượng SimpleITK.Image nếu cần

        # Gán giá trị mặc định cho window/level dựa trên modality
        self._set_default_window_level()

    def _set_default_window_level(self):
        """Thiết lập giá trị mặc định cho window/level dựa trên modality."""
        if self.modality == ImageModality.CT:
            self.window_center = 40  # Lung window
            self.window_width = 400
        elif self.modality == ImageModality.MR:
            if self.data is not None and len(self.data.flatten()) > 0:
                # Với MRI, mặc định là giá trị trung bình +/- 3 độ lệch chuẩn
                flat_data = self.data.flatten()
                mean_val = np.mean(flat_data)
                std_val = np.std(flat_data)
                self.window_center = mean_val
                self.window_width = std_val * 6  # +/- 3 std
            else:
                self.window_center = 500
                self.window_width = 1000
        elif self.modality == ImageModality.PT:  # PET
            self.window_center = 2.5  # SUV
            self.window_width = 5.0
        elif self.modality == ImageModality.RTDOSE:
            self.window_center = 50  # 50% của liều tối đa
            self.window_width = 100  # 0-100% range
        else:
            # Default values for other modalities
            if self.data is not None and len(self.data.flatten()) > 0:
                min_val = np.min(self.data)
                max_val = np.max(self.data)
                self.window_center = (min_val + max_val) / 2
                self.window_width = max_val - min_val
            else:
                self.window_center = 100
                self.window_width = 200

    def get_hu_data(self) -> np.ndarray:
        """
        Lấy dữ liệu hình ảnh được chuyển đổi thành đơn vị Hounsfield (HU) cho CT.
        Với các modality khác, trả về dữ liệu đã hiệu chỉnh.

        Returns
        -------
        np.ndarray
            Mảng dữ liệu đã chuyển đổi
        """
        if self._hu_data is None and self.data is not None:
            # Chuyển đổi dữ liệu thô thành đơn vị Hounsfield (chỉ áp dụng cho CT)
            self._hu_data = self.data * self.rescale_slope + self.rescale_intercept
        return self._hu_data

    def invalidate_hu_cache(self):
        """Xóa cache dữ liệu HU khi dữ liệu gốc thay đổi."""
        self._hu_data = None

    def get_array(self) -> np.ndarray:
        """
        Lấy mảng dữ liệu hình ảnh gốc.

        Returns
        -------
        np.ndarray
            Mảng numpy chứa dữ liệu hình ảnh
        """
        return self.data

    def set_array(self, array: np.ndarray) -> None:
        """
        Cập nhật mảng dữ liệu hình ảnh.

        Parameters
        ----------
        array : np.ndarray
            Mảng dữ liệu hình ảnh mới
        """
        self.data = array
        self.dimensions = array.shape if array is not None else (0, 0, 0)
        # Xóa cache dữ liệu HU
        self.invalidate_hu_cache()

    def get_value_at_idx(self, x: int, y: int, z: int) -> float:
        """
        Lấy giá trị pixel tại vị trí chỉ số (x, y, z).

        Parameters
        ----------
        x : int
            Chỉ số trục x
        y : int
            Chỉ số trục y
        z : int
            Chỉ số trục z

        Returns
        -------
        float
            Giá trị pixel tại vị trí đó
        """
        if self.data is None:
            return 0.0

        if (
            x < 0
            or y < 0
            or z < 0
            or x >= self.dimensions[0]
            or y >= self.dimensions[1]
            or z >= self.dimensions[2]
        ):
            return 0.0

        return float(self.data[x, y, z])

    def get_value_at_point(self, point: Tuple[float, float, float]) -> float:
        """
        Lấy giá trị pixel tại vị trí tọa độ thực (mm).

        Parameters
        ----------
        point : Tuple[float, float, float]
            Tọa độ điểm (x, y, z) trong không gian thực (mm)

        Returns
        -------
        float
            Giá trị nội suy tại điểm đó
        """
        if self.data is None:
            return 0.0

        # Chuyển từ tọa độ không gian thực (mm) sang chỉ số
        x_pos = (point[0] - self.origin[0]) / self.pixel_spacing[0]
        y_pos = (point[1] - self.origin[1]) / self.pixel_spacing[1]
        z_pos = (point[2] - self.origin[2]) / self.slice_thickness

        # Làm tròn để lấy chỉ số gần nhất
        x_idx = int(round(x_pos))
        y_idx = int(round(y_pos))
        z_idx = int(round(z_pos))

        # Kiểm tra nếu nằm ngoài hình ảnh
        if (
            x_idx < 0
            or y_idx < 0
            or z_idx < 0
            or x_idx >= self.dimensions[0]
            or y_idx >= self.dimensions[1]
            or z_idx >= self.dimensions[2]
        ):
            return 0.0

        return float(self.data[x_idx, y_idx, z_idx])

    def get_interpolated_value(self, point: Tuple[float, float, float]) -> float:
        """
        Lấy giá trị pixel tại vị trí tọa độ thực (mm) với nội suy tuyến tính.

        Parameters
        ----------
        point : Tuple[float, float, float]
            Tọa độ điểm (x, y, z) trong không gian thực (mm)

        Returns
        -------
        float
            Giá trị nội suy tại điểm đó
        """
        if self.data is None:
            return 0.0

        # Chuyển từ tọa độ không gian thực (mm) sang chỉ số
        x_pos = (point[0] - self.origin[0]) / self.pixel_spacing[0]
        y_pos = (point[1] - self.origin[1]) / self.pixel_spacing[1]
        z_pos = (point[2] - self.origin[2]) / self.slice_thickness

        # Lấy chỉ số nguyên và phần thập phân
        x0 = int(x_pos)
        y0 = int(y_pos)
        z0 = int(z_pos)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Phần thập phân cho nội suy
        dx = x_pos - x0
        dy = y_pos - y0
        dz = z_pos - z0

        # Kiểm tra giới hạn
        if (
            x0 < 0
            or x1 >= self.dimensions[0]
            or y0 < 0
            or y1 >= self.dimensions[1]
            or z0 < 0
            or z1 >= self.dimensions[2]
        ):
            return 0.0

        # Nội suy tuyến tính 3D
        c00 = self.data[x0, y0, z0] * (1 - dx) + self.data[x1, y0, z0] * dx
        c01 = self.data[x0, y0, z1] * (1 - dx) + self.data[x1, y0, z1] * dx
        c10 = self.data[x0, y1, z0] * (1 - dx) + self.data[x1, y1, z0] * dx
        c11 = self.data[x0, y1, z1] * (1 - dx) + self.data[x1, y1, z1] * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        return float(c0 * (1 - dz) + c1 * dz)

    def get_slice(
        self, idx: int, orientation: str = ImageOrientation.AXIAL
    ) -> np.ndarray:
        """
        Lấy một lát cắt 2D từ hình ảnh 3D.

        Parameters
        ----------
        idx : int
            Chỉ số lát cắt
        orientation : str, optional
            Hướng cắt ("AXIAL", "SAGITTAL", "CORONAL"), mặc định là "AXIAL"

        Returns
        -------
        np.ndarray
            Mảng 2D chứa dữ liệu lát cắt
        """
        if self.data is None:
            return np.zeros((1, 1))

        try:
            if orientation.upper() == ImageOrientation.AXIAL:
                if 0 <= idx < self.dimensions[2]:
                    return self.data[:, :, idx]
                else:
                    logger.warning(
                        f"Chỉ số lát cắt {idx} nằm ngoài phạm vi [0, {self.dimensions[2] - 1}]"
                    )
                    return np.zeros((self.dimensions[0], self.dimensions[1]))

            elif orientation.upper() == ImageOrientation.SAGITTAL:
                if 0 <= idx < self.dimensions[0]:
                    return self.data[idx, :, :]
                else:
                    logger.warning(
                        f"Chỉ số lát cắt {idx} nằm ngoài phạm vi [0, {self.dimensions[0] - 1}]"
                    )
                    return np.zeros((self.dimensions[1], self.dimensions[2]))

            elif orientation.upper() == ImageOrientation.CORONAL:
                if 0 <= idx < self.dimensions[1]:
                    return self.data[:, idx, :]
                else:
                    logger.warning(
                        f"Chỉ số lát cắt {idx} nằm ngoài phạm vi [0, {self.dimensions[1] - 1}]"
                    )
                    return np.zeros((self.dimensions[0], self.dimensions[2]))

            else:
                logger.error(f"Hướng không hợp lệ: {orientation}")
                return np.zeros((1, 1))

        except Exception as e:
            logger.error(f"Lỗi khi lấy lát cắt: {str(e)}")
            return np.zeros((1, 1))

    def get_window_level(self) -> Tuple[float, float]:
        """
        Lấy thông số window/level.

        Returns
        -------
        Tuple[float, float]
            Tuple chứa (window_width, window_center)
        """
        return (self.window_width, self.window_center)

    def set_window_level(self, window_width: float, window_center: float):
        """
        Thiết lập thông số window/level.

        Parameters
        ----------
        window_width : float
            Độ rộng cửa sổ
        window_center : float
            Vị trí trung tâm cửa sổ
        """
        self.window_width = window_width
        self.window_center = window_center

    def adjust_contrast(self, data: np.ndarray) -> np.ndarray:
        """
        Điều chỉnh độ tương phản dựa trên window/level.

        Parameters
        ----------
        data : np.ndarray
            Dữ liệu hình ảnh đầu vào

        Returns
        -------
        np.ndarray
            Dữ liệu hình ảnh đã điều chỉnh (0-1)
        """
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2

        # Clip và chuẩn hóa về [0, 1]
        result = np.clip(data, min_val, max_val)
        result = (result - min_val) / (max_val - min_val)

        return result

    def to_sitk_image(self) -> sitk.Image:
        """
        Chuyển đổi sang đối tượng SimpleITK.Image.

        Returns
        -------
        sitk.Image
            Đối tượng SimpleITK.Image
        """
        if self.data is None:
            return sitk.Image(1, 1, 1, sitk.sitkFloat32)

        # Chuyển đổi từ numpy sang SimpleITK
        sitk_image = sitk.GetImageFromArray(self.data.astype(np.float32))

        # Thiết lập thông tin không gian
        sitk_image.SetSpacing(
            (self.pixel_spacing[0], self.pixel_spacing[1], self.slice_thickness)
        )
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetDirection(self.direction)

        # Lưu lại đối tượng SimpleITK
        self.sitk_image = sitk_image

        return sitk_image

    @classmethod
    def from_sitk_image(
        cls, sitk_image: sitk.Image, id: Optional[str] = None, modality: str = "CT"
    ) -> "Image":
        """
        Tạo đối tượng Image từ SimpleITK.Image.

        Parameters
        ----------
        sitk_image : sitk.Image
            Đối tượng SimpleITK.Image
        id : Optional[str], optional
            ID của hình ảnh, mặc định là None (tự động tạo)
        modality : str, optional
            Loại hình ảnh, mặc định là "CT"

        Returns
        -------
        Image
            Đối tượng Image mới
        """
        # Lấy dữ liệu và thông tin không gian
        data = sitk.GetArrayFromImage(sitk_image)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()

        # Trích xuất thông tin không gian
        pixel_spacing = (spacing[0], spacing[1])
        slice_thickness = spacing[2] if len(spacing) > 2 else 1.0

        # Tạo đối tượng Image mới
        image = cls(
            id=id,
            modality=modality,
            data=data,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            origin=origin,
            direction=direction,
        )

        # Lưu đối tượng SimpleITK
        image.sitk_image = sitk_image

        return image

    def resample(
        self,
        new_spacing: Tuple[float, float, float],
        interpolator: Optional[int] = None,
    ) -> "Image":
        """
        Lấy mẫu lại hình ảnh với khoảng cách voxel mới.

        Parameters
        ----------
        new_spacing : Tuple[float, float, float]
            Khoảng cách voxel mới (mm)
            interpolator : Optional[int], optional
            Phương pháp nội suy, mặc định là None (sitk.sitkLinear)

        Returns
        -------
        Image
            Hình ảnh đã lấy mẫu lại
        """
        # Thiết lập interpolator mặc định
        if interpolator is None:
            interpolator = sitk.sitkLinear

        # Chuyển đổi sang SimpleITK nếu chưa có
        if self.sitk_image is None:
            self.to_sitk_image()

        # Tính toán kích thước mới
        old_size = self.sitk_image.GetSize()
        old_spacing = self.sitk_image.GetSpacing()

        new_size = [
            int(round(old_size[0] * old_spacing[0] / new_spacing[0])),
            int(round(old_size[1] * old_spacing[1] / new_spacing[1])),
            int(round(old_size[2] * old_spacing[2] / new_spacing[2])),
        ]

        # Tạo bộ lọc lấy mẫu lại
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(self.sitk_image.GetDirection())
        resample.SetOutputOrigin(self.sitk_image.GetOrigin())
        resample.SetDefaultPixelValue(0)

        # Thực hiện lấy mẫu lại
        resampled_sitk = resample.Execute(self.sitk_image)

        # Tạo đối tượng Image mới
        resampled_image = Image.from_sitk_image(
            resampled_sitk, id=f"{self.id}_resampled", modality=self.modality
        )

        # Sao chép các thuộc tính khác
        resampled_image.description = self.description
        resampled_image.series_description = self.series_description
        resampled_image.study_description = self.study_description
        resampled_image.acquisition_date = self.acquisition_date
        resampled_image.acquisition_time = self.acquisition_time
        resampled_image.patient_id = self.patient_id
        resampled_image.patient_name = self.patient_name
        resampled_image.patient_birth_date = self.patient_birth_date
        resampled_image.patient_sex = self.patient_sex
        resampled_image.window_center = self.window_center
        resampled_image.window_width = self.window_width
        resampled_image.rescale_intercept = self.rescale_intercept
        resampled_image.rescale_slope = self.rescale_slope
        resampled_image.metadata = self.metadata.copy()

        return resampled_image

    def apply_threshold(self, min_val: float, max_val: float) -> "Image":
        """
        Áp dụng ngưỡng lên hình ảnh.

        Parameters
        ----------
        min_val : float
            Giá trị ngưỡng tối thiểu
        max_val : float
            Giá trị ngưỡng tối đa

        Returns
        -------
        Image
            Hình ảnh đã áp dụng ngưỡng
        """
        if self.data is None:
            return self

        # Tạo bản sao dữ liệu
        thresholded_data = self.data.copy()

        # Áp dụng ngưỡng (gán giá trị ngoài ngưỡng thành 0)
        thresholded_data[
            (thresholded_data < min_val) | (thresholded_data > max_val)
        ] = 0

        # Tạo đối tượng Image mới
        result = Image(
            id=f"{self.id}_threshold",
            modality=self.modality,
            data=thresholded_data,
            pixel_spacing=self.pixel_spacing,
            slice_thickness=self.slice_thickness,
            origin=self.origin,
            direction=self.direction,
        )

        # Sao chép các thuộc tính khác
        result.description = self.description
        result.series_description = self.series_description
        result.window_center = self.window_center
        result.window_width = self.window_width

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng Image thành từ điển.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin của đối tượng Image
        """
        # Không bao gồm các mảng lớn như data và sitk_image
        return {
            "id": self.id,
            "modality": str(self.modality),
            "description": self.description,
            "series_description": self.series_description,
            "study_description": self.study_description,
            "acquisition_date": self.acquisition_date.isoformat()
            if self.acquisition_date
            else None,
            "acquisition_time": self.acquisition_time,
            "series_number": self.series_number,
            "instance_number": self.instance_number,
            "accession_number": self.accession_number,
            "study_id": self.study_id,
            "series_id": self.series_id,
            "pixel_spacing": self.pixel_spacing,
            "slice_thickness": self.slice_thickness,
            "origin": self.origin,
            "direction": self.direction,
            "dimensions": self.dimensions,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "patient_birth_date": self.patient_birth_date.isoformat()
            if self.patient_birth_date
            else None,
            "patient_sex": self.patient_sex,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "rescale_intercept": self.rescale_intercept,
            "rescale_slope": self.rescale_slope,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Image":
        """
        Tạo đối tượng Image từ từ điển.

        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin Image

        Returns
        -------
        Image
            Đối tượng Image mới
        """
        # Tạo đối tượng Image với các thông tin cơ bản
        image = cls(
            id=data.get("id"),
            modality=data.get("modality", "CT"),
            pixel_spacing=data.get("pixel_spacing", (1.0, 1.0)),
            slice_thickness=data.get("slice_thickness", 1.0),
            origin=data.get("origin", (0.0, 0.0, 0.0)),
            direction=data.get("direction", (1, 0, 0, 0, 1, 0, 0, 0, 1)),
        )

        # Cập nhật các thuộc tính khác
        image.description = data.get("description", "")
        image.series_description = data.get("series_description", "")
        image.study_description = data.get("study_description", "")

        # Xử lý các trường datetime
        if data.get("acquisition_date"):
            try:
                image.acquisition_date = datetime.datetime.fromisoformat(
                    data["acquisition_date"]
                )
            except (ValueError, TypeError):
                pass

        image.acquisition_time = data.get("acquisition_time")
        image.series_number = data.get("series_number", 0)
        image.instance_number = data.get("instance_number", 0)
        image.accession_number = data.get("accession_number", "")
        image.study_id = data.get("study_id", "")
        image.series_id = data.get("series_id", "")

        image.patient_id = data.get("patient_id", "")
        image.patient_name = data.get("patient_name", "")

        if data.get("patient_birth_date"):
            try:
                image.patient_birth_date = datetime.datetime.fromisoformat(
                    data["patient_birth_date"]
                )
            except (ValueError, TypeError):
                pass

        image.patient_sex = data.get("patient_sex", "")

        image.window_center = data.get("window_center", 0)
        image.window_width = data.get("window_width", 0)
        image.rescale_intercept = data.get("rescale_intercept", 0)
        image.rescale_slope = data.get("rescale_slope", 1)

        image.metadata = data.get("metadata", {})

        # Lưu ý: dữ liệu hình ảnh (data) không được lưu trong từ điển,
        # cần được tải từ nguồn khác

        return image


def create_empty_image(
    dimensions: Tuple[int, int, int] = (512, 512, 100),
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 1.0,
    modality: str = "CT",
) -> Image:
    """
    Tạo một đối tượng Image trống với kích thước xác định.

    Parameters
    ----------
    dimensions : Tuple[int, int, int], optional
        Kích thước hình ảnh (x, y, z), mặc định là (512, 512, 100)
    pixel_spacing : Tuple[float, float], optional
        Khoảng cách pixel (mm), mặc định là (1.0, 1.0)
    slice_thickness : float, optional
        Độ dày lát cắt (mm), mặc định là 1.0
    modality : str, optional
        Loại hình ảnh, mặc định là "CT"

    Returns
    -------
    Image
        Đối tượng Image mới
    """
    # Tạo mảng dữ liệu trống
    data = np.zeros(dimensions, dtype=np.float32)

    # Tạo đối tượng Image
    return Image(
        id=f"empty_image_{uuid.uuid4()}",
        modality=modality,
        data=data,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
    )
