"""
Quản lý lưới liều 3D.
"""

import numpy as np
import SimpleITK as sitk
import logging
import os
from enum import Enum
import pydicom
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime

from quangtps.core.exceptions import ValidationError, IOError
from quangtps.core.config import Config

logger = logging.getLogger(__name__)


class DoseUnit(Enum):
    """Đơn vị liều"""

    GY = "GY"
    CGY = "CGY"


class DoseGrid:
    """Lớp quản lý lưới liều 3D"""

    def __init__(self, grid_data=None, origin=None, spacing=None, direction=None):
        """
        Khởi tạo lưới liều.

        Parameters:
            grid_data (numpy.ndarray, optional): Dữ liệu lưới liều
            origin (tuple, optional): Tọa độ gốc (x, y, z)
            spacing (tuple, optional): Khoảng cách giữa các điểm lưới (dx, dy, dz)
            direction (tuple, optional): Ma trận hướng
        """
        self.grid_data = None
        self.sitk_image = None
        self.origin = origin or (0.0, 0.0, 0.0)
        self.spacing = spacing or (1.0, 1.0, 1.0)
        self.direction = direction or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.dose_unit = DoseUnit.GY

        if grid_data is not None:
            self.set_grid_data(grid_data)

    def set_grid_data(self, grid_data):
        """
        Đặt dữ liệu lưới liều.

        Parameters:
            grid_data (numpy.ndarray): Dữ liệu lưới liều 3D

        Raises:
            ValidationError: Nếu dữ liệu không hợp lệ
        """
        if not isinstance(grid_data, np.ndarray):
            raise ValidationError("Grid data must be a numpy.ndarray")

        if len(grid_data.shape) != 3:
            raise ValidationError("Grid data must be 3D")

        # Kiểm tra giá trị
        if np.any(np.isnan(grid_data)):
            logger.warning("Grid data contains NaN values. Converting to zeros.")
            grid_data = np.nan_to_num(grid_data, nan=0.0)

        self.grid_data = grid_data.astype(np.float32)

        # Tạo SimpleITK Image
        self.sitk_image = sitk.GetImageFromArray(self.grid_data)
        self.sitk_image.SetOrigin(self.origin)
        self.sitk_image.SetSpacing(self.spacing)
        self.sitk_image.SetDirection(self.direction)

    def get_grid_data(self):
        """
        Lấy dữ liệu lưới liều.

        Returns:
            numpy.ndarray: Dữ liệu lưới liều 3D
        """
        return self.grid_data

    def get_sitk_image(self):
        """
        Lấy SimpleITK Image.

        Returns:
            sitk.Image: SimpleITK Image
        """
        return self.sitk_image

    def get_shape(self):
        """
        Lấy kích thước lưới liều.

        Returns:
            tuple: Kích thước (depth, height, width)
        """
        if self.grid_data is None:
            return (0, 0, 0)

        return self.grid_data.shape

    def get_min_max(self):
        """
        Lấy giá trị nhỏ nhất và lớn nhất.

        Returns:
            tuple: (min, max)
        """
        if self.grid_data is None:
            return (0.0, 0.0)

        return (np.min(self.grid_data), np.max(self.grid_data))

    def get_dose_at_point(self, point):
        """
        Lấy giá trị liều tại một điểm.

        Parameters:
            point (tuple): Tọa độ điểm (x, y, z) trong không gian thực

        Returns:
            float: Giá trị liều
        """
        if self.sitk_image is None:
            return 0.0

        # Chuyển đổi từ tọa độ thực sang tọa độ lưới (trilinear interpolation)
        try:
            return float(sitk.Resample(self.sitk_image, point))
        except Exception as e:
            logger.error(f"Error getting dose at point {point}: {str(e)}")
            return 0.0

    def get_dose_at_index(self, idx):
        """
        Lấy giá trị liều tại một chỉ số.

        Parameters:
            idx (tuple): Chỉ số (i, j, k)

        Returns:
            float: Giá trị liều
        """
        if self.grid_data is None:
            return 0.0

        i, j, k = idx

        if (
            0 <= i < self.grid_data.shape[0]
            and 0 <= j < self.grid_data.shape[1]
            and 0 <= k < self.grid_data.shape[2]
        ):
            return float(self.grid_data[i, j, k])

        return 0.0

    def set_dose_at_index(self, idx, value):
        """
        Đặt giá trị liều tại một chỉ số.

        Parameters:
            idx (tuple): Chỉ số (i, j, k)
            value (float): Giá trị liều

        Returns:
            bool: True nếu thành công
        """
        if self.grid_data is None:
            return False

        i, j, k = idx

        if (
            0 <= i < self.grid_data.shape[0]
            and 0 <= j < self.grid_data.shape[1]
            and 0 <= k < self.grid_data.shape[2]
        ):
            self.grid_data[i, j, k] = value

            # Cập nhật SimpleITK Image
            self.sitk_image = sitk.GetImageFromArray(self.grid_data)
            self.sitk_image.SetOrigin(self.origin)
            self.sitk_image.SetSpacing(self.spacing)
            self.sitk_image.SetDirection(self.direction)

            return True

        return False

    def add_dose_grid(self, other_grid, weight=1.0):
        """
        Cộng với lưới liều khác.

        Parameters:
            other_grid (DoseGrid): Lưới liều khác
            weight (float, optional): Trọng số

        Returns:
            DoseGrid: Lưới liều sau khi cộng

        Raises:
            ValidationError: Nếu lưới liều không tương thích
        """
        if self.grid_data is None:
            # Nếu lưới hiện tại trống, sao chép từ lưới khác
            if other_grid.grid_data is not None:
                self.grid_data = other_grid.grid_data * weight
                self.origin = other_grid.origin
                self.spacing = other_grid.spacing
                self.direction = other_grid.direction

                # Cập nhật SimpleITK Image
                self.sitk_image = sitk.GetImageFromArray(self.grid_data)
                self.sitk_image.SetOrigin(self.origin)
                self.sitk_image.SetSpacing(self.spacing)
                self.sitk_image.SetDirection(self.direction)

            return self

        # Kiểm tra kích thước
        if self.grid_data.shape != other_grid.grid_data.shape:
            raise ValidationError("Grid shapes do not match")

        # Kiểm tra thông số không gian
        if (
            self.origin != other_grid.origin
            or self.spacing != other_grid.spacing
            or self.direction != other_grid.direction
        ):
            logger.warning("Spatial parameters do not match. Using resampling.")

            # Sử dụng resampling
            resampled_grid = sitk.Resample(other_grid.sitk_image, self.sitk_image)
            other_data = sitk.GetArrayFromImage(resampled_grid)
        else:
            other_data = other_grid.grid_data

        # Cộng dữ liệu
        self.grid_data += other_data * weight

        # Cập nhật SimpleITK Image
        self.sitk_image = sitk.GetImageFromArray(self.grid_data)
        self.sitk_image.SetOrigin(self.origin)
        self.sitk_image.SetSpacing(self.spacing)
        self.sitk_image.SetDirection(self.direction)

        return self

    def multiply(self, factor):
        """
        Nhân lưới liều với một hệ số.

        Parameters:
            factor (float): Hệ số nhân

        Returns:
            DoseGrid: Lưới liều sau khi nhân
        """
        if self.grid_data is not None:
            self.grid_data *= factor

            # Cập nhật SimpleITK Image
            self.sitk_image = sitk.GetImageFromArray(self.grid_data)
            self.sitk_image.SetOrigin(self.origin)
            self.sitk_image.SetSpacing(self.spacing)
            self.sitk_image.SetDirection(self.direction)

        return self

    def resample(self, reference_grid):
        """
        Lấy mẫu lại lưới liều theo lưới tham chiếu.

        Parameters:
            reference_grid (DoseGrid): Lưới tham chiếu

        Returns:
            DoseGrid: Lưới liều sau khi lấy mẫu lại

        Raises:
            ValidationError: Nếu lưới tham chiếu không hợp lệ
        """
        if self.sitk_image is None:
            raise ValidationError("Grid data is empty")

        if reference_grid.sitk_image is None:
            raise ValidationError("Reference grid data is empty")

        # Thực hiện resampling
        resampled_image = sitk.Resample(self.sitk_image, reference_grid.sitk_image)

        # Tạo lưới liều mới
        resampled_grid = DoseGrid()
        resampled_grid.grid_data = sitk.GetArrayFromImage(resampled_image)
        resampled_grid.sitk_image = resampled_image
        resampled_grid.origin = reference_grid.origin
        resampled_grid.spacing = reference_grid.spacing
        resampled_grid.direction = reference_grid.direction
        resampled_grid.dose_unit = self.dose_unit

        return resampled_grid

    def get_dose_volume_histogram(self, structure_mask, bins=100):
        """
        Tính histogram thể tích liều.

        Parameters:
            structure_mask (numpy.ndarray): Mask của cấu trúc
            bins (int, optional): Số lượng bins

        Returns:
            tuple: (dvh, bin_edges)
                dvh (numpy.ndarray): Histogram thể tích liều
                bin_edges (numpy.ndarray): Các cạnh của bin

        Raises:
            ValidationError: Nếu mask không tương thích
        """
        if self.grid_data is None:
            raise ValidationError("Grid data is empty")

        if structure_mask.shape != self.grid_data.shape:
            raise ValidationError("Structure mask shape does not match dose grid shape")

        # Lấy giá trị liều trong cấu trúc
        doses_in_structure = self.grid_data[structure_mask > 0]

        if len(doses_in_structure) == 0:
            logger.warning("No dose points found in structure")
            return np.zeros(bins), np.linspace(0, 1, bins + 1)

        # Tính histogram
        hist, bin_edges = np.histogram(doses_in_structure, bins=bins)

        # Chuyển đổi thành DVH (tích lũy)
        total_voxels = len(doses_in_structure)
        dvh = np.zeros(bins)

        for i in range(bins):
            # Số lượng voxel có liều >= bin_edge[i]
            dvh[i] = np.sum(doses_in_structure >= bin_edges[i]) / total_voxels * 100.0

        return dvh, bin_edges

    def get_mean_dose(self, structure_mask):
        """
        Tính liều trung bình trong cấu trúc.

        Parameters:
            structure_mask (numpy.ndarray): Mask của cấu trúc

        Returns:
            float: Liều trung bình

        Raises:
            ValidationError: Nếu mask không tương thích
        """
        if self.grid_data is None:
            raise ValidationError("Grid data is empty")

        if structure_mask.shape != self.grid_data.shape:
            raise ValidationError("Structure mask shape does not match dose grid shape")

        # Lấy giá trị liều trong cấu trúc
        doses_in_structure = self.grid_data[structure_mask > 0]

        if len(doses_in_structure) == 0:
            logger.warning("No dose points found in structure")
            return 0.0

        return float(np.mean(doses_in_structure))

    def get_max_dose(self, structure_mask=None):
        """
        Tính liều tối đa trong cấu trúc.

        Parameters:
            structure_mask (numpy.ndarray, optional): Mask của cấu trúc

        Returns:
            float: Liều tối đa
        """
        if self.grid_data is None:
            return 0.0

        if structure_mask is None:
            return float(np.max(self.grid_data))

        if structure_mask.shape != self.grid_data.shape:
            raise ValidationError("Structure mask shape does not match dose grid shape")

        # Lấy giá trị liều trong cấu trúc
        doses_in_structure = self.grid_data[structure_mask > 0]

        if len(doses_in_structure) == 0:
            logger.warning("No dose points found in structure")
            return 0.0

        return float(np.max(doses_in_structure))

    def get_min_dose(self, structure_mask):
        """
        Tính liều tối thiểu trong cấu trúc.

        Parameters:
            structure_mask (numpy.ndarray): Mask của cấu trúc

        Returns:
            float: Liều tối thiểu

        Raises:
            ValidationError: Nếu mask không tương thích
        """
        if self.grid_data is None:
            raise ValidationError("Grid data is empty")

        if structure_mask.shape != self.grid_data.shape:
            raise ValidationError("Structure mask shape does not match dose grid shape")

        # Lấy giá trị liều trong cấu trúc
        doses_in_structure = self.grid_data[structure_mask > 0]

        if len(doses_in_structure) == 0:
            logger.warning("No dose points found in structure")
            return 0.0

        return float(np.min(doses_in_structure))

    def get_dose_at_volume(self, structure_mask, volume_percent):
        """
        Tính liều tại thể tích phần trăm.

        Parameters:
            structure_mask (numpy.ndarray): Mask của cấu trúc
            volume_percent (float): Phần trăm thể tích (0-100)

        Returns:
            float: Giá trị liều

        Raises:
            ValidationError: Nếu mask không tương thích
        """
        if self.grid_data is None:
            raise ValidationError("Grid data is empty")

        if structure_mask.shape != self.grid_data.shape:
            raise ValidationError("Structure mask shape does not match dose grid shape")

        # Kiểm tra giá trị volume_percent
        if volume_percent < 0 or volume_percent > 100:
            raise ValidationError("Volume percent must be in range [0, 100]")

        # Lấy giá trị liều trong cấu trúc
        doses_in_structure = self.grid_data[structure_mask > 0]

        if len(doses_in_structure) == 0:
            logger.warning("No dose points found in structure")
            return 0.0

        # Sắp xếp giá trị liều
        sorted_doses = np.sort(doses_in_structure)

        # Tính chỉ số tương ứng với phần trăm thể tích
        index = int(len(sorted_doses) * (100 - volume_percent) / 100)
        if index >= len(sorted_doses):
            index = len(sorted_doses) - 1

        return float(sorted_doses[index])

    def get_volume_at_dose(self, structure_mask, dose_value):
        """
        Tính thể tích nhận liều >= dose_value.

        Parameters:
            structure_mask (numpy.ndarray): Mask của cấu trúc
            dose_value (float): Giá trị liều

        Returns:
            float: Phần trăm thể tích

        Raises:
            ValidationError: Nếu mask không tương thích
        """
        if self.grid_data is None:
            raise ValidationError("Grid data is empty")

        if structure_mask.shape != self.grid_data.shape:
            raise ValidationError("Structure mask shape does not match dose grid shape")

        # Lấy giá trị liều trong cấu trúc
        doses_in_structure = self.grid_data[structure_mask > 0]

        if len(doses_in_structure) == 0:
            logger.warning("No dose points found in structure")
            return 0.0

        # Tính số lượng voxel nhận liều >= dose_value
        count = np.sum(doses_in_structure >= dose_value)

        # Tính phần trăm
        percent = count / len(doses_in_structure) * 100.0

        return float(percent)

    @classmethod
    def create_from_reference(cls, reference_image, shape=None, spacing=None):
        """
        Tạo lưới liều từ hình ảnh tham chiếu.

        Parameters:
            reference_image (sitk.Image): Hình ảnh tham chiếu
            shape (tuple, optional): Kích thước lưới (depth, height, width)
            spacing (tuple, optional): Khoảng cách giữa các điểm lưới (dx, dy, dz)

        Returns:
            DoseGrid: Lưới liều
        """
        # Lấy thông tin không gian từ hình ảnh tham chiếu
        origin = reference_image.GetOrigin()
        direction = reference_image.GetDirection()

        # Nếu không chỉ định spacing thì sử dụng spacing của hình ảnh tham chiếu
        if spacing is None:
            spacing = reference_image.GetSpacing()

        # Tính kích thước mới dựa trên spacing
        ref_size = reference_image.GetSize()
        ref_spacing = reference_image.GetSpacing()

        if shape is None:
            # Tính kích thước lưới sao cho khớp với hình ảnh tham chiếu
            size_x = int(ref_size[0] * ref_spacing[0] / spacing[0] + 0.5)
            size_y = int(ref_size[1] * ref_spacing[1] / spacing[1] + 0.5)
            size_z = int(ref_size[2] * ref_spacing[2] / spacing[2] + 0.5)
            shape = (size_z, size_y, size_x)  # SimpleITK uses (x, y, z) order

        # Tạo lưới liều mới
        grid_data = np.zeros(shape, dtype=np.float32)

        # Tạo đối tượng DoseGrid
        dose_grid = cls(grid_data, origin, spacing, direction)

        return dose_grid

    def save_to_file(
        self, file_path, study_instance_uid=None, series_instance_uid=None
    ):
        """
        Lưu lưới liều thành file DICOM RT Dose.

        Parameters:
            file_path (str): Đường dẫn đến file
            study_instance_uid (str, optional): Study Instance UID
            series_instance_uid (str, optional): Series Instance UID

        Returns:
            bool: True nếu thành công

        Raises:
            IOError: Nếu không thể lưu file
        """
        if self.grid_data is None:
            raise ValidationError("Grid data is empty")

        try:
            # Chuẩn bị metadata
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = (
                "1.2.840.10008.5.1.4.1.1.481.2"  # RT Dose Storage
            )
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

            # Tạo dataset
            ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

            # Thêm các thuộc tính cần thiết
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"  # RT Dose Storage
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

            # Các thuộc tính về bệnh nhân và nghiên cứu
            ds.PatientName = "Anonymous"
            ds.PatientID = "Unknown"
            ds.StudyInstanceUID = study_instance_uid or pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = series_instance_uid or pydicom.uid.generate_uid()
            ds.StudyDate = datetime.now().strftime("%Y%m%d")
            ds.StudyTime = datetime.now().strftime("%H%M%S")

            # Thông tin về hình ảnh
            ds.Modality = "RTDOSE"
            ds.DoseUnits = self.dose_unit.value
            ds.DoseType = "PHYSICAL"
            ds.DoseGridScaling = (
                np.max(self.grid_data) / (2**16 - 1)
                if np.max(self.grid_data) > 0
                else 1.0
            )

            # Chuyển đổi dữ liệu
            if ds.DoseGridScaling != 0:
                scaled_data = self.grid_data / ds.DoseGridScaling
            else:
                scaled_data = self.grid_data

            # Thêm thông tin không gian
            ds.ImagePositionPatient = list(self.origin)
            ds.ImageOrientationPatient = list(self.direction)[
                :6
            ]  # Only need first 6 values
            ds.PixelSpacing = [self.spacing[0], self.spacing[1]]
            ds.SliceThickness = self.spacing[2]

            # Các thông tin khác
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.Rows = self.grid_data.shape[1]
            ds.Columns = self.grid_data.shape[2]
            ds.NumberOfFrames = self.grid_data.shape[0]
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            # Lưu dữ liệu pixel
            ds.PixelData = scaled_data.astype(np.uint16).tobytes()

            # Lưu file
            ds.save_as(file_path)

            logger.info(f"Dose grid saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving dose grid to file: {str(e)}")
            raise IOError(f"Error saving dose grid to file: {str(e)}")

    @classmethod
    def load_from_file(cls, file_path):
        """
        Tải lưới liều từ file DICOM RT Dose.

        Parameters:
            file_path (str): Đường dẫn đến file

        Returns:
            DoseGrid: Lưới liều

        Raises:
            IOError: Nếu không thể đọc file
        """
        try:
            # Đọc file DICOM
            ds = pydicom.dcmread(file_path)

            # Kiểm tra loại file
            if ds.SOPClassUID != "1.2.840.10008.5.1.4.1.1.481.2":  # RT Dose Storage
                raise ValidationError(
                    f"File is not an RT Dose (SOP Class UID: {ds.SOPClassUID})"
                )

            # Lấy thông tin không gian
            origin = tuple(map(float, ds.ImagePositionPatient))
            spacing = (
                float(ds.PixelSpacing[0]),
                float(ds.PixelSpacing[1]),
                float(ds.SliceThickness),
            )

            # Lấy ma trận hướng
            orientation = list(map(float, ds.ImageOrientationPatient))
            direction = (
                orientation[0],
                orientation[1],
                orientation[2],
                orientation[3],
                orientation[4],
                orientation[5],
                0.0,
                0.0,
                1.0,  # Assume standard orientation for the third axis
            )

            # Lấy dữ liệu pixel
            pixel_data = ds.pixel_array

            # Áp dụng DoseGridScaling
            if hasattr(ds, "DoseGridScaling"):
                pixel_data = pixel_data.astype(np.float32) * float(ds.DoseGridScaling)

            # Chuyển đổi thành lưới 3D
            if hasattr(ds, "NumberOfFrames"):
                num_frames = int(ds.NumberOfFrames)
                grid_data = pixel_data.reshape((num_frames, ds.Rows, ds.Columns))
            else:
                # Nếu không có NumberOfFrames, giả sử là 1 frame
                grid_data = pixel_data.reshape((1, ds.Rows, ds.Columns))

            # Tạo lưới liều
            dose_grid = cls(grid_data, origin, spacing, direction)

            # Thiết lập đơn vị liều
            if hasattr(ds, "DoseUnits"):
                try:
                    dose_grid.dose_unit = DoseUnit(ds.DoseUnits)
                except ValueError:
                    logger.warning(f"Unknown dose unit: {ds.DoseUnits}. Using GY.")
                    dose_grid.dose_unit = DoseUnit.GY

            logger.info(f"Dose grid loaded from {file_path}")
            return dose_grid

        except Exception as e:
            logger.error(f"Error loading dose grid from file: {str(e)}")
            raise IOError(f"Error loading dose grid from file: {str(e)}")

    @classmethod
    def create_empty_grid(
        cls,
        shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    ):
        """
        Tạo lưới liều trống.

        Parameters:
            shape (tuple): Kích thước lưới (depth, height, width)
            origin (tuple, optional): Tọa độ gốc (x, y, z)
            spacing (tuple, optional): Khoảng cách giữa các điểm lưới (dx, dy, dz)
            direction (tuple, optional): Ma trận hướng

        Returns:
            DoseGrid: Lưới liều
        """
        grid_data = np.zeros(shape, dtype=np.float32)
        return cls(grid_data, origin, spacing, direction)

    def get_dose_array(self):
        """
        Lấy dữ liệu lưới liều (alias cho get_grid_data).

        Returns:
            numpy.ndarray: Dữ liệu lưới liều 3D
        """
        return self.get_grid_data()

    def get_spacing(self):
        """
        Lấy khoảng cách giữa các điểm lưới.

        Returns:
            tuple: Spacing (dx, dy, dz)
        """
        return self.spacing

    def get_origin(self):
        """
        Lấy tọa độ gốc.

        Returns:
            tuple: Origin (x, y, z)
        """
        return self.origin

    def get_direction(self):
        """
        Lấy ma trận hướng.

        Returns:
            tuple: Direction matrix (9 elements)
        """
        return self.direction
