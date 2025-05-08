#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module dự đoán thay đổi giải phẫu dựa trên biến dạng hình ảnh trong QuangTPS.

Module này cung cấp các chức năng nâng cao để dự đoán sự thay đổi hình ảnh và cấu trúc
dựa trên phân tích biến dạng qua thời gian, hỗ trợ cho việc lập kế hoạch thích ứng chủ động.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Sequence
from enum import Enum, auto
from scipy import interpolate
import SimpleITK as sitk

from quangtps.core.types import Patient, Image, Structure, DoseGrid, Plan
from quangtps.core.exceptions import PredictionError
from quangtps.imaging.registration import ImageRegistration, RegistrationType
from quangtps.adaptive.deformation.deformable_registration import DeformableRegistration
from quangtps.adaptive.deformation.displacement_field import DisplacementField
from quangtps.adaptive.prediction.anatomy_prediction import (
    AnatomyPrediction,
    PredictionMethod,
    AnatomyPredictor,
)
from quangtps.core.utils import get_timestamp, create_directory_if_not_exists

logger = logging.getLogger(__name__)


class DeformationModelType(Enum):
    """Các loại mô hình biến dạng giải phẫu."""

    LINEAR = auto()  # Biến dạng tuyến tính
    ELASTIC = auto()  # Biến dạng đàn hồi
    VISCOUS = auto()  # Biến dạng nhớt
    GROWTH = auto()  # Mô hình tăng trưởng/teo nhỏ mô
    COMBINED = auto()  # Kết hợp các mô hình


class DeformationVectorAnalysis:
    """
    Phân tích trường vector biến dạng để xác định các pattern thay đổi.
    """

    def __init__(self, displacement_field: DisplacementField):
        """
        Khởi tạo phân tích trường vector biến dạng.

        Parameters
        ----------
        displacement_field : DisplacementField
            Trường vector biến dạng cần phân tích
        """
        self.displacement_field = displacement_field
        self.vector_data = displacement_field.get_vector_field()
        self.magnitudes = None
        self.directions = None
        self.divergence = None
        self.curl = None
        self._analyze()

    def _analyze(self):
        """Phân tích các đặc tính của trường vector biến dạng."""
        # Tính độ lớn của các vector
        self.magnitudes = np.sqrt(np.sum(self.vector_data**2, axis=3))

        # Tính hướng của các vector (có thể bổ sung tùy ứng dụng)

        # Tính divergence để phát hiện vùng co giãn
        self._calculate_divergence()

        # Tính curl để phát hiện vùng xoay
        self._calculate_curl()

    def _calculate_divergence(self):
        """Tính divergence của trường vector."""
        # Giả sử vector_data có shape (z, y, x, 3)
        vx = self.vector_data[:, :, :, 0]
        vy = self.vector_data[:, :, :, 1]
        vz = self.vector_data[:, :, :, 2]

        # Tính gradient
        z, y, x = self.vector_data.shape[:3]
        dvx_dx = np.zeros((z, y, x))
        dvy_dy = np.zeros((z, y, x))
        dvz_dz = np.zeros((z, y, x))

        # Tính xấp xỉ gradient bằng sai phân hữu hạn
        dvx_dx[:, :, :-1] = vx[:, :, 1:] - vx[:, :, :-1]
        dvy_dy[:, :-1, :] = vy[:, 1:, :] - vy[:, :-1, :]
        dvz_dz[:-1, :, :] = vz[1:, :, :] - vz[:-1, :, :]

        # Divergence = dvx/dx + dvy/dy + dvz/dz
        self.divergence = dvx_dx + dvy_dy + dvz_dz

    def _calculate_curl(self):
        """Tính curl của trường vector."""
        # Giả sử vector_data có shape (z, y, x, 3)
        vx = self.vector_data[:, :, :, 0]
        vy = self.vector_data[:, :, :, 1]
        vz = self.vector_data[:, :, :, 2]

        # Tính gradient
        z, y, x = self.vector_data.shape[:3]
        dvz_dy = np.zeros((z, y, x))
        dvy_dz = np.zeros((z, y, x))
        dvx_dz = np.zeros((z, y, x))
        dvz_dx = np.zeros((z, y, x))
        dvy_dx = np.zeros((z, y, x))
        dvx_dy = np.zeros((z, y, x))

        # Tính xấp xỉ gradient bằng sai phân hữu hạn
        dvz_dy[:, :-1, :] = vz[:, 1:, :] - vz[:, :-1, :]
        dvy_dz[:-1, :, :] = vy[1:, :, :] - vy[:-1, :, :]

        dvx_dz[:-1, :, :] = vx[1:, :, :] - vx[:-1, :, :]
        dvz_dx[:, :, :-1] = vz[:, :, 1:] - vz[:, :, :-1]

        dvy_dx[:, :, :-1] = vy[:, :, 1:] - vy[:, :, :-1]
        dvx_dy[:, :-1, :] = vx[:, 1:, :] - vx[:, :-1, :]

        # Curl = [dvz/dy - dvy/dz, dvx/dz - dvz/dx, dvy/dx - dvx/dy]
        curl_x = dvz_dy - dvy_dz
        curl_y = dvx_dz - dvz_dx
        curl_z = dvy_dx - dvx_dy

        self.curl = np.stack([curl_x, curl_y, curl_z], axis=3)

    def get_growth_regions(self, threshold: float = 0.5) -> np.ndarray:
        """
        Xác định các vùng mô đang tăng trưởng (divergence > 0).

        Parameters
        ----------
        threshold : float, optional
            Ngưỡng divergence để xác định vùng tăng trưởng, mặc định là 0.5

        Returns
        -------
        np.ndarray
            Mặt nạ nhị phân của các vùng tăng trưởng
        """
        return self.divergence > threshold

    def get_shrinkage_regions(self, threshold: float = 0.5) -> np.ndarray:
        """
        Xác định các vùng mô đang co lại (divergence < 0).

        Parameters
        ----------
        threshold : float, optional
            Ngưỡng divergence (âm) để xác định vùng co lại, mặc định là 0.5

        Returns
        -------
        np.ndarray
            Mặt nạ nhị phân của các vùng co lại
        """
        return self.divergence < -threshold

    def get_deformation_statistics(self) -> Dict[str, Any]:
        """
        Tính toán các thống kê về biến dạng.

        Returns
        -------
        Dict[str, Any]
            Từ điển các thống kê về biến dạng
        """
        stats = {
            "mean_magnitude": np.mean(self.magnitudes),
            "max_magnitude": np.max(self.magnitudes),
            "std_magnitude": np.std(self.magnitudes),
            "mean_divergence": np.mean(self.divergence),
            "positive_divergence_ratio": np.sum(self.divergence > 0)
            / self.divergence.size,
            "negative_divergence_ratio": np.sum(self.divergence < 0)
            / self.divergence.size,
        }
        return stats


class DeformableAnatomyPredictor:
    """
    Dự đoán thay đổi giải phẫu dựa trên biến dạng hình ảnh.
    """

    def __init__(
        self,
        deformable_registration: Optional[DeformableRegistration] = None,
        base_predictor: Optional[AnatomyPredictor] = None,
    ):
        """
        Khởi tạo bộ dự đoán thay đổi giải phẫu dựa trên biến dạng.

        Parameters
        ----------
        deformable_registration : Optional[DeformableRegistration], optional
            Đối tượng đăng ký biến dạng, mặc định là None
        base_predictor : Optional[AnatomyPredictor], optional
            Bộ dự đoán cơ bản, mặc định là None
        """
        self.deformable_registration = (
            deformable_registration or DeformableRegistration()
        )
        self.base_predictor = base_predictor or AnatomyPredictor()
        self.historical_displacement_fields = []
        self.historical_dates = []
        self.deformation_models = {}

    def add_historical_data(
        self,
        reference_image: Image,
        target_image: Image,
        reference_date: datetime.datetime,
        target_date: datetime.datetime,
    ):
        """
        Thêm dữ liệu lịch sử để dùng cho phân tích biến dạng.

        Parameters
        ----------
        reference_image : Image
            Hình ảnh tham chiếu
        target_image : Image
            Hình ảnh đích
        reference_date : datetime.datetime
            Ngày của hình ảnh tham chiếu
        target_date : datetime.datetime
            Ngày của hình ảnh đích
        """
        # Thực hiện đăng ký biến dạng
        try:
            displacement_field = self.deformable_registration.register(
                reference_image, target_image
            )

            # Lưu trữ trường biến dạng và ngày tương ứng
            self.historical_displacement_fields.append(displacement_field)
            self.historical_dates.append((reference_date, target_date))

            # Phân tích trường biến dạng
            analysis = DeformationVectorAnalysis(displacement_field)

            # Tính toán thời gian giữa hai ảnh
            days_between = (target_date - reference_date).days

            # Lưu mô hình biến dạng
            self.deformation_models[target_date] = {
                "displacement_field": displacement_field,
                "analysis": analysis,
                "reference_date": reference_date,
                "days_between": days_between,
                "statistics": analysis.get_deformation_statistics(),
            }

        except Exception as e:
            logger.error(f"Không thể thêm dữ liệu lịch sử: {str(e)}")
            raise PredictionError(f"Lỗi khi thêm dữ liệu lịch sử: {str(e)}")

    def predict_future_anatomy(
        self,
        patient: Patient,
        reference_image: Image,
        reference_structures: Dict[str, Structure],
        reference_date: datetime.datetime,
        prediction_date: datetime.datetime,
        model_type: DeformationModelType = DeformationModelType.COMBINED,
        target_structures: Optional[List[str]] = None,
    ) -> AnatomyPrediction:
        """
        Dự đoán giải phẫu tương lai dựa trên mô hình biến dạng.

        Parameters
        ----------
        patient : Patient
            Thông tin bệnh nhân
        reference_image : Image
            Hình ảnh tham chiếu hiện tại
        reference_structures : Dict[str, Structure]
            Các cấu trúc tham chiếu hiện tại
        reference_date : datetime.datetime
            Ngày của hình ảnh tham chiếu
        prediction_date : datetime.datetime
            Ngày cần dự đoán
        model_type : DeformationModelType, optional
            Loại mô hình biến dạng, mặc định là DeformationModelType.COMBINED
        target_structures : Optional[List[str]], optional
            Danh sách tên cấu trúc cần dự đoán, mặc định là None (tất cả)

        Returns
        -------
        AnatomyPrediction
            Kết quả dự đoán giải phẫu
        """
        if not self.historical_displacement_fields:
            logger.warning("Không có dữ liệu lịch sử để dự đoán giải phẫu tương lai")
            # Sử dụng bộ dự đoán cơ bản nếu không có dữ liệu biến dạng
            return self.base_predictor.predict_anatomy_changes(
                patient=patient,
                historical_images=[reference_image],
                historical_structures=[reference_structures],
                historical_dates=[reference_date],
                prediction_dates=[prediction_date],
                method=PredictionMethod.LINEAR,
                target_structures=target_structures,
            )

        # Tạo đối tượng dự đoán
        prediction = AnatomyPrediction(reference_date, patient.patient_id)

        # Tính toán trường biến dạng dự đoán
        predicted_displacement_field = self._predict_displacement_field(
            reference_date, prediction_date, model_type
        )

        # Áp dụng trường biến dạng dự đoán vào hình ảnh tham chiếu
        predicted_image = self._apply_displacement_field_to_image(
            reference_image, predicted_displacement_field
        )

        # Lọc cấu trúc cần dự đoán
        structures_to_predict = reference_structures
        if target_structures:
            structures_to_predict = {
                name: struct
                for name, struct in reference_structures.items()
                if name in target_structures
            }

        # Áp dụng trường biến dạng dự đoán vào các cấu trúc
        predicted_structures = self._apply_displacement_field_to_structures(
            structures_to_predict, predicted_displacement_field
        )

        # Tính toán độ tin cậy dựa trên độ chênh lệch thời gian
        # Độ tin cậy giảm khi khoảng thời gian dự đoán tăng
        days_to_predict = (prediction_date - reference_date).days
        max_confident_days = 30  # Giả sử dự đoán tin cậy trong vòng 30 ngày
        confidence = max(
            0.1, min(1.0, 1.0 - (days_to_predict / (max_confident_days * 2)))
        )

        # Thêm dữ liệu dự đoán
        prediction.add_prediction_timepoint(
            prediction_date, predicted_structures, predicted_image, confidence
        )

        return prediction

    def _predict_displacement_field(
        self,
        reference_date: datetime.datetime,
        prediction_date: datetime.datetime,
        model_type: DeformationModelType,
    ) -> DisplacementField:
        """
        Dự đoán trường biến dạng tại một thời điểm trong tương lai.

        Parameters
        ----------
        reference_date : datetime.datetime
            Ngày tham chiếu
        prediction_date : datetime.datetime
            Ngày cần dự đoán
        model_type : DeformationModelType
            Loại mô hình biến dạng

        Returns
        -------
        DisplacementField
            Trường biến dạng dự đoán
        """
        # Tìm mô hình biến dạng gần nhất với reference_date
        closest_model = None
        min_diff = float("inf")

        for date, model in self.deformation_models.items():
            if model["reference_date"] == reference_date:
                diff = abs((date - prediction_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_model = model

        if not closest_model:
            # Nếu không tìm thấy mô hình phù hợp, sử dụng mô hình gần nhất với reference_date
            for date, model in self.deformation_models.items():
                diff = abs((model["reference_date"] - reference_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_model = model

        if not closest_model:
            raise PredictionError("Không thể tìm thấy mô hình biến dạng phù hợp")

        # Lấy trường biến dạng cơ sở
        base_field = closest_model["displacement_field"]

        # Tính toán hệ số nhân dựa trên thời gian
        target_days = (prediction_date - reference_date).days
        base_days = closest_model["days_between"]

        # Hệ số nhân khác nhau tùy theo loại mô hình
        if model_type == DeformationModelType.LINEAR:
            # Mô hình tuyến tính
            scale_factor = target_days / base_days if base_days != 0 else 1.0

        elif model_type == DeformationModelType.ELASTIC:
            # Mô hình đàn hồi: biến dạng ban đầu nhanh, sau đó chậm lại
            scale_factor = (
                1.0 - np.exp(-target_days / (base_days * 1.5))
                if base_days != 0
                else 1.0
            )

        elif model_type == DeformationModelType.VISCOUS:
            # Mô hình nhớt: biến dạng ban đầu chậm, sau đó nhanh hơn
            scale_factor = np.tanh(target_days / base_days) if base_days != 0 else 1.0

        elif model_type == DeformationModelType.GROWTH:
            # Mô hình tăng trưởng: phi tuyến, có thể tăng hoặc giảm
            growth_rate = 0.03  # Tỷ lệ tăng trưởng ngày
            scale_factor = np.exp(growth_rate * target_days) - 1

        else:  # DeformationModelType.COMBINED
            # Kết hợp các mô hình
            linear_factor = target_days / base_days if base_days != 0 else 1.0
            elastic_factor = (
                1.0 - np.exp(-target_days / (base_days * 1.5))
                if base_days != 0
                else 1.0
            )
            scale_factor = (linear_factor + elastic_factor) / 2.0

        # Nhân trường vector biến dạng với hệ số
        scaled_field = base_field.scale(scale_factor)

        return scaled_field

    def _apply_displacement_field_to_image(
        self, image: Image, displacement_field: DisplacementField
    ) -> Image:
        """
        Áp dụng trường biến dạng vào hình ảnh.

        Parameters
        ----------
        image : Image
            Hình ảnh đầu vào
        displacement_field : DisplacementField
            Trường biến dạng cần áp dụng

        Returns
        -------
        Image
            Hình ảnh đã biến dạng
        """
        # Trong ứng dụng thực tế, đây sẽ là việc áp dụng trường biến dạng vào hình ảnh
        # Ở đây tạo một bản sao của hình ảnh gốc
        deformed_image = Image(id=f"{image.id}_deformed", modality=image.modality)

        try:
            # Chuyển đổi hình ảnh sang định dạng SimpleITK
            sitk_image = sitk.GetImageFromArray(image.data)
            sitk_image.SetSpacing(
                (image.pixel_spacing[0], image.pixel_spacing[1], image.slice_thickness)
            )

            # Chuyển đổi trường biến dạng sang định dạng SimpleITK
            vector_field = displacement_field.get_vector_field()
            sitk_vector_field = sitk.GetImageFromArray(vector_field, isVector=True)
            sitk_vector_field.SetSpacing(
                (image.pixel_spacing[0], image.pixel_spacing[1], image.slice_thickness)
            )

            # Áp dụng biến dạng
            displacement_filter = sitk.DisplacementFieldTransform(sitk_vector_field)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitk_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(displacement_filter)

            deformed_sitk_image = resampler.Execute(sitk_image)

            # Chuyển đổi trở lại
            deformed_image.data = sitk.GetArrayFromImage(deformed_sitk_image)
            deformed_image.pixel_spacing = image.pixel_spacing
            deformed_image.slice_thickness = image.slice_thickness

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng trường biến dạng vào hình ảnh: {str(e)}")
            # Trường hợp lỗi, trả về bản sao hình ảnh gốc
            deformed_image.data = image.data.copy() if image.data is not None else None
            deformed_image.pixel_spacing = image.pixel_spacing
            deformed_image.slice_thickness = image.slice_thickness

        return deformed_image

    def _apply_displacement_field_to_structures(
        self, structures: Dict[str, Structure], displacement_field: DisplacementField
    ) -> Dict[str, Structure]:
        """
        Áp dụng trường biến dạng vào các cấu trúc.

        Parameters
        ----------
        structures : Dict[str, Structure]
            Từ điển các cấu trúc đầu vào
        displacement_field : DisplacementField
            Trường biến dạng cần áp dụng

        Returns
        -------
        Dict[str, Structure]
            Từ điển các cấu trúc đã biến dạng
        """
        deformed_structures = {}

        for name, structure in structures.items():
            try:
                # Tạo cấu trúc mới
                deformed_struct = Structure(
                    id=f"{structure.id}_deformed",
                    name=structure.name,
                    type=structure.type,
                    color=structure.color,
                )

                # Lấy mặt nạ nhị phân
                mask = structure.get_binary_mask()

                # Chuyển đổi mặt nạ sang định dạng SimpleITK
                sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
                voxel_spacing = structure.get_voxel_spacing()
                sitk_mask.SetSpacing(voxel_spacing)

                # Chuyển đổi trường biến dạng sang định dạng SimpleITK
                vector_field = displacement_field.get_vector_field()
                sitk_vector_field = sitk.GetImageFromArray(vector_field, isVector=True)
                sitk_vector_field.SetSpacing(voxel_spacing)

                # Áp dụng biến dạng
                displacement_filter = sitk.DisplacementFieldTransform(sitk_vector_field)
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(sitk_mask)
                resampler.SetInterpolator(
                    sitk.sitkNearestNeighbor
                )  # Sử dụng nearest neighbor cho mặt nạ nhị phân
                resampler.SetTransform(displacement_filter)

                deformed_sitk_mask = resampler.Execute(sitk_mask)

                # Chuyển đổi trở lại
                deformed_mask = sitk.GetArrayFromImage(deformed_sitk_mask).astype(bool)

                # Thiết lập mặt nạ mới
                deformed_struct.set_binary_mask(deformed_mask)
                deformed_struct.set_voxel_spacing(voxel_spacing)

                # Tính toán thể tích
                deformed_struct.calculate_volume()

                deformed_structures[name] = deformed_struct

            except Exception as e:
                logger.error(
                    f"Lỗi khi áp dụng trường biến dạng vào cấu trúc {name}: {str(e)}"
                )
                # Trường hợp lỗi, sử dụng cấu trúc gốc
                deformed_structures[name] = structure

        return deformed_structures

    def visualize_deformation_field(
        self,
        displacement_field: DisplacementField,
        slice_idx: int = None,
        save_path: Optional[str] = None,
    ):
        """
        Trực quan hóa trường biến dạng.

        Parameters
        ----------
        displacement_field : DisplacementField
            Trường biến dạng cần trực quan hóa
        slice_idx : int, optional
            Chỉ số lát cắt cần hiển thị, mặc định là None (lát cắt giữa)
        save_path : Optional[str], optional
            Đường dẫn để lưu hình ảnh, mặc định là None
        """
        vector_field = displacement_field.get_vector_field()

        if slice_idx is None:
            slice_idx = vector_field.shape[0] // 2

        # Lấy lát cắt
        slice_data = vector_field[slice_idx, :, :, :]

        # Tính độ lớn vector
        magnitudes = np.sqrt(np.sum(slice_data**2, axis=2))

        plt.figure(figsize=(15, 10))

        # Vẽ trường biến dạng
        plt.subplot(2, 2, 1)
        plt.imshow(magnitudes, cmap="jet")
        plt.colorbar(label="Magnitude (mm)")
        plt.title(f"Magnitude of Deformation Field (Slice {slice_idx})")

        # Vẽ vector field (subsample để dễ nhìn)
        plt.subplot(2, 2, 2)
        step = 8  # Subsampling step
        Y, X = np.mgrid[: slice_data.shape[0] : step, : slice_data.shape[1] : step]
        U = slice_data[::step, ::step, 0]
        V = slice_data[::step, ::step, 1]

        plt.quiver(X, Y, U, V, magnitudes[::step, ::step], cmap="jet")
        plt.colorbar(label="Magnitude (mm)")
        plt.title("Vector Direction")

        # Vẽ thành phần X
        plt.subplot(2, 2, 3)
        plt.imshow(slice_data[:, :, 0], cmap="RdBu")
        plt.colorbar(label="X Displacement (mm)")
        plt.title("X Component")

        # Vẽ thành phần Y
        plt.subplot(2, 2, 4)
        plt.imshow(slice_data[:, :, 1], cmap="RdBu")
        plt.colorbar(label="Y Displacement (mm)")
        plt.title("Y Component")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def create_deformable_anatomy_predictor() -> DeformableAnatomyPredictor:
    """
    Tạo và cấu hình bộ dự đoán giải phẫu biến dạng.

    Returns
    -------
    DeformableAnatomyPredictor
        Bộ dự đoán đã cấu hình
    """
    try:
        # Tạo đối tượng đăng ký biến dạng
        deformable_reg = DeformableRegistration()

        # Tạo bộ dự đoán cơ bản
        base_predictor = AnatomyPredictor()

        # Tạo bộ dự đoán biến dạng
        predictor = DeformableAnatomyPredictor(
            deformable_registration=deformable_reg, base_predictor=base_predictor
        )

        return predictor

    except Exception as e:
        logger.error(f"Lỗi khi tạo bộ dự đoán giải phẫu biến dạng: {str(e)}")
        raise
