#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp các lớp dự đoán thay đổi giải phẫu dựa trên mô hình biến dạng.

Module này cung cấp các công cụ để dự đoán thay đổi thể tích, hình dạng và vị trí
của các cấu trúc giải phẫu theo thời gian, phục vụ cho lập kế hoạch thích ứng.
"""

import os
import enum
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

import SimpleITK as sitk
import joblib
from sklearn.linear_model import ElasticNet

from quangtps.core.types import Image, Structure
from quangtps.core.patient import Patient
from quangtps.core.exceptions import PredictionError
from quangtps.adaptive.deformation.deformation_map import DeformationMap

logger = logging.getLogger(__name__)


class DeformationModelType(enum.Enum):
    """Enum định nghĩa các loại mô hình biến dạng."""

    LINEAR = "linear"
    BSPLINE = "bspline"
    DIFFEOMORPHIC = "diffeomorphic"
    BIOMECHANICAL = "biomechanical"
    DEEP_LEARNING = "deep_learning"
    CUSTOM = "custom"


class DeformableAnatomyPredictor:
    """
    Lớp dự đoán thay đổi giải phẫu dựa trên mô hình biến dạng.

    Lớp này cung cấp các phương thức để dự đoán thay đổi thể tích, hình dạng và vị trí
    của các cấu trúc giải phẫu dựa trên mô hình học máy và biến dạng.
    """

    def __init__(
        self,
        patient: Patient,
        reference_image: Image,
        model_type: DeformationModelType = DeformationModelType.BSPLINE,
    ):
        """
        Khởi tạo bộ dự đoán thay đổi giải phẫu.

        Args:
            patient: Đối tượng bệnh nhân.
            reference_image: Ảnh tham chiếu.
            model_type: Loại mô hình biến dạng.
        """
        self.patient = patient
        self.reference_image = reference_image
        self.model_type = model_type

        # Từ điển mô hình dự đoán theo cấu trúc
        self.prediction_models: Dict[str, Any] = {}

        # Tham số mô hình
        self.model_params: Dict[str, Any] = {
            "grid_spacing": 20.0,  # mm
            "smoothing_sigma": 3.0,  # mm
            "optimization_steps": 100,
            "learning_rate": 0.1,
        }

        logger.info(
            f"Khởi tạo bộ dự đoán thay đổi giải phẫu với mô hình {model_type.name}"
        )

    def load_model(self, structure_id: str, model_path: str) -> bool:
        """
        Tải mô hình dự đoán cho một cấu trúc cụ thể.

        Args:
            structure_id: ID của cấu trúc.
            model_path: Đường dẫn đến tệp mô hình.

        Returns:
            bool: True nếu tải thành công, False nếu thất bại.
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Không tìm thấy tệp mô hình: {model_path}")
                return False

            # Tải mô hình
            model = joblib.load(model_path)

            # Kiểm tra loại mô hình
            if not isinstance(model, (ElasticNet, dict)):
                logger.error(f"Loại mô hình không được hỗ trợ: {type(model)}")
                return False

            # Lưu mô hình
            self.prediction_models[structure_id] = model

            logger.info(
                f"Đã tải mô hình dự đoán cho cấu trúc {structure_id} từ {model_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình dự đoán: {str(e)}")
            return False

    def predict_volume_changes(self, structure_id: str, days: int) -> List[float]:
        """
        Dự đoán thay đổi thể tích của một cấu trúc theo thời gian.

        Args:
            structure_id: ID của cấu trúc.
            days: Số ngày cần dự đoán trong tương lai.

        Returns:
            List[float]: Danh sách thể tích dự đoán, mỗi phần tử tương ứng với một ngày.
        """
        logger.info(
            f"Dự đoán thay đổi thể tích cho cấu trúc {structure_id} trong {days} ngày"
        )

        # Tìm cấu trúc trong danh sách
        structure = None
        for s in self.patient.structures:
            if s.id == structure_id:
                structure = s
                break

        if not structure:
            logger.warning(f"Không tìm thấy cấu trúc {structure_id}")
            return [0.0] * days

        # Kiểm tra mô hình dự đoán
        model = self.prediction_models.get(structure_id)

        if model is not None:
            try:
                # Dự đoán sử dụng mô hình ML
                return self._predict_volumes_with_ml(structure, model, days)
            except Exception as e:
                logger.error(f"Lỗi khi dự đoán với mô hình ML: {str(e)}")

        # Sử dụng phương pháp dự đoán mặc định nếu không có mô hình
        return self._predict_volumes_default(structure, days)

    def _predict_volumes_with_ml(
        self, structure: Structure, model: Any, days: int
    ) -> List[float]:
        """
        Dự đoán thể tích sử dụng mô hình học máy.

        Args:
            structure: Cấu trúc cần dự đoán.
            model: Mô hình học máy.
            days: Số ngày cần dự đoán.

        Returns:
            List[float]: Danh sách thể tích dự đoán.
        """
        # Lấy thể tích hiện tại
        current_volume = structure.volume

        if isinstance(model, ElasticNet):
            # Tạo đặc trưng đầu vào là số ngày
            X = np.array([i + 1 for i in range(days)]).reshape(-1, 1)

            # Dự đoán tỷ lệ thay đổi
            change_ratios = model.predict(X).flatten()

            # Tính thể tích mới
            volumes = [current_volume * (1.0 + ratio) for ratio in change_ratios]

            return volumes

        elif isinstance(model, dict) and "type" in model:
            # Mô hình tùy chỉnh dưới dạng từ điển
            if model["type"] == "exponential_decay":
                decay_rate = model.get("decay_rate", 0.01)
                volumes = [
                    current_volume * np.exp(-decay_rate * i) for i in range(1, days + 1)
                ]
                return volumes

            elif model["type"] == "linear_trend":
                slope = model.get("slope", -0.5)  # cc/ngày
                volumes = [
                    max(0.1, current_volume + slope * i) for i in range(1, days + 1)
                ]
                return volumes

            else:
                logger.warning(f"Không hỗ trợ loại mô hình: {model['type']}")

        # Mô hình không được hỗ trợ, sử dụng mặc định
        return self._predict_volumes_default(structure, days)

    def _predict_volumes_default(self, structure: Structure, days: int) -> List[float]:
        """
        Phương pháp dự đoán mặc định khi không có mô hình học máy.

        Args:
            structure: Cấu trúc cần dự đoán.
            days: Số ngày cần dự đoán.

        Returns:
            List[float]: Danh sách thể tích dự đoán.
        """
        # Thể tích hiện tại
        current_volume = structure.volume

        # Tỷ lệ thay đổi mặc định dựa trên loại cấu trúc
        change_ratio = 0.0

        if structure.type.lower() in ("ptv", "ctv", "gtv", "target"):
            # Mục tiêu có xu hướng giảm
            change_ratio = -0.01  # Giảm 1%/ngày
        elif "parotid" in structure.name.lower():
            # Tuyến mang tai có xu hướng giảm mạnh
            change_ratio = -0.015  # Giảm 1.5%/ngày
        elif any(
            org in structure.name.lower()
            for org in ["heart", "lung", "kidney", "liver"]
        ):
            # Các cơ quan này thường ổn định hơn
            change_ratio = -0.005  # Giảm 0.5%/ngày

        # Áp dụng thay đổi
        volumes = []
        for i in range(1, days + 1):
            # Công thức thay đổi theo hàm mũ
            new_volume = current_volume * (1.0 + change_ratio) ** i
            volumes.append(max(0.1, new_volume))  # Đảm bảo thể tích không âm

        return volumes

    def predict_deformation_map(self, days: int = 1) -> DeformationMap:
        """
        Dự đoán ánh xạ biến dạng cho ngày cụ thể trong tương lai.

        Args:
            days: Số ngày cần dự đoán trong tương lai.

        Returns:
            DeformationMap: Ánh xạ biến dạng dự đoán.
        """
        logger.info(f"Dự đoán ánh xạ biến dạng cho {days} ngày trong tương lai")

        if days < 1:
            logger.warning(f"Số ngày không hợp lệ: {days}, sử dụng mặc định là 1")
            days = 1

        # Tạo ánh xạ biến dạng dựa trên loại mô hình
        if self.model_type == DeformationModelType.BSPLINE:
            return self._create_bspline_deformation_map(days)

        elif self.model_type == DeformationModelType.DIFFEOMORPHIC:
            return self._create_diffeomorphic_deformation_map(days)

        elif self.model_type == DeformationModelType.BIOMECHANICAL:
            return self._create_biomechanical_deformation_map(days)

        elif self.model_type == DeformationModelType.DEEP_LEARNING:
            return self._create_deep_learning_deformation_map(days)

        else:
            # Mặc định sử dụng B-spline
            return self._create_bspline_deformation_map(days)

    def _create_bspline_deformation_map(self, days: int) -> DeformationMap:
        """
        Tạo ánh xạ biến dạng dự đoán sử dụng biến dạng B-spline.

        Args:
            days: Số ngày cần dự đoán trong tương lai.

        Returns:
            DeformationMap: Ánh xạ biến dạng dự đoán.
        """
        logger.info("Tạo ánh xạ biến dạng B-spline")

        # Lấy tham số từ cấu hình
        grid_spacing = self.model_params.get("grid_spacing", 20.0)
        sigma = self.model_params.get("smoothing_sigma", 3.0)

        # Tạo đối tượng sitk cho biến dạng B-spline
        image_sitk = (
            self.reference_image.to_sitk()
            if hasattr(self.reference_image, "to_sitk")
            else self.reference_image
        )

        # Tạo trường dịch chuyển
        displacement_field = sitk.Image(image_sitk.GetSize(), sitk.sitkVectorFloat64)
        displacement_field.SetSpacing(image_sitk.GetSpacing())
        displacement_field.SetOrigin(image_sitk.GetOrigin())
        displacement_field.SetDirection(image_sitk.GetDirection())

        # Tỷ lệ biến dạng dựa trên số ngày
        deformation_scale = min(1.0, days * 0.2)  # Giới hạn tỷ lệ tối đa

        # Tạo biến dạng giả định dựa trên các thay đổi sinh lý thường gặp
        # Ví dụ: thu nhỏ các mô mềm, giảm thể tích nước

        # Tạo lưới điểm điều khiển B-spline
        transform = sitk.BSplineTransformInitializer(
            image_sitk, [int(image_sitk.GetSize()[i] / grid_spacing) for i in range(3)]
        )

        # Tạo các hệ số biến dạng
        params = transform.GetParameters()

        # Thay đổi các tham số biến dạng dựa trên các mô hình thay đổi
        # Đây là mô phỏng đơn giản về thay đổi giải phẫu theo thời gian
        np.random.seed(42)  # Đảm bảo tính lặp lại

        # Cấu trúc của tham số biến dạng:
        # mỗi 3 tham số liên tiếp là Vector3D (dx, dy, dz) cho một điểm lưới
        num_params = len(params)

        # Đối với mỗi thông số biến dạng (nhóm theo 3)
        for i in range(0, num_params, 3):
            # Tạo biến dạng có xu hướng co lại về phía trung tâm
            coords = [
                (i // 3)
                % transform.GetTransform().GetCoefficientImages()[0].GetSize()[j]
                for j in range(3)
            ]

            # Khoảng cách từ trung tâm
            center = [
                transform.GetTransform().GetCoefficientImages()[0].GetSize()[j] / 2
                for j in range(3)
            ]
            dist_from_center = (
                sum([(coords[j] - center[j]) ** 2 for j in range(3)]) ** 0.5
            )

            # Biến dạng phụ thuộc vào khoảng cách từ trung tâm
            direction = [coords[j] - center[j] for j in range(3)]
            magnitude = (
                deformation_scale
                * (dist_from_center / 10.0)
                * np.exp(-dist_from_center / 20.0)
            )

            # Áp dụng biến dạng
            if i + 2 < num_params:
                params[i] = -direction[0] * magnitude
                params[i + 1] = -direction[1] * magnitude
                params[i + 2] = -direction[2] * magnitude

        # Cập nhật tham số biến dạng
        transform.SetParameters(params)

        # Áp dụng biến dạng vào trường dịch chuyển
        displacement = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            displacement_field.GetSize(),
            displacement_field.GetOrigin(),
            displacement_field.GetSpacing(),
            displacement_field.GetDirection(),
        )

        # Làm trơn trường dịch chuyển
        displacement_smooth = sitk.SmoothingRecursiveGaussian(displacement, sigma)

        # Tạo ánh xạ biến dạng từ trường dịch chuyển
        return DeformationMap(
            reference_image=self.reference_image,
            displacement_field=displacement_smooth,
            description=f"Dự đoán biến dạng B-spline sau {days} ngày",
        )

    def _create_diffeomorphic_deformation_map(self, days: int) -> DeformationMap:
        """
        Tạo ánh xạ biến dạng dự đoán sử dụng biến dạng diffeomorphic.

        Args:
            days: Số ngày cần dự đoán trong tương lai.

        Returns:
            DeformationMap: Ánh xạ biến dạng dự đoán.
        """
        logger.info("Tạo ánh xạ biến dạng diffeomorphic")

        # Trong trường hợp thực tế, sẽ triển khai thuật toán biến dạng diffeomorphic
        # như Demons hoặc SyN

        # Hiện tại, sử dụng B-spline làm dự phòng
        return self._create_bspline_deformation_map(days)

    def _create_biomechanical_deformation_map(self, days: int) -> DeformationMap:
        """
        Tạo ánh xạ biến dạng dự đoán sử dụng mô hình cơ học sinh học.

        Args:
            days: Số ngày cần dự đoán trong tương lai.

        Returns:
            DeformationMap: Ánh xạ biến dạng dự đoán.
        """
        logger.info("Tạo ánh xạ biến dạng cơ học sinh học")

        # Mô hình cơ học sinh học cần tích hợp với các công cụ mô phỏng FEM
        # như FEniCS, ANSYS, hoặc SOFA

        # Hiện tại, sử dụng B-spline làm dự phòng
        return self._create_bspline_deformation_map(days)

    def _create_deep_learning_deformation_map(self, days: int) -> DeformationMap:
        """
        Tạo ánh xạ biến dạng dự đoán sử dụng mô hình học sâu.

        Args:
            days: Số ngày cần dự đoán trong tương lai.

        Returns:
            DeformationMap: Ánh xạ biến dạng dự đoán.
        """
        logger.info("Tạo ánh xạ biến dạng học sâu")

        # Trong trường hợp thực tế, sẽ tải mô hình học sâu đã huấn luyện trước
        # như U-Net hoặc VoxelMorph

        # Hiện tại, sử dụng B-spline làm dự phòng
        return self._create_bspline_deformation_map(days)

    def get_prediction_uncertainty(self, structure_id: str, days: int) -> List[float]:
        """
        Ước tính độ không chắc chắn của dự đoán thay đổi thể tích.

        Args:
            structure_id: ID của cấu trúc.
            days: Số ngày cần dự đoán trong tương lai.

        Returns:
            List[float]: Danh sách độ không chắc chắn (độ lệch chuẩn) cho mỗi ngày.
        """
        # Độ không chắc chắn tăng theo số ngày dự đoán
        base_uncertainty = 0.02  # 2% cho ngày đầu tiên

        # Tính toán độ không chắc chắn cho mỗi ngày
        uncertainties = [base_uncertainty * (1 + 0.2 * i) for i in range(days)]

        return uncertainties

    def export_prediction_report(
        self, predictions: Dict[str, Dict[str, Any]], output_path: str
    ) -> bool:
        """
        Xuất báo cáo dự đoán thay đổi giải phẫu.

        Args:
            predictions: Kết quả dự đoán từ phương thức predict_volume_changes.
            output_path: Đường dẫn tệp đầu ra.

        Returns:
            bool: True nếu xuất thành công, False nếu thất bại.
        """
        try:
            with open(output_path, "w") as f:
                f.write("BÁOCÁO DỰ ĐOÁN THAY ĐỔI GIẢI PHẪU\n")
                f.write("=" * 40 + "\n\n")

                f.write(f"ID bệnh nhân: {self.patient.patient_id}\n")
                f.write(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for structure_id, data in predictions.items():
                    f.write(f"Cấu trúc: {data.get('structure_name', structure_id)}\n")
                    f.write(
                        f"Thể tích hiện tại: {data.get('current_volume', 0):.2f} cc\n"
                    )

                    f.write("Dự đoán thay đổi:\n")
                    for pred in data.get("predictions", []):
                        date = pred.get("date", "")
                        volume = pred.get("volume", 0)

                        date_str = (
                            date.strftime("%Y-%m-%d")
                            if isinstance(date, datetime)
                            else str(date)
                        )
                        f.write(f"  - {date_str}: {volume:.2f} cc\n")

                    f.write("\n")

            logger.info(f"Đã xuất báo cáo dự đoán thành công: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xuất báo cáo dự đoán: {str(e)}")
            return False
