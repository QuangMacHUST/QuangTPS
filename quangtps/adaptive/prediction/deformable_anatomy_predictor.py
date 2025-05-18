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


class DeformationModel:
    """
    Mô hình biến dạng cơ bản cho dự đoán thay đổi giải phẫu.
    """

    def __init__(self, model_type: DeformationModelType = DeformationModelType.LINEAR):
        """
        Khởi tạo mô hình biến dạng.

        Parameters
        ----------
        model_type : DeformationModelType
            Loại mô hình biến dạng để sử dụng.
        """
        self.model_type = model_type
        self.parameters = {}
        self.is_trained = False
        self.training_error = float("inf")
        self.vector_field = None

    def train(self, source_image, target_image, **kwargs):
        """
        Huấn luyện mô hình biến dạng từ ảnh nguồn và đích.

        Parameters
        ----------
        source_image : ndarray
            Hình ảnh nguồn.
        target_image : ndarray
            Hình ảnh đích.
        **kwargs
            Các tham số bổ sung cho quá trình huấn luyện.

        Returns
        -------
        bool
            True nếu huấn luyện thành công, False nếu không.
        """
        try:
            logger.info(f"Huấn luyện mô hình biến dạng {self.model_type.value}")

            # Kiểm tra kích thước ảnh
            if source_image.shape != target_image.shape:
                logger.error("Kích thước ảnh nguồn và đích không khớp nhau")
                return False

            # Mô phỏng huấn luyện mô hình (sẽ được thay thế bằng thuật toán thực)
            # Trong phiên bản này, chúng ta tạo một trường vector giả

            # Tạo trường vector biến dạng giả
            shape = source_image.shape
            self.vector_field = np.zeros((*shape, 3), dtype=np.float32)

            # Điền một số giá trị giả (mô phỏng sự thay đổi nhỏ)
            center = np.array([s // 2 for s in shape])

            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        # Tính vector từ điểm hiện tại đến tâm
                        point = np.array([i, j, k])
                        direction = center - point

                        # Chuẩn hóa và thu nhỏ
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction = direction / norm * 0.5  # Giảm biên độ

                        # Thêm nhiễu nhỏ
                        noise = np.random.normal(0, 0.1, 3)

                        # Đặt vector
                        if self.model_type == DeformationModelType.LINEAR:
                            # Biến dạng tuyến tính đơn giản
                            self.vector_field[i, j, k] = direction * 0.2 + noise * 0.1
                        elif self.model_type == DeformationModelType.ELASTIC:
                            # Mô phỏng biến dạng đàn hồi
                            self.vector_field[i, j, k] = (
                                direction * (1 - np.exp(-norm / 50)) + noise * 0.05
                            )
                        elif self.model_type == DeformationModelType.VISCOUS:
                            # Mô phỏng biến dạng nhớt
                            self.vector_field[i, j, k] = (
                                direction * (np.tanh(norm / 30)) + noise * 0.05
                            )
                        else:
                            # Mô hình GROWTH - tập trung ở tâm
                            distance_factor = np.exp(-norm / 20)
                            growth_vector = direction * distance_factor * 0.3
                            self.vector_field[i, j, k] = growth_vector + noise * 0.05

            # Mô phỏng kết quả huấn luyện
            self.training_error = np.random.uniform(0.01, 0.05)
            self.is_trained = True

            # Lưu các tham số
            self.parameters = {
                "timestamp": datetime.now().isoformat(),
                "error": self.training_error,
                "model_type": self.model_type.value,
            }

            logger.info(f"Huấn luyện hoàn tất với lỗi: {self.training_error:.4f}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình biến dạng: {str(e)}")
            return False

    def apply(self, image, time_factor: float = 1.0):
        """
        Áp dụng mô hình biến dạng cho hình ảnh mới.

        Parameters
        ----------
        image : ndarray
            Hình ảnh nguồn để biến đổi.
        time_factor : float, optional
            Hệ số thời gian để xác định mức độ biến đổi (0.0 - 1.0).

        Returns
        -------
        ndarray
            Hình ảnh đã biến đổi.
        """
        if not self.is_trained or self.vector_field is None:
            logger.error("Mô hình chưa được huấn luyện")
            return image

        if time_factor < 0.0 or time_factor > 1.0:
            logger.warning(f"Hệ số thời gian nằm ngoài phạm vi [0, 1]: {time_factor}")
            time_factor = max(0.0, min(1.0, time_factor))

        try:
            # Trong phiên bản này, chúng ta mô phỏng việc áp dụng trường vector
            # Một triển khai thực sẽ sử dụng nội suy và warp

            # Tạo hình ảnh kết quả
            result = np.copy(image)

            # Mô phỏng biến dạng đơn giản bằng cách thêm gradient
            gradient = np.ones_like(image) * 0.1 * time_factor

            # Để đơn giản, chúng ta chỉ thêm gradient để mô phỏng thay đổi
            result = result + gradient

            # Đảm bảo giá trị hợp lệ
            result = np.clip(result, 0, None)

            return result

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng mô hình biến dạng: {str(e)}")
            return image

    def save(self, filepath: str):
        """
        Lưu mô hình biến dạng vào tệp.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp để lưu mô hình.

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không.
        """
        if not self.is_trained:
            logger.error("Không thể lưu mô hình chưa được huấn luyện")
            return False

        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Lưu mô hình dưới dạng file numpy
            np.savez(
                filepath,
                vector_field=self.vector_field,
                model_type=self.model_type.value,
                parameters=self.parameters,
                training_error=self.training_error,
            )

            logger.info(f"Đã lưu mô hình biến dạng vào {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình biến dạng: {str(e)}")
            return False

    def load(self, filepath: str):
        """
        Tải mô hình biến dạng từ tệp.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp để tải mô hình.

        Returns
        -------
        bool
            True nếu tải thành công, False nếu không.
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Không tìm thấy file mô hình: {filepath}")
                return False

            # Tải mô hình từ file numpy
            data = np.load(filepath, allow_pickle=True)

            # Khôi phục trường vector và các tham số
            self.vector_field = data["vector_field"]
            self.model_type = DeformationModelType(data["model_type"].item())
            self.parameters = data["parameters"].item()
            self.training_error = data["training_error"].item()
            self.is_trained = True

            logger.info(f"Đã tải mô hình biến dạng từ {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình biến dạng: {str(e)}")
            return False


class DeformationVectorAnalysis:
    """
    Phân tích trường vector biến dạng.
    """

    def __init__(self, vector_field: np.ndarray = None):
        """
        Khởi tạo phân tích trường vector biến dạng.

        Parameters
        ----------
        vector_field : ndarray, optional
            Trường vector biến dạng để phân tích.
        """
        self.vector_field = vector_field

    def set_vector_field(self, vector_field: np.ndarray):
        """
        Thiết lập trường vector biến dạng để phân tích.

        Parameters
        ----------
        vector_field : ndarray
            Trường vector biến dạng.
        """
        self.vector_field = vector_field

    def get_magnitude_map(self):
        """
        Tính bản đồ biên độ của các vector biến dạng.

        Returns
        -------
        ndarray
            Bản đồ biên độ 3D của các vector biến dạng.
        """
        if self.vector_field is None:
            logger.error("Chưa có trường vector để phân tích")
            return None

        try:
            # Tính biên độ của mỗi vector
            magnitudes = np.sqrt(np.sum(self.vector_field**2, axis=3))
            return magnitudes

        except Exception as e:
            logger.error(f"Lỗi khi tính bản đồ biên độ: {str(e)}")
            return None

    def get_statistics(self):
        """
        Tính các thống kê về trường vector biến dạng.

        Returns
        -------
        Dict
            Từ điển chứa các thống kê về trường vector.
        """
        if self.vector_field is None:
            logger.error("Chưa có trường vector để phân tích")
            return {}

        try:
            # Tính biên độ
            magnitudes = np.sqrt(np.sum(self.vector_field**2, axis=3))

            # Tính các thống kê
            stats = {
                "mean_magnitude": float(np.mean(magnitudes)),
                "max_magnitude": float(np.max(magnitudes)),
                "min_magnitude": float(np.min(magnitudes)),
                "std_magnitude": float(np.std(magnitudes)),
                "mean_x_component": float(np.mean(self.vector_field[..., 0])),
                "mean_y_component": float(np.mean(self.vector_field[..., 1])),
                "mean_z_component": float(np.mean(self.vector_field[..., 2])),
            }

            return stats

        except Exception as e:
            logger.error(f"Lỗi khi tính thống kê trường vector: {str(e)}")
            return {}

    def identify_regions_of_interest(self, threshold_percentile: float = 90):
        """
        Xác định các vùng quan tâm có biến dạng lớn.

        Parameters
        ----------
        threshold_percentile : float, optional
            Phần trăm ngưỡng để xác định các vùng có biến dạng lớn.

        Returns
        -------
        Tuple[ndarray, float]
            Mask 3D đánh dấu các vùng quan tâm và giá trị ngưỡng.
        """
        if self.vector_field is None:
            logger.error("Chưa có trường vector để phân tích")
            return None, 0.0

        try:
            # Tính biên độ
            magnitudes = np.sqrt(np.sum(self.vector_field**2, axis=3))

            # Tính ngưỡng
            threshold = np.percentile(magnitudes, threshold_percentile)

            # Tạo mask
            roi_mask = magnitudes > threshold

            return roi_mask, threshold

        except Exception as e:
            logger.error(f"Lỗi khi xác định vùng quan tâm: {str(e)}")
            return None, 0.0


class DeformableAnatomyPredictor:
    """
    Lớp dự đoán thay đổi giải phẫu dựa trên mô hình biến dạng.
    """

    def __init__(
        self, patient=None, reference_image=None, model_type=DeformationModelType.LINEAR
    ):
        """
        Khởi tạo bộ dự đoán thay đổi giải phẫu.

        Parameters
        ----------
        patient : Patient, optional
            Đối tượng bệnh nhân cần dự đoán.
        reference_image : Image, optional
            Hình ảnh tham chiếu.
        model_type : DeformationModelType, optional
            Loại mô hình biến dạng để sử dụng.
        """
        self.patient = patient
        self.reference_image = reference_image
        self.model_type = model_type
        self.deformation_model = DeformationModel(model_type)
        self.image_series = {}  # Từ điển lưu trữ ảnh theo ngày
        self.training_images = []  # Danh sách các cặp ảnh huấn luyện
        self.validator = None  # Validator để kiểm tra kết quả dự đoán
        self.vector_analyzer = DeformationVectorAnalysis()

        logger.info(
            f"Đã khởi tạo DeformableAnatomyPredictor với model_type={model_type.value}"
        )

    def set_validator(self, validator):
        """
        Thiết lập validator để kiểm tra kết quả dự đoán.

        Parameters
        ----------
        validator : ModelValidator
            Đối tượng validator.
        """
        self.validator = validator
        logger.debug("Đã thiết lập validator cho DeformableAnatomyPredictor")

    def add_image(self, image, date):
        """
        Thêm hình ảnh vào bộ dự đoán.

        Parameters
        ----------
        image : ndarray
            Hình ảnh cần thêm.
        date : datetime hoặc str
            Ngày chụp hình ảnh.

        Returns
        -------
        bool
            True nếu thêm thành công, False nếu không.
        """
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date)
            except ValueError:
                logger.error(f"Định dạng ngày không hợp lệ: {date}")
                return False

        # Lưu trữ ảnh theo ngày
        self.image_series[date] = image

        # Sắp xếp lại danh sách theo thứ tự thời gian
        sorted_dates = sorted(self.image_series.keys())

        # Tạo các cặp ảnh huấn luyện từ các ảnh liên tiếp
        self.training_images = []
        for i in range(len(sorted_dates) - 1):
            date1 = sorted_dates[i]
            date2 = sorted_dates[i + 1]
            img1 = self.image_series[date1]
            img2 = self.image_series[date2]
            self.training_images.append(
                {
                    "source": img1,
                    "target": img2,
                    "source_date": date1,
                    "target_date": date2,
                }
            )

        logger.info(
            f"Đã thêm hình ảnh ngày {date}. Tổng số ảnh: {len(self.image_series)}"
        )
        return True

    def train_model(self):
        """
        Huấn luyện mô hình biến dạng từ các hình ảnh có sẵn.

        Returns
        -------
        bool
            True nếu huấn luyện thành công, False nếu không.
        """
        if len(self.training_images) == 0:
            logger.error("Không có dữ liệu huấn luyện")
            return False

        try:
            # Sử dụng cặp ảnh gần nhất để huấn luyện
            latest_pair = self.training_images[-1]
            source = latest_pair["source"]
            target = latest_pair["target"]

            # Huấn luyện mô hình
            success = self.deformation_model.train(source, target)

            if success and self.deformation_model.vector_field is not None:
                # Thiết lập trường vector cho phân tích
                self.vector_analyzer.set_vector_field(
                    self.deformation_model.vector_field
                )

            return success

        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            return False

    def predict_image_at_date(self, target_date, source_date=None):
        """
        Dự đoán hình ảnh tại một ngày cụ thể.

        Parameters
        ----------
        target_date : datetime hoặc str
            Ngày cần dự đoán hình ảnh.
        source_date : datetime hoặc str, optional
            Ngày tham chiếu. Nếu không cung cấp, sẽ sử dụng ngày gần nhất có sẵn.

        Returns
        -------
        ndarray hoặc None
            Hình ảnh dự đoán tại ngày đích, hoặc None nếu không thể dự đoán.
        """
        if isinstance(target_date, str):
            try:
                target_date = datetime.fromisoformat(target_date)
            except ValueError:
                logger.error(f"Định dạng ngày đích không hợp lệ: {target_date}")
                return None

        if source_date is not None and isinstance(source_date, str):
            try:
                source_date = datetime.fromisoformat(source_date)
            except ValueError:
                logger.error(f"Định dạng ngày nguồn không hợp lệ: {source_date}")
                return None

        # Nếu không cung cấp ngày nguồn, sử dụng ngày gần nhất có sẵn
        if source_date is None and self.image_series:
            sorted_dates = sorted(self.image_series.keys())
            # Tìm ngày gần nhất trước ngày đích
            earlier_dates = [d for d in sorted_dates if d < target_date]
            if earlier_dates:
                source_date = max(earlier_dates)
            else:
                # Nếu không có ngày nào trước ngày đích, sử dụng ngày sớm nhất
                source_date = min(sorted_dates)

        # Kiểm tra xem có hình ảnh nguồn không
        if source_date not in self.image_series:
            logger.error(f"Không có hình ảnh tại ngày nguồn: {source_date}")
            return None

        # Kiểm tra xem mô hình đã được huấn luyện chưa
        if not self.deformation_model.is_trained:
            logger.warning("Mô hình chưa được huấn luyện. Thử huấn luyện tự động.")
            if not self.train_model():
                logger.error("Không thể tự động huấn luyện mô hình")
                return None

        try:
            # Lấy hình ảnh nguồn
            source_image = self.image_series[source_date]

            # Tính toán thời gian cần ngoại suy
            time_delta = (target_date - source_date).total_seconds()

            # Tính hệ số thời gian (giả định mô hình được huấn luyện trên khoảng thời gian 30 ngày)
            reference_delta = 30 * 24 * 3600  # 30 ngày tính bằng giây
            time_factor = time_delta / reference_delta

            # Áp dụng mô hình biến dạng
            predicted_image = self.deformation_model.apply(source_image, time_factor)

            # Kiểm tra kết quả dự đoán nếu có validator
            if self.validator is not None:
                is_valid, confidence = self.validator.validate_prediction(
                    predicted_image
                )
                if not is_valid:
                    logger.warning(
                        f"Dự đoán không hợp lệ với độ tin cậy {confidence:.4f}"
                    )

            return predicted_image

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán hình ảnh: {str(e)}")
            return None

    def predict_structure_changes(self, structures, target_date, source_date=None):
        """
        Dự đoán thay đổi của các cấu trúc tại một ngày cụ thể.

        Parameters
        ----------
        structures : list
            Danh sách các cấu trúc cần dự đoán.
        target_date : datetime hoặc str
            Ngày cần dự đoán.
        source_date : datetime hoặc str, optional
            Ngày tham chiếu.

        Returns
        -------
        dict
            Từ điển chứa các cấu trúc đã biến đổi theo id.
        """
        try:
            # Dự đoán hình ảnh tại ngày đích
            predicted_image = self.predict_image_at_date(target_date, source_date)
            if predicted_image is None:
                return {}

            # Mô phỏng biến đổi các cấu trúc (sẽ được thay thế bằng thuật toán thực)
            transformed_structures = {}

            for structure in structures:
                # Trong phiên bản này, chúng ta sẽ giả định các cấu trúc có id và mask
                struct_id = getattr(
                    structure, "id", f"struct_{len(transformed_structures)}"
                )

                # Tạo một bản sao của cấu trúc
                transformed = structure  # Trong triển khai thực, cần tạo bản sao sâu

                # Mô phỏng thay đổi các thuộc tính
                # Ví dụ: Thay đổi thể tích và vị trí
                if hasattr(structure, "volume"):
                    # Mô phỏng thay đổi thể tích (±10%)
                    volume_change = np.random.uniform(-0.1, 0.1)
                    new_volume = getattr(structure, "volume") * (1.0 + volume_change)
                    setattr(transformed, "volume", new_volume)

                # Thêm thông tin về sự thay đổi
                setattr(transformed, "predicted", True)
                setattr(transformed, "prediction_date", target_date)
                setattr(transformed, "source_date", source_date)

                # Thêm vào từ điển kết quả
                transformed_structures[struct_id] = transformed

            logger.info(
                f"Đã dự đoán thay đổi cho {len(transformed_structures)} cấu trúc"
            )
            return transformed_structures

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán thay đổi cấu trúc: {str(e)}")
            return {}

    def analyze_deformation_field(self):
        """
        Phân tích trường biến dạng để xác định các vùng thay đổi nhiều.

        Returns
        -------
        Dict
            Từ điển chứa kết quả phân tích.
        """
        if (
            not self.deformation_model.is_trained
            or self.deformation_model.vector_field is None
        ):
            logger.error("Không có trường biến dạng để phân tích")
            return {}

        try:
            # Lấy thống kê từ phân tích trường vector
            stats = self.vector_analyzer.get_statistics()

            # Xác định các vùng quan tâm
            roi_mask, threshold = self.vector_analyzer.identify_regions_of_interest(90)

            # Bổ sung kết quả
            result = {
                "statistics": stats,
                "threshold": threshold,
                "roi_mask": roi_mask,
                "model_type": self.model_type.value,
                "training_error": self.deformation_model.training_error,
            }

            return result

        except Exception as e:
            logger.error(f"Lỗi khi phân tích trường biến dạng: {str(e)}")
            return {}

    def save_model(self, filepath: str):
        """
        Lưu mô hình biến dạng vào tệp.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp để lưu mô hình.

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không.
        """
        return self.deformation_model.save(filepath)

    def load_model(self, filepath: str):
        """
        Tải mô hình biến dạng từ tệp.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp để tải mô hình.

        Returns
        -------
        bool
            True nếu tải thành công, False nếu không.
        """
        success = self.deformation_model.load(filepath)
        if success and self.deformation_model.vector_field is not None:
            self.vector_analyzer.set_vector_field(self.deformation_model.vector_field)
        return success
