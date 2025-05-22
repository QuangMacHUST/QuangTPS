#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module dự đoán thay đổi giải phẫu dựa trên các mô hình thống kê.

Module này cung cấp các lớp và hàm để dự đoán thay đổi giải phẫu theo thời gian
sử dụng các phương pháp thống kê, hỗ trợ cho lập kế hoạch xạ trị thích ứng.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence
from enum import Enum, auto
import joblib
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from quangtps.core.types import Patient, Image, Structure
from quangtps.core.exceptions import PredictionError
from quangtps.adaptive.prediction.anatomy_prediction import (
    AnatomyPrediction,
    PredictionMethod,
)
from quangtps.segmentation.contour.dice import calculate_dice_coefficient

logger = logging.getLogger(__name__)


class StatisticalModelType(Enum):
    """Các loại mô hình thống kê được hỗ trợ."""

    LINEAR = auto()  # Hồi quy tuyến tính đơn giản
    RIDGE = auto()  # Hồi quy Ridge với điều chuẩn L2
    ELASTIC_NET = auto()  # Hồi quy ElasticNet (L1 + L2)
    RANDOM_FOREST = auto()  # Random Forest
    GRADIENT_BOOSTING = auto()  # Gradient Boosting
    POLYNOMIAL = auto()  # Hồi quy đa thức


class StatisticalPredictor:
    """
    Lớp dự đoán thay đổi giải phẫu sử dụng các phương pháp thống kê.

    Lớp này hỗ trợ nhiều phương pháp dự đoán thống kê khác nhau để dự đoán
    thay đổi giải phẫu theo thời gian, từ các phương pháp đơn giản như hồi quy
    tuyến tính đến các phương pháp phức tạp hơn như Random Forest và Gradient Boosting.
    """

    def __init__(
        self, model_type: StatisticalModelType = StatisticalModelType.GRADIENT_BOOSTING
    ):
        """
        Khởi tạo bộ dự đoán thay đổi giải phẫu thống kê.

        Parameters
        ----------
        model_type : StatisticalModelType, optional
            Loại mô hình thống kê để sử dụng, mặc định là GRADIENT_BOOSTING
        """
        self.model_type = model_type
        self.models = {}  # Dict chứa các mô hình cho từng cấu trúc
        self.scalers = {}  # Dict chứa các scaler cho từng cấu trúc
        self.feature_names = []
        self.trained = False
        self.confidence_model = None  # Mô hình dự đoán độ tin cậy
        self.min_samples = 5  # Số lượng mẫu tối thiểu để huấn luyện mô hình

    def train(
        self, historical_data: List[Dict[str, Any]], structure_names: List[str] = None
    ) -> bool:
        """
        Huấn luyện mô hình dự đoán thống kê từ dữ liệu lịch sử.

        Parameters
        ----------
        historical_data : List[Dict[str, Any]]
            Danh sách từ điển chứa dữ liệu lịch sử của bệnh nhân
            Mỗi từ điển cần có: 'date', 'structures', 'images', 'patient_id'
        structure_names : List[str], optional
            Danh sách tên cấu trúc cần huấn luyện mô hình, nếu None thì huấn luyện cho tất cả

        Returns
        -------
        bool
            True nếu huấn luyện thành công, False nếu không
        """
        if len(historical_data) < self.min_samples:
            logger.warning(
                f"Không đủ dữ liệu để huấn luyện ({len(historical_data)}/{self.min_samples})"
            )
            return False

        try:
            # Chuẩn bị dữ liệu huấn luyện
            processed_data = self._prepare_training_data(
                historical_data, structure_names
            )

            if not processed_data:
                logger.error("Không thể chuẩn bị dữ liệu huấn luyện")
                return False

            # Huấn luyện mô hình cho từng cấu trúc
            for struct_name, data in processed_data.items():
                logger.info(f"Huấn luyện mô hình cho cấu trúc {struct_name}")

                X = data["features"]
                y_volume = data["volume_changes"]
                y_position = data["position_changes"]
                y_shape = data["shape_changes"]

                # Chuẩn hóa dữ liệu
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[struct_name] = scaler

                # Tạo mô hình cho cấu trúc
                self.models[struct_name] = {
                    "volume": self._create_model().fit(X_scaled, y_volume),
                    "position": self._create_model().fit(X_scaled, y_position),
                    "shape": self._create_model().fit(X_scaled, y_shape),
                }

                # Lưu tên đặc trưng nếu chưa có
                if not self.feature_names and "feature_names" in data:
                    self.feature_names = data["feature_names"]

            # Huấn luyện mô hình dự đoán độ tin cậy
            self._train_confidence_model(processed_data)

            self.trained = True
            logger.info(f"Đã huấn luyện mô hình cho {len(self.models)} cấu trúc")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            return False

    def predict_multiple_timepoints(
        self,
        patient: Patient,
        reference_structures: Dict[str, Structure],
        reference_image: Image,
        timepoints: List[datetime.datetime],
    ) -> Dict[datetime.datetime, Dict[str, Any]]:
        """
        Dự đoán thay đổi giải phẫu tại nhiều thời điểm khác nhau.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        reference_structures : Dict[str, Structure]
            Từ điển chứa cấu trúc tham chiếu (hiện tại)
        reference_image : Image
            Hình ảnh tham chiếu (hiện tại)
        timepoints : List[datetime.datetime]
            Danh sách các thời điểm cần dự đoán

        Returns
        -------
        Dict[datetime.datetime, Dict[str, Any]]
            Từ điển chứa kết quả dự đoán cho từng thời điểm
        """
        if not self.trained:
            raise PredictionError("Mô hình chưa được huấn luyện")

        if not timepoints:
            logger.warning("Không có thời điểm nào để dự đoán")
            return {}

        results = {}
        reference_time = datetime.datetime.now()

        # Tính toán đặc trưng từ dữ liệu tham chiếu
        features = self._extract_features(
            patient, reference_structures, reference_image
        )

        # Dự đoán cho từng thời điểm
        for tp in timepoints:
            # Tính số ngày giữa thời điểm tham chiếu và thời điểm dự đoán
            days_delta = (tp - reference_time).days
            time_feature = np.array([days_delta])  # Đặc trưng thời gian

            # Dự đoán cho từng cấu trúc
            structures_predictions = {}
            for struct_name, struct in reference_structures.items():
                # Kiểm tra xem có mô hình cho cấu trúc này không
                if struct_name not in self.models:
                    logger.warning(f"Không có mô hình cho cấu trúc {struct_name}")
                    continue

                # Kết hợp đặc trưng thời gian với đặc trưng cấu trúc
                combined_features = np.concatenate(
                    [features[struct_name], time_feature]
                )

                # Chuẩn hóa đặc trưng
                scaled_features = self.scalers[struct_name].transform(
                    [combined_features]
                )[0]

                # Dự đoán thay đổi
                volume_change = self.models[struct_name]["volume"].predict(
                    [scaled_features]
                )[0]
                position_change = self.models[struct_name]["position"].predict(
                    [scaled_features]
                )[0]
                shape_change = self.models[struct_name]["shape"].predict(
                    [scaled_features]
                )[0]

                # Tính độ tin cậy
                confidence = self._predict_confidence(
                    struct_name, scaled_features, days_delta
                )

                # Lưu kết quả dự đoán
                structures_predictions[struct_name] = {
                    "volume_change": float(
                        volume_change
                    ),  # Phần trăm thay đổi thể tích
                    "position_change": position_change,  # Vector thay đổi vị trí (mm)
                    "shape_change": float(
                        shape_change
                    ),  # Chỉ số thay đổi hình dạng (0-1)
                    "confidence": float(confidence),  # Độ tin cậy của dự đoán (0-1)
                }

            results[tp] = structures_predictions

        return results

    def predict_anatomy(
        self,
        patient: Patient,
        reference_structures: Dict[str, Structure],
        reference_image: Image,
        prediction_days: List[int],
    ) -> AnatomyPrediction:
        """
        Dự đoán thay đổi giải phẫu và trả về đối tượng AnatomyPrediction.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        reference_structures : Dict[str, Structure]
            Từ điển chứa cấu trúc tham chiếu (hiện tại)
        reference_image : Image
            Hình ảnh tham chiếu (hiện tại)
        prediction_days : List[int]
            Danh sách số ngày tính từ hiện tại cần dự đoán

        Returns
        -------
        AnatomyPrediction
            Kết quả dự đoán thay đổi giải phẫu
        """
        reference_time = datetime.datetime.now()
        prediction = AnatomyPrediction(reference_time, patient.id)

        # Tạo danh sách thời điểm dự đoán
        prediction_dates = [
            reference_time + datetime.timedelta(days=days) for days in prediction_days
        ]

        # Dự đoán thay đổi cho nhiều thời điểm
        predictions = self.predict_multiple_timepoints(
            patient, reference_structures, reference_image, prediction_dates
        )

        # Chuyển đổi kết quả dự đoán sang đối tượng AnatomyPrediction
        for date, struct_predictions in predictions.items():
            # Tạo bản sao của cấu trúc tham chiếu và điều chỉnh theo dự đoán
            predicted_structures = {}
            predicted_image = None  # Chưa hỗ trợ dự đoán hình ảnh

            for struct_name, pred in struct_predictions.items():
                if struct_name in reference_structures:
                    # Tạo bản sao của cấu trúc tham chiếu
                    new_struct = reference_structures[struct_name].create_copy()

                    # Điều chỉnh thể tích dựa trên dự đoán
                    volume_change = pred["volume_change"]
                    current_volume = new_struct.get_volume()
                    new_volume = current_volume * (1.0 + volume_change / 100.0)

                    # Lưu ý: Đây chỉ là mô phỏng, cần triển khai thực tế để thay đổi hình dạng cấu trúc
                    # Trong triển khai thực tế, cần biến đổi mask thực sự
                    new_struct.volume = new_volume

                    predicted_structures[struct_name] = new_struct

            # Thêm vào kết quả dự đoán với độ tin cậy trung bình
            avg_confidence = np.mean(
                [pred["confidence"] for pred in struct_predictions.values()]
            )
            prediction.add_prediction_timepoint(
                date, predicted_structures, predicted_image, avg_confidence
            )

        return prediction

    def visualize_predictions(
        self,
        structure_name: str,
        prediction_days: List[int],
        reference_value: float = 100.0,
    ) -> None:
        """
        Trực quan hóa kết quả dự đoán thay đổi thể tích cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần trực quan hóa
        prediction_days : List[int]
            Danh sách các ngày cần dự đoán
        reference_value : float, optional
            Giá trị tham chiếu (100% là mặc định)
        """
        if structure_name not in self.models:
            logger.error(f"Không có mô hình cho cấu trúc {structure_name}")
            return

        try:
            # Tạo dữ liệu mẫu để dự đoán
            sample_features = np.zeros(self.scalers[structure_name].n_features_in_ - 1)

            # Dự đoán thay đổi thể tích theo thời gian
            volumes = []
            confidences = []

            for days in prediction_days:
                # Kết hợp đặc trưng với thời gian
                combined_features = np.concatenate([sample_features, np.array([days])])
                scaled_features = self.scalers[structure_name].transform(
                    [combined_features]
                )[0]

                # Dự đoán thay đổi thể tích
                volume_change = self.models[structure_name]["volume"].predict(
                    [scaled_features]
                )[0]
                volumes.append(reference_value * (1.0 + volume_change / 100.0))

                # Dự đoán độ tin cậy
                confidence = self._predict_confidence(
                    structure_name, scaled_features, days
                )
                confidences.append(confidence)

            # Tạo biểu đồ
            plt.figure(figsize=(10, 6))

            # Vẽ đường thay đổi thể tích
            plt.plot(prediction_days, volumes, "b-", linewidth=2)

            # Vẽ dải độ tin cậy
            error_range = [
                (1.0 - conf) * vol * 0.2 for vol, conf in zip(volumes, confidences)
            ]
            plt.fill_between(
                prediction_days,
                [v - e for v, e in zip(volumes, error_range)],
                [v + e for v, e in zip(volumes, error_range)],
                alpha=0.3,
                color="blue",
            )

            # Thêm đường tham chiếu
            plt.axhline(y=reference_value, color="r", linestyle="--")

            # Thêm nhãn và tiêu đề
            plt.xlabel("Ngày")
            plt.ylabel("Thể tích (%)")
            plt.title(f"Dự đoán thay đổi thể tích cấu trúc {structure_name}")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Lỗi khi trực quan hóa dự đoán: {str(e)}")

    def save_model(self, filepath: str) -> bool:
        """
        Lưu mô hình đã huấn luyện vào file.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến file lưu mô hình

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không
        """
        if not self.trained:
            logger.warning("Mô hình chưa được huấn luyện, không thể lưu")
            return False

        try:
            model_data = {
                "models": self.models,
                "scalers": self.scalers,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "trained": self.trained,
                "confidence_model": self.confidence_model,
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Đã lưu mô hình vào {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Tải mô hình từ file.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến file chứa mô hình

        Returns
        -------
        bool
            True nếu tải thành công, False nếu không
        """
        try:
            model_data = joblib.load(filepath)

            self.models = model_data["models"]
            self.scalers = model_data["scalers"]
            self.feature_names = model_data["feature_names"]
            self.model_type = model_data["model_type"]
            self.trained = model_data["trained"]
            self.confidence_model = model_data.get("confidence_model")

            logger.info(f"Đã tải mô hình từ {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False

    def _create_model(self):
        """
        Tạo mô hình thống kê dựa trên loại đã chọn.

        Returns
        -------
        object
            Đối tượng mô hình thống kê
        """
        if self.model_type == StatisticalModelType.LINEAR:
            return LinearRegression()
        elif self.model_type == StatisticalModelType.RIDGE:
            return Ridge(alpha=1.0)
        elif self.model_type == StatisticalModelType.ELASTIC_NET:
            return ElasticNet(alpha=1.0, l1_ratio=0.5)
        elif self.model_type == StatisticalModelType.RANDOM_FOREST:
            return RandomForestRegressor(n_estimators=100, n_jobs=-1)
        elif self.model_type == StatisticalModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(n_estimators=100)
        elif self.model_type == StatisticalModelType.POLYNOMIAL:
            # Đối với đa thức, ta sẽ xử lý đặc biệt sau
            return LinearRegression()
        else:
            # Mặc định là Gradient Boosting
            return GradientBoostingRegressor(n_estimators=100)

    def _prepare_training_data(
        self, historical_data: List[Dict[str, Any]], structure_names: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Chuẩn bị dữ liệu huấn luyện từ dữ liệu lịch sử.

        Parameters
        ----------
        historical_data : List[Dict[str, Any]]
            Danh sách từ điển chứa dữ liệu lịch sử
        structure_names : List[str], optional
            Danh sách tên cấu trúc cần huấn luyện, nếu None thì huấn luyện cho tất cả

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Từ điển chứa dữ liệu huấn luyện đã chuẩn bị
        """
        try:
            # Sắp xếp dữ liệu theo thời gian
            sorted_data = sorted(historical_data, key=lambda x: x["date"])

            # Xác định danh sách cấu trúc cần xử lý
            if not structure_names:
                structure_names = set()
                for data in sorted_data:
                    structure_names.update(data["structures"].keys())
                structure_names = list(structure_names)

            # Chuẩn bị dữ liệu cho từng cấu trúc
            result = {}
            for struct_name in structure_names:
                struct_data = {
                    "features": [],
                    "volume_changes": [],
                    "position_changes": [],
                    "shape_changes": [],
                    "feature_names": [],
                }

                # Tham chiếu đến dữ liệu đầu tiên
                ref_data = sorted_data[0]
                if struct_name not in ref_data["structures"]:
                    logger.warning(
                        f"Cấu trúc {struct_name} không có trong dữ liệu tham chiếu"
                    )
                    continue

                ref_struct = ref_data["structures"][struct_name]
                ref_time = ref_data["date"]
                ref_volume = ref_struct.get_volume()
                ref_centroid = ref_struct.get_centroid()

                # Thu thập dữ liệu từ các thời điểm sau
                for i in range(1, len(sorted_data)):
                    curr_data = sorted_data[i]

                    if struct_name not in curr_data["structures"]:
                        continue

                    curr_struct = curr_data["structures"][struct_name]
                    curr_time = curr_data["date"]

                    # Tính số ngày từ thời điểm tham chiếu
                    days_delta = (curr_time - ref_time).days

                    # Trích xuất đặc trưng
                    features = self._extract_structure_features(
                        ref_struct,
                        curr_struct,
                        ref_data["patient_id"],
                        curr_data["patient_id"],
                    )

                    # Thêm đặc trưng thời gian
                    features.append(days_delta)

                    # Tính thay đổi thể tích (phần trăm)
                    curr_volume = curr_struct.get_volume()
                    volume_change = ((curr_volume - ref_volume) / ref_volume) * 100

                    # Tính thay đổi vị trí (khoảng cách Euclid giữa tâm)
                    curr_centroid = curr_struct.get_centroid()
                    position_change = np.linalg.norm(curr_centroid - ref_centroid)

                    # Tính thay đổi hình dạng (dựa trên chỉ số Dice)
                    try:
                        dice = calculate_dice_coefficient(ref_struct, curr_struct)
                        shape_change = (
                            1.0 - dice
                        )  # Càng khác nhau, shape_change càng gần 1
                    except Exception:
                        shape_change = 0.0  # Giả sử không thay đổi nếu không tính được

                    # Thêm vào dữ liệu huấn luyện
                    struct_data["features"].append(features)
                    struct_data["volume_changes"].append(volume_change)
                    struct_data["position_changes"].append(position_change)
                    struct_data["shape_changes"].append(shape_change)

                # Chỉ lưu dữ liệu nếu có đủ mẫu
                if len(struct_data["features"]) >= self.min_samples:
                    # Chuyển dữ liệu sang định dạng numpy
                    struct_data["features"] = np.array(struct_data["features"])
                    struct_data["volume_changes"] = np.array(
                        struct_data["volume_changes"]
                    )
                    struct_data["position_changes"] = np.array(
                        struct_data["position_changes"]
                    )
                    struct_data["shape_changes"] = np.array(
                        struct_data["shape_changes"]
                    )

                    # Tạo danh sách tên đặc trưng
                    feature_names = [
                        "volume",
                        "surface_area",
                        "compactness",
                        "centroid_x",
                        "centroid_y",
                        "centroid_z",
                        "days_delta",
                    ]
                    struct_data["feature_names"] = feature_names

                    result[struct_name] = struct_data

            return result

        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dữ liệu huấn luyện: {str(e)}")
            return {}

    def _extract_features(
        self, patient: Patient, structures: Dict[str, Structure], image: Image
    ) -> Dict[str, np.ndarray]:
        """
        Trích xuất đặc trưng từ dữ liệu hiện tại.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        structures : Dict[str, Structure]
            Từ điển chứa cấu trúc
        image : Image
            Đối tượng hình ảnh

        Returns
        -------
        Dict[str, np.ndarray]
            Từ điển chứa đặc trưng cho từng cấu trúc
        """
        result = {}

        for struct_name, struct in structures.items():
            try:
                features = []

                # Đặc trưng kích thước và hình dạng
                volume = struct.get_volume()
                features.append(volume)

                surface_area = struct.get_surface_area()
                features.append(surface_area)

                # Tính đặc trưng compactness (sphericity)
                compactness = (surface_area**1.5) / (6 * np.sqrt(np.pi) * volume)
                features.append(compactness)

                # Vị trí tâm
                centroid = struct.get_centroid()
                features.extend(centroid)

                result[struct_name] = np.array(features)

            except Exception as e:
                logger.warning(
                    f"Lỗi khi trích xuất đặc trưng cho {struct_name}: {str(e)}"
                )

        return result

    def _extract_structure_features(
        self,
        ref_struct: Structure,
        curr_struct: Structure,
        ref_patient_id: str,
        curr_patient_id: str,
    ) -> List[float]:
        """
        Trích xuất đặc trưng từ hai cấu trúc để huấn luyện.

        Parameters
        ----------
        ref_struct : Structure
            Cấu trúc tham chiếu
        curr_struct : Structure
            Cấu trúc hiện tại
        ref_patient_id : str
            ID bệnh nhân tham chiếu
        curr_patient_id : str
            ID bệnh nhân hiện tại

        Returns
        -------
        List[float]
            Danh sách các đặc trưng
        """
        features = []

        # Thêm đặc trưng từ cấu trúc tham chiếu
        ref_volume = ref_struct.get_volume()
        features.append(ref_volume)

        ref_surface_area = ref_struct.get_surface_area()
        features.append(ref_surface_area)

        # Tính đặc trưng compactness (sphericity)
        ref_compactness = (ref_surface_area**1.5) / (6 * np.sqrt(np.pi) * ref_volume)
        features.append(ref_compactness)

        # Vị trí tâm
        ref_centroid = ref_struct.get_centroid()
        features.extend(ref_centroid)

        return features

    def _train_confidence_model(
        self, processed_data: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Huấn luyện mô hình dự đoán độ tin cậy.

        Parameters
        ----------
        processed_data : Dict[str, Dict[str, Any]]
            Dữ liệu đã xử lý cho việc huấn luyện
        """
        try:
            # Kết hợp dữ liệu từ tất cả các cấu trúc
            all_features = []
            all_errors = []

            for struct_name, data in processed_data.items():
                # Lấy đặc trưng đã chuẩn hóa
                X = self.scalers[struct_name].transform(data["features"])

                # Lấy thực tế và dự đoán
                y_true_volume = data["volume_changes"]
                y_pred_volume = self.models[struct_name]["volume"].predict(X)

                # Tính lỗi tuyệt đối
                errors = np.abs(y_true_volume - y_pred_volume)

                # Thêm vào dữ liệu huấn luyện cho mô hình độ tin cậy
                all_features.extend(X)
                all_errors.extend(errors)

            if not all_features:
                logger.warning("Không có dữ liệu để huấn luyện mô hình độ tin cậy")
                return

            # Chuyển sang định dạng numpy
            all_features = np.array(all_features)
            all_errors = np.array(all_errors)

            # Chuẩn hóa lỗi thành độ tin cậy (0-1)
            # Lỗi lớn -> độ tin cậy thấp
            max_error = np.max(all_errors) if len(all_errors) > 0 else 1.0
            confidences = 1.0 - (all_errors / max_error)

            # Huấn luyện mô hình độ tin cậy
            self.confidence_model = RandomForestRegressor(n_estimators=50)
            self.confidence_model.fit(all_features, confidences)

            logger.info("Đã huấn luyện mô hình dự đoán độ tin cậy")

        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình độ tin cậy: {str(e)}")

    def _predict_confidence(
        self, struct_name: str, features: np.ndarray, days_delta: int
    ) -> float:
        """
        Dự đoán độ tin cậy cho một dự đoán.

        Parameters
        ----------
        struct_name : str
            Tên cấu trúc
        features : np.ndarray
            Đặc trưng đã chuẩn hóa
        days_delta : int
            Số ngày khác biệt

        Returns
        -------
        float
            Độ tin cậy từ 0 đến 1
        """
        try:
            if self.confidence_model is not None:
                # Sử dụng mô hình dự đoán độ tin cậy
                confidence = self.confidence_model.predict([features])[0]

                # Điều chỉnh độ tin cậy theo thời gian (càng xa càng ít tin cậy)
                time_factor = max(0.1, 1.0 / (1.0 + abs(days_delta) / 30.0))
                confidence = confidence * time_factor

                return float(max(0.0, min(1.0, confidence)))
            else:
                # Nếu không có mô hình, tính toán đơn giản dựa trên thời gian
                return max(0.1, 1.0 / (1.0 + abs(days_delta) / 30.0))

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán độ tin cậy: {str(e)}")
            return 0.5  # Giá trị mặc định


def predict_statistical_changes(
    patient: Patient,
    reference_structures: Dict[str, Structure],
    reference_image: Image,
    prediction_days: List[int],
    model_path: str = None,
) -> Dict[datetime.datetime, Dict[str, Dict[str, Any]]]:
    """
    Hàm tiện ích để dự đoán thay đổi giải phẫu sử dụng mô hình thống kê.

    Parameters
    ----------
    patient : Patient
        Đối tượng bệnh nhân
    reference_structures : Dict[str, Structure]
        Từ điển chứa cấu trúc tham chiếu (hiện tại)
    reference_image : Image
        Hình ảnh tham chiếu (hiện tại)
    prediction_days : List[int]
        Danh sách số ngày cần dự đoán
    model_path : str, optional
        Đường dẫn đến file chứa mô hình, nếu None sẽ sử dụng mô hình mặc định

    Returns
    -------
    Dict[datetime.datetime, Dict[str, Dict[str, Any]]]
        Từ điển chứa kết quả dự đoán cho từng thời điểm và cấu trúc
    """
    # Tạo bộ dự đoán
    predictor = StatisticalPredictor()

    # Tải mô hình nếu được cung cấp
    if model_path and os.path.exists(model_path):
        predictor.load_model(model_path)
    else:
        logger.warning("Không tìm thấy file mô hình, không thể dự đoán")
        return {}

    # Tạo danh sách thời điểm dự đoán
    reference_time = datetime.datetime.now()
    prediction_dates = [
        reference_time + datetime.timedelta(days=days) for days in prediction_days
    ]

    # Thực hiện dự đoán
    return predictor.predict_multiple_timepoints(
        patient, reference_structures, reference_image, prediction_dates
    )
