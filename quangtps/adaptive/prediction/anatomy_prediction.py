#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module dự đoán thay đổi giải phẫu trong QuangTPS.

Module này cung cấp các chức năng để dự đoán sự thay đổi hình ảnh và cấu trúc
trong quá trình xạ trị, hỗ trợ cho việc lập kế hoạch thích ứng chủ động.
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

from quangtps.core.types import Patient, Image, Structure, Dose, Plan
from quangtps.core.exceptions import PredictionError
from quangtps.imaging.registration import ImageRegistration
from quangtps.adaptive.deformation.deformable_registration import DeformableRegistration
from quangtps.adaptive.temporal_analysis import TemporalAnalyzer, TemporalAnalysisResult
from quangtps.adaptive.deformation.displacement_field import DisplacementField
from quangtps.core.utils import get_timestamp, create_directory_if_not_exists

logger = logging.getLogger(__name__)


class PredictionMethod(Enum):
    """Các phương pháp dự đoán thay đổi giải phẫu theo thời gian."""

    LINEAR = auto()  # Ngoại suy tuyến tính
    EXPONENTIAL = auto()  # Ngoại suy hàm mũ
    SPLINE = auto()  # Ngoại suy spline
    MACHINE_LEARNING = auto()  # Dự đoán bằng học máy (nếu có đủ dữ liệu)


class AnatomyPrediction:
    """Kết quả dự đoán thay đổi giải phẫu theo thời gian."""

    def __init__(self, reference_date: datetime.datetime, patient_id: str):
        """
        Khởi tạo đối tượng dự đoán thay đổi giải phẫu.

        Parameters
        ----------
        reference_date : datetime.datetime
            Ngày tham chiếu (ngày bắt đầu dự đoán)
        patient_id : str
            ID của bệnh nhân
        """
        self.reference_date = reference_date
        self.patient_id = patient_id
        self.prediction_date = datetime.datetime.now()
        self.prediction_timeline = []  # Các mốc thời gian dự đoán
        self.predicted_structures = {}  # Dict[datetime, Dict[str, Structure]]
        self.predicted_images = {}  # Dict[datetime, Image]
        self.confidence_scores = {}  # Dict[datetime, float]
        self.metadata = {}  # Thông tin bổ sung về dự đoán

    def add_prediction_timepoint(
        self,
        date: datetime.datetime,
        structures: Dict[str, Structure] = None,
        image: Image = None,
        confidence: float = 1.0,
    ):
        """
        Thêm một mốc thời gian dự đoán với cấu trúc và hình ảnh.

        Parameters
        ----------
        date : datetime.datetime
            Ngày dự đoán
        structures : Dict[str, Structure], optional
            Từ điển cấu trúc dự đoán, mặc định là None
        image : Image, optional
            Hình ảnh dự đoán, mặc định là None
        confidence : float, optional
            Điểm tin cậy của dự đoán (0-1), mặc định là 1.0
        """
        timepoint_info = {
            "date": date,
            "days_from_reference": (date - self.reference_date).days,
            "confidence": confidence,
        }
        self.prediction_timeline.append(timepoint_info)

        if structures:
            self.predicted_structures[date] = structures

        if image:
            self.predicted_images[date] = image

        self.confidence_scores[date] = confidence

    def get_structure_at_date(
        self, structure_name: str, date: datetime.datetime
    ) -> Optional[Structure]:
        """
        Lấy cấu trúc dự đoán tại một ngày cụ thể.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần lấy
        date : datetime.datetime
            Ngày cần lấy dự đoán

        Returns
        -------
        Optional[Structure]
            Cấu trúc dự đoán hoặc None nếu không có
        """
        if date in self.predicted_structures:
            structures = self.predicted_structures[date]
            return structures.get(structure_name)
        return None

    def get_image_at_date(self, date: datetime.datetime) -> Optional[Image]:
        """
        Lấy hình ảnh dự đoán tại một ngày cụ thể.

        Parameters
        ----------
        date : datetime.datetime
            Ngày cần lấy dự đoán

        Returns
        -------
        Optional[Image]
            Hình ảnh dự đoán hoặc None nếu không có
        """
        return self.predicted_images.get(date)

    def get_confidence_at_date(self, date: datetime.datetime) -> float:
        """
        Lấy độ tin cậy của dự đoán tại một ngày cụ thể.

        Parameters
        ----------
        date : datetime.datetime
            Ngày cần lấy độ tin cậy

        Returns
        -------
        float
            Độ tin cậy của dự đoán (0-1)
        """
        return self.confidence_scores.get(date, 0.0)

    def plot_structure_volume_prediction(
        self, structure_name: str, save_path: Optional[str] = None
    ):
        """
        Vẽ đồ thị dự đoán thể tích cấu trúc theo thời gian.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần vẽ
        save_path : Optional[str], optional
            Đường dẫn để lưu đồ thị, mặc định là None
        """
        dates = []
        volumes = []
        confidence = []

        for timepoint in self.prediction_timeline:
            date = timepoint["date"]
            struct = self.get_structure_at_date(structure_name, date)
            if struct:
                dates.append(timepoint["days_from_reference"])
                volumes.append(struct.get_volume())
                confidence.append(self.get_confidence_at_date(date))

        if not dates:
            logger.warning(f"Không có dữ liệu dự đoán cho cấu trúc {structure_name}")
            return

        plt.figure(figsize=(10, 6))

        # Vùng tin cậy
        if len(dates) > 1:
            lower_bound = [v * (1 - (1 - c) * 0.5) for v, c in zip(volumes, confidence)]
            upper_bound = [v * (1 + (1 - c) * 0.5) for v, c in zip(volumes, confidence)]
            plt.fill_between(dates, lower_bound, upper_bound, alpha=0.2, color="blue")

        plt.plot(dates, volumes, "o-", color="blue")
        plt.title(f"Dự đoán thể tích của {structure_name} theo thời gian")
        plt.xlabel("Ngày từ tham chiếu")
        plt.ylabel("Thể tích (cc)")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi kết quả dự đoán thành DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame chứa thông tin dự đoán
        """
        rows = []
        for timepoint in self.prediction_timeline:
            date = timepoint["date"]
            row = {
                "date": date,
                "days_from_reference": timepoint["days_from_reference"],
                "confidence": self.get_confidence_at_date(date),
            }

            # Thêm thông tin thể tích cấu trúc
            if date in self.predicted_structures:
                for name, struct in self.predicted_structures[date].items():
                    row[f"{name}_volume"] = struct.get_volume()

            rows.append(row)

        return pd.DataFrame(rows)


class AnatomyPredictor:
    """Lớp dự đoán thay đổi giải phẫu theo thời gian dựa trên dữ liệu lịch sử."""

    def __init__(
        self,
        temporal_analyzer: Optional[TemporalAnalyzer] = None,
        deformable_registration: Optional[DeformableRegistration] = None,
    ):
        """
        Khởi tạo bộ dự đoán thay đổi giải phẫu.

        Parameters
        ----------
        temporal_analyzer : Optional[TemporalAnalyzer], optional
            Bộ phân tích thay đổi giải phẫu theo thời gian, mặc định là None
        deformable_registration : Optional[DeformableRegistration], optional
            Đối tượng đăng ký biến dạng, mặc định là None
        """
        self.temporal_analyzer = temporal_analyzer or TemporalAnalyzer()
        self.deformable_registration = (
            deformable_registration or DeformableRegistration()
        )

    def predict_anatomy_changes(
        self,
        patient: Patient,
        historical_images: List[Image],
        historical_structures: List[Dict[str, Structure]],
        historical_dates: List[datetime.datetime],
        prediction_dates: List[datetime.datetime],
        method: PredictionMethod = PredictionMethod.SPLINE,
        target_structures: Optional[List[str]] = None,
    ) -> AnatomyPrediction:
        """
        Dự đoán thay đổi giải phẫu cho bệnh nhân dựa trên dữ liệu lịch sử.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        historical_images : List[Image]
            Danh sách hình ảnh lịch sử
        historical_structures : List[Dict[str, Structure]]
            Danh sách các từ điển cấu trúc lịch sử
        historical_dates : List[datetime.datetime]
            Danh sách các ngày tương ứng với dữ liệu lịch sử
        prediction_dates : List[datetime.datetime]
            Danh sách các ngày cần dự đoán
        method : PredictionMethod, optional
            Phương pháp dự đoán, mặc định là PredictionMethod.SPLINE
        target_structures : Optional[List[str]], optional
            Danh sách tên cấu trúc cần dự đoán, mặc định là None (dự đoán tất cả)

        Returns
        -------
        AnatomyPrediction
            Kết quả dự đoán thay đổi giải phẫu
        """
        if len(historical_images) < 2 or len(historical_structures) < 2:
            raise PredictionError("Cần ít nhất 2 điểm dữ liệu lịch sử để dự đoán")

        if len(historical_images) != len(historical_structures) or len(
            historical_images
        ) != len(historical_dates):
            raise PredictionError("Số lượng hình ảnh, cấu trúc và ngày không khớp nhau")

        # Phân tích dữ liệu lịch sử
        historical_analysis = self._analyze_historical_data(
            historical_images, historical_structures, historical_dates
        )

        # Khởi tạo kết quả dự đoán
        prediction = AnatomyPrediction(historical_dates[-1], patient.id)

        # Xác định danh sách cấu trúc cần dự đoán
        if not target_structures:
            # Lấy tất cả cấu trúc có trong lịch sử
            target_structures = set()
            for struct_dict in historical_structures:
                target_structures.update(struct_dict.keys())
            target_structures = list(target_structures)

        # Dự đoán cho từng mốc thời gian
        for pred_date in prediction_dates:
            # Tính toán độ tin cậy dựa trên khoảng cách thời gian
            max_hist_date = max(historical_dates)
            days_from_last = (pred_date - max_hist_date).days
            confidence = max(
                0.1, 1.0 / (1.0 + days_from_last / 30.0)
            )  # Giảm đều theo thời gian

            # Dự đoán cấu trúc
            predicted_structures = self._predict_structures(
                historical_structures,
                historical_dates,
                pred_date,
                method,
                target_structures,
            )

            # Dự đoán hình ảnh (nếu cần)
            predicted_image = self._predict_image(
                historical_images, historical_dates, pred_date, method
            )

            # Thêm vào kết quả dự đoán
            prediction.add_prediction_timepoint(
                pred_date, predicted_structures, predicted_image, confidence
            )

        return prediction

    def _analyze_historical_data(
        self,
        images: List[Image],
        structures: List[Dict[str, Structure]],
        dates: List[datetime.datetime],
    ) -> Dict[str, Any]:
        """
        Phân tích dữ liệu lịch sử để chuẩn bị cho việc dự đoán.

        Parameters
        ----------
        images : List[Image]
            Danh sách hình ảnh lịch sử
        structures : List[Dict[str, Structure]]
            Danh sách các từ điển cấu trúc lịch sử
        dates : List[datetime.datetime]
            Danh sách các ngày tương ứng với dữ liệu lịch sử

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa dữ liệu phân tích
        """
        # Khởi tạo kết quả phân tích
        analysis = {
            "structure_volumes": {},  # Dict[str, List[float]]
            "structure_centroids": {},  # Dict[str, List[np.ndarray]]
            "structure_metrics": {},  # Dict[str, Dict[str, List[float]]]
            "image_metrics": [],  # List[Dict[str, float]]
            "days_from_first": [],  # List[int]
        }

        # Tính ngày từ thời điểm đầu tiên
        first_date = dates[0]
        analysis["days_from_first"] = [(date - first_date).days for date in dates]

        # Phân tích cấu trúc
        all_struct_names = set()
        for struct_dict in structures:
            all_struct_names.update(struct_dict.keys())

        for struct_name in all_struct_names:
            analysis["structure_volumes"][struct_name] = []
            analysis["structure_centroids"][struct_name] = []
            analysis["structure_metrics"][struct_name] = {"dice": [], "hausdorff": []}

            for i, struct_dict in enumerate(structures):
                if struct_name in struct_dict:
                    struct = struct_dict[struct_name]
                    # Lưu thể tích
                    analysis["structure_volumes"][struct_name].append(
                        struct.get_volume()
                    )
                    # Lưu tâm
                    analysis["structure_centroids"][struct_name].append(
                        struct.get_centroid()
                    )

                    # Tính các chỉ số so với cấu trúc đầu tiên
                    if i > 0 and struct_name in structures[0]:
                        ref_struct = structures[0][struct_name]
                        from quangtps.segmentation.contour.dice import calculate_dice

                        dice = calculate_dice(ref_struct, struct)
                        analysis["structure_metrics"][struct_name]["dice"].append(dice)

                        # TODO: Tính khoảng cách Hausdorff
                        analysis["structure_metrics"][struct_name]["hausdorff"].append(
                            0.0
                        )
                else:
                    # Nếu cấu trúc không có ở mốc thời gian này, thêm giá trị NaN
                    analysis["structure_volumes"][struct_name].append(np.nan)
                    analysis["structure_centroids"][struct_name].append(
                        np.array([np.nan, np.nan, np.nan])
                    )
                    if i > 0:
                        analysis["structure_metrics"][struct_name]["dice"].append(
                            np.nan
                        )
                        analysis["structure_metrics"][struct_name]["hausdorff"].append(
                            np.nan
                        )

        # Phân tích hình ảnh
        for i, image in enumerate(images):
            metrics = {
                "mean_hu": np.mean(image.get_array()),
                "std_hu": np.std(image.get_array()),
            }

            if i > 0:
                # Tính chỉ số tương quan với hình ảnh đầu tiên
                try:
                    corr = np.corrcoef(
                        images[0].get_array().flatten(), image.get_array().flatten()
                    )[0, 1]
                    metrics["correlation"] = corr
                except:
                    metrics["correlation"] = np.nan

            analysis["image_metrics"].append(metrics)

        return analysis

    def _predict_structures(
        self,
        historical_structures: List[Dict[str, Structure]],
        historical_dates: List[datetime.datetime],
        prediction_date: datetime.datetime,
        method: PredictionMethod,
        target_structures: List[str],
    ) -> Dict[str, Structure]:
        """
        Dự đoán cấu trúc tại một thời điểm trong tương lai.

        Parameters
        ----------
        historical_structures : List[Dict[str, Structure]]
            Danh sách các từ điển cấu trúc lịch sử
        historical_dates : List[datetime.datetime]
            Danh sách các ngày tương ứng với dữ liệu lịch sử
        prediction_date : datetime.datetime
            Ngày cần dự đoán
        method : PredictionMethod
            Phương pháp dự đoán
        target_structures : List[str]
            Danh sách tên cấu trúc cần dự đoán

        Returns
        -------
        Dict[str, Structure]
            Từ điển cấu trúc dự đoán
        """
        predicted_structures = {}

        # Chuyển đổi ngày sang số ngày từ thời điểm đầu tiên
        first_date = historical_dates[0]
        days_historical = [(date - first_date).days for date in historical_dates]
        days_prediction = (prediction_date - first_date).days

        for struct_name in target_structures:
            # Lấy dữ liệu lịch sử của cấu trúc
            historical_volumes = []
            historical_indices = []

            for i, struct_dict in enumerate(historical_structures):
                if struct_name in struct_dict:
                    historical_volumes.append(struct_dict[struct_name].get_volume())
                    historical_indices.append(i)

            if not historical_volumes:
                # Không có dữ liệu lịch sử cho cấu trúc này
                continue

            # Lấy ngày tương ứng với dữ liệu cấu trúc
            struct_days = [days_historical[i] for i in historical_indices]

            # Dự đoán thể tích
            predicted_volume = self._predict_value(
                struct_days, historical_volumes, days_prediction, method
            )

            # Lấy cấu trúc gần nhất
            latest_index = max(historical_indices)
            latest_struct = historical_structures[latest_index][struct_name]

            # Tạo cấu trúc mới dựa trên cấu trúc mới nhất với thể tích được dự đoán
            # Đây là một phương pháp đơn giản, cần cải thiện để dự đoán chính xác hơn
            # trong một triển khai thực tế
            predicted_struct = latest_struct.create_copy()

            # Điều chỉnh thể tích (đơn giản hóa bằng cách chỉ thay đổi giá trị thể tích)
            # Trong triển khai thực tế, cần biến đổi cấu trúc thực sự
            predicted_struct.volume = predicted_volume

            # Thêm vào kết quả
            predicted_structures[struct_name] = predicted_struct

        return predicted_structures

    def _predict_image(
        self,
        historical_images: List[Image],
        historical_dates: List[datetime.datetime],
        prediction_date: datetime.datetime,
        method: PredictionMethod,
    ) -> Optional[Image]:
        """
        Dự đoán hình ảnh tại một thời điểm trong tương lai.

        Parameters
        ----------
        historical_images : List[Image]
            Danh sách hình ảnh lịch sử
        historical_dates : List[datetime.datetime]
            Danh sách các ngày tương ứng với dữ liệu lịch sử
        prediction_date : datetime.datetime
            Ngày cần dự đoán
        method : PredictionMethod
            Phương pháp dự đoán

        Returns
        -------
        Optional[Image]
            Hình ảnh dự đoán hoặc None nếu không thể dự đoán
        """
        # Dự đoán hình ảnh đòi hỏi thuật toán nâng cao hơn
        # Trong phiên bản đơn giản này, chúng ta chỉ trả về hình ảnh gần nhất
        # TODO: Triển khai dự đoán hình ảnh bằng cách sử dụng trường biến dạng

        # Tìm hình ảnh mới nhất
        latest_idx = np.argmax(
            [(date - historical_dates[0]).days for date in historical_dates]
        )
        latest_image = historical_images[latest_idx]

        # Tạo một bản sao của hình ảnh mới nhất
        predicted_image = latest_image.clone()

        # TODO: Điều chỉnh hình ảnh theo dự đoán

        return predicted_image

    def _predict_value(
        self, x: List[float], y: List[float], x_pred: float, method: PredictionMethod
    ) -> float:
        """
        Dự đoán giá trị tại một điểm dựa trên dữ liệu lịch sử.

        Parameters
        ----------
        x : List[float]
            Danh sách các giá trị x (thời gian)
        y : List[float]
            Danh sách các giá trị y (đo lường)
        x_pred : float
            Giá trị x cần dự đoán
        method : PredictionMethod
            Phương pháp dự đoán

        Returns
        -------
        float
            Giá trị dự đoán
        """
        # Chuyển đổi danh sách thành numpy array
        x_array = np.array(x)
        y_array = np.array(y)

        # Loại bỏ các giá trị NaN
        mask = ~np.isnan(y_array)
        x_array = x_array[mask]
        y_array = y_array[mask]

        if len(x_array) < 2:
            # Không đủ dữ liệu để dự đoán
            if len(y_array) > 0:
                return y_array[0]  # Trả về giá trị đầu tiên
            return 0.0

        # Dự đoán dựa trên phương pháp đã chọn
        if method == PredictionMethod.LINEAR:
            # Ngoại suy tuyến tính
            model = np.polyfit(x_array, y_array, 1)
            return np.polyval(model, x_pred)

        elif method == PredictionMethod.EXPONENTIAL:
            # Ngoại suy hàm mũ
            # Chỉ áp dụng nếu tất cả y > 0
            if np.all(y_array > 0):
                log_y = np.log(y_array)
                model = np.polyfit(x_array, log_y, 1)
                return np.exp(np.polyval(model, x_pred))
            else:
                # Fallback to linear if y contains non-positive values
                model = np.polyfit(x_array, y_array, 1)
                return np.polyval(model, x_pred)

        elif method == PredictionMethod.SPLINE:
            # Ngoại suy spline
            if len(x_array) >= 3:
                spline = interpolate.splrep(
                    x_array, y_array, k=min(3, len(x_array) - 1)
                )
                return float(interpolate.splev(x_pred, spline))
            else:
                # Fallback to linear if not enough points for spline
                model = np.polyfit(x_array, y_array, 1)
                return np.polyval(model, x_pred)

        elif method == PredictionMethod.MACHINE_LEARNING:
            # TODO: Triển khai dự đoán bằng học máy
            # Fallback to spline for now
            if len(x_array) >= 3:
                spline = interpolate.splrep(
                    x_array, y_array, k=min(3, len(x_array) - 1)
                )
                return float(interpolate.splev(x_pred, spline))
            else:
                model = np.polyfit(x_array, y_array, 1)
                return np.polyval(model, x_pred)

        # Mặc định: ngoại suy tuyến tính
        model = np.polyfit(x_array, y_array, 1)
        return np.polyval(model, x_pred)


def predict_anatomy_changes(
    patient: Patient,
    historical_images: List[Image],
    historical_structures: List[Dict[str, Structure]],
    historical_dates: List[datetime.datetime],
    prediction_days: List[int],
    method: PredictionMethod = PredictionMethod.SPLINE,
) -> AnatomyPrediction:
    """
    Hàm tiện ích để dự đoán thay đổi giải phẫu.

    Parameters
    ----------
    patient : Patient
        Đối tượng bệnh nhân
    historical_images : List[Image]
        Danh sách hình ảnh lịch sử
    historical_structures : List[Dict[str, Structure]]
        Danh sách các từ điển cấu trúc lịch sử
    historical_dates : List[datetime.datetime]
        Danh sách các ngày tương ứng với dữ liệu lịch sử
    prediction_days : List[int]
        Danh sách số ngày kể từ ngày mới nhất cần dự đoán
    method : PredictionMethod, optional
        Phương pháp dự đoán, mặc định là PredictionMethod.SPLINE

    Returns
    -------
    AnatomyPrediction
        Kết quả dự đoán thay đổi giải phẫu
    """
    predictor = AnatomyPredictor()

    # Tính toán các ngày dự đoán
    latest_date = max(historical_dates)
    prediction_dates = [
        latest_date + datetime.timedelta(days=days) for days in prediction_days
    ]

    return predictor.predict_anatomy_changes(
        patient,
        historical_images,
        historical_structures,
        historical_dates,
        prediction_dates,
        method,
    )
