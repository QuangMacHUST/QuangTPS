#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module kiểm tra và đánh giá độ chính xác của các mô hình dự đoán thay đổi giải phẫu.

Module này cung cấp các công cụ để đánh giá hiệu suất của các mô hình dự đoán
thay đổi giải phẫu, so sánh với dữ liệu thực tế thu thập được trong quá trình điều trị.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from quangtps.adaptive.prediction.anatomy_prediction import (
    AnatomyPredictor,
    AnatomyPrediction,
    PredictionMethod,
)
from quangtps.adaptive.prediction.deformable_anatomy_predictor import (
    DeformableAnatomyPredictor,
)
from quangtps.core.types import Patient, Image, Structure, Dose
from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class PredictionMetrics:
    """Lớp chứa các kết quả đánh giá độ chính xác của dự đoán."""

    def __init__(self):
        self.mae = {}  # Mean Absolute Error
        self.rmse = {}  # Root Mean Squared Error
        self.r2 = {}  # R-squared
        self.volume_errors = {}  # % sai lệch thể tích
        self.position_errors = {}  # Sai lệch vị trí (mm)
        self.structure_metrics = {}  # Các chỉ số cho từng cấu trúc
        self.dice_coefficients = {}  # Hệ số Dice cho từng cấu trúc
        self.hausdorff_distances = {}  # Khoảng cách Hausdorff (mm)

    def add_volume_error(self, structure_id: str, predicted: float, actual: float):
        """Thêm sai lệch thể tích cho một cấu trúc cụ thể."""
        if structure_id not in self.volume_errors:
            self.volume_errors[structure_id] = []

        error_percent = (
            100.0 * abs(predicted - actual) / (actual if actual > 0 else 1.0)
        )
        self.volume_errors[structure_id].append(error_percent)

    def add_position_error(self, structure_id: str, error_mm: float):
        """Thêm sai lệch vị trí (mm) cho một cấu trúc cụ thể."""
        if structure_id not in self.position_errors:
            self.position_errors[structure_id] = []

        self.position_errors[structure_id].append(error_mm)

    def add_dice_coefficient(self, structure_id: str, dice: float):
        """Thêm hệ số Dice cho một cấu trúc cụ thể."""
        if structure_id not in self.dice_coefficients:
            self.dice_coefficients[structure_id] = []

        self.dice_coefficients[structure_id].append(dice)

    def add_hausdorff_distance(self, structure_id: str, distance: float):
        """Thêm khoảng cách Hausdorff cho một cấu trúc cụ thể."""
        if structure_id not in self.hausdorff_distances:
            self.hausdorff_distances[structure_id] = []

        self.hausdorff_distances[structure_id].append(distance)

    def calculate_average_metrics(self):
        """Tính toán các chỉ số trung bình cho tất cả các cấu trúc."""
        # Tính trung bình các chỉ số thể tích
        for structure_id, errors in self.volume_errors.items():
            if structure_id not in self.structure_metrics:
                self.structure_metrics[structure_id] = {}
            self.structure_metrics[structure_id]["avg_volume_error"] = np.mean(errors)

        # Tính trung bình các chỉ số vị trí
        for structure_id, errors in self.position_errors.items():
            if structure_id not in self.structure_metrics:
                self.structure_metrics[structure_id] = {}
            self.structure_metrics[structure_id]["avg_position_error"] = np.mean(errors)

        # Tính trung bình các hệ số Dice
        for structure_id, dices in self.dice_coefficients.items():
            if structure_id not in self.structure_metrics:
                self.structure_metrics[structure_id] = {}
            self.structure_metrics[structure_id]["avg_dice"] = np.mean(dices)

        # Tính trung bình các khoảng cách Hausdorff
        for structure_id, distances in self.hausdorff_distances.items():
            if structure_id not in self.structure_metrics:
                self.structure_metrics[structure_id] = {}
            self.structure_metrics[structure_id]["avg_hausdorff"] = np.mean(distances)

        # Tính các chỉ số tổng hợp cho tất cả các cấu trúc
        all_volume_errors = []
        for errors in self.volume_errors.values():
            all_volume_errors.extend(errors)

        all_position_errors = []
        for errors in self.position_errors.values():
            all_position_errors.extend(errors)

        all_dice_coefficients = []
        for dices in self.dice_coefficients.values():
            all_dice_coefficients.extend(dices)

        self.mae["volume"] = np.mean(all_volume_errors) if all_volume_errors else 0.0
        self.mae["position"] = (
            np.mean(all_position_errors) if all_position_errors else 0.0
        )
        self.mae["dice"] = (
            1.0 - np.mean(all_dice_coefficients) if all_dice_coefficients else 1.0
        )

        self.rmse["volume"] = (
            np.sqrt(np.mean(np.square(all_volume_errors))) if all_volume_errors else 0.0
        )
        self.rmse["position"] = (
            np.sqrt(np.mean(np.square(all_position_errors)))
            if all_position_errors
            else 0.0
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển đổi kết quả đánh giá thành DataFrame."""
        data = []

        for structure_id, metrics in self.structure_metrics.items():
            row = {"structure_id": structure_id}
            row.update(metrics)
            data.append(row)

        return pd.DataFrame(data)

    def plot_results(self, output_dir: Optional[str] = None):
        """Tạo các biểu đồ kết quả đánh giá."""
        if not self.structure_metrics:
            logger.warning("Không có dữ liệu để vẽ biểu đồ")
            return

        # Tạo thư mục đầu ra nếu chưa tồn tại
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Vẽ biểu đồ sai lệch thể tích
        plt.figure(figsize=(12, 8))
        structs = list(self.structure_metrics.keys())
        volume_errors = [
            self.structure_metrics[s].get("avg_volume_error", 0) for s in structs
        ]

        plt.bar(structs, volume_errors)
        plt.title("Sai lệch thể tích trung bình theo cấu trúc")
        plt.xlabel("Cấu trúc")
        plt.ylabel("Sai lệch (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, "volume_errors.png"))
            plt.close()
        else:
            plt.show()

        # Vẽ biểu đồ hệ số Dice
        plt.figure(figsize=(12, 8))
        dice_values = [self.structure_metrics[s].get("avg_dice", 0) for s in structs]

        plt.bar(structs, dice_values)
        plt.title("Hệ số Dice trung bình theo cấu trúc")
        plt.xlabel("Cấu trúc")
        plt.ylabel("Hệ số Dice")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, "dice_coefficients.png"))
            plt.close()
        else:
            plt.show()


class ModelValidator:
    """Lớp kiểm tra và đánh giá mô hình dự đoán thay đổi giải phẫu."""

    def __init__(self):
        self.metrics = PredictionMetrics()

    def validate_predictions(
        self,
        prediction: AnatomyPrediction,
        actual_images: List[Image],
        actual_structures: List[Dict[str, Structure]],
        actual_dates: List[datetime.datetime],
    ) -> PredictionMetrics:
        """
        Kiểm tra độ chính xác của các dự đoán thay đổi giải phẫu.

        Parameters
        ----------
        prediction : AnatomyPrediction
            Kết quả dự đoán thay đổi giải phẫu
        actual_images : List[Image]
            Danh sách hình ảnh thực tế
        actual_structures : List[Dict[str, Structure]]
            Danh sách cấu trúc thực tế tương ứng với mỗi hình ảnh
        actual_dates : List[datetime.datetime]
            Danh sách ngày tương ứng với mỗi hình ảnh/cấu trúc

        Returns
        -------
        PredictionMetrics
            Các chỉ số đánh giá độ chính xác
        """
        logger.info("Bắt đầu đánh giá độ chính xác mô hình dự đoán thay đổi giải phẫu")

        # Khởi tạo đối tượng kết quả
        self.metrics = PredictionMetrics()

        # Kiểm tra tất cả các mốc thời gian dự đoán
        for timepoint in prediction.prediction_timeline:
            pred_date = timepoint["date"]

            # Tìm ngày thực tế gần nhất
            closest_idx = self._find_closest_date_index(pred_date, actual_dates)

            if closest_idx is None:
                logger.warning(
                    f"Không tìm thấy dữ liệu thực tế gần với ngày dự đoán {pred_date}"
                )
                continue

            # Lấy dữ liệu thực tế tương ứng
            actual_date = actual_dates[closest_idx]
            actual_struct_dict = actual_structures[closest_idx]

            # So sánh dữ liệu thực tế với dự đoán
            if pred_date in prediction.predicted_structures:
                pred_struct_dict = prediction.predicted_structures[pred_date]

                for struct_id, pred_struct in pred_struct_dict.items():
                    if struct_id in actual_struct_dict:
                        actual_struct = actual_struct_dict[struct_id]

                        # So sánh thể tích
                        pred_volume = pred_struct.get_volume()
                        actual_volume = actual_struct.get_volume()
                        self.metrics.add_volume_error(
                            struct_id, pred_volume, actual_volume
                        )

                        # So sánh vị trí (trọng tâm)
                        pred_center = pred_struct.get_center()
                        actual_center = actual_struct.get_center()
                        if pred_center is not None and actual_center is not None:
                            distance = np.linalg.norm(
                                np.array(pred_center) - np.array(actual_center)
                            )
                            self.metrics.add_position_error(struct_id, distance)

                        # Tính hệ số Dice
                        dice = self._calculate_dice(pred_struct, actual_struct)
                        self.metrics.add_dice_coefficient(struct_id, dice)

                        # Tính khoảng cách Hausdorff
                        hausdorff = self._calculate_hausdorff(
                            pred_struct, actual_struct
                        )
                        self.metrics.add_hausdorff_distance(struct_id, hausdorff)

        # Tính toán các chỉ số trung bình
        self.metrics.calculate_average_metrics()

        logger.info("Hoàn thành đánh giá độ chính xác mô hình dự đoán")
        return self.metrics

    def _find_closest_date_index(
        self, target_date: datetime.datetime, date_list: List[datetime.datetime]
    ) -> Optional[int]:
        """Tìm chỉ số của ngày gần nhất với ngày mục tiêu."""
        if not date_list:
            return None

        closest_idx = 0
        min_diff = abs((target_date - date_list[0]).total_seconds())

        for i, date in enumerate(date_list[1:], 1):
            diff = abs((target_date - date).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        # Nếu chênh lệch quá lớn (> 7 ngày), trả về None
        if min_diff > 7 * 24 * 3600:
            return None

        return closest_idx

    def _calculate_dice(self, struct1: Structure, struct2: Structure) -> float:
        """Tính hệ số Dice giữa hai cấu trúc."""
        try:
            # Lấy các mặt nạ nhị phân
            mask1 = struct1.get_binary_mask()
            mask2 = struct2.get_binary_mask()

            if mask1 is None or mask2 is None:
                return 0.0

            # Tính toán hệ số Dice
            intersection = np.sum(np.logical_and(mask1, mask2))
            dice = (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2))

            return float(dice)
        except Exception as e:
            logger.error(f"Lỗi khi tính hệ số Dice: {str(e)}")
            return 0.0

    def _calculate_hausdorff(self, struct1: Structure, struct2: Structure) -> float:
        """Tính khoảng cách Hausdorff giữa hai cấu trúc."""
        try:
            # Lấy các điểm bề mặt
            surface1 = struct1.get_surface_points()
            surface2 = struct2.get_surface_points()

            if not surface1 or not surface2:
                return 100.0  # Giá trị lớn nếu không thể tính toán

            # Tính toán khoảng cách Hausdorff
            max_distance = 0.0

            for point1 in surface1:
                min_dist_to_surface2 = min(
                    np.linalg.norm(np.array(point1) - np.array(point2))
                    for point2 in surface2
                )
                max_distance = max(max_distance, min_dist_to_surface2)

            for point2 in surface2:
                min_dist_to_surface1 = min(
                    np.linalg.norm(np.array(point2) - np.array(point1))
                    for point1 in surface1
                )
                max_distance = max(max_distance, min_dist_to_surface1)

            return float(max_distance)
        except Exception as e:
            logger.error(f"Lỗi khi tính khoảng cách Hausdorff: {str(e)}")
            return 100.0

    def plot_validation_results(self, output_dir: Optional[str] = None):
        """Tạo các biểu đồ kết quả đánh giá."""
        self.metrics.plot_results(output_dir)

    def generate_validation_report(self, output_path: str) -> bool:
        """Tạo báo cáo đánh giá độ chính xác mô hình dự đoán."""
        try:
            # Tạo thư mục chứa nếu chưa tồn tại
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Chuyển đổi kết quả thành DataFrame
            df = self.metrics.to_dataframe()

            # Ghi DataFrame ra file CSV
            df.to_csv(output_path, index=False)

            # Tạo các biểu đồ
            self.plot_validation_results(output_dir)

            logger.info(f"Đã tạo báo cáo đánh giá tại {output_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo đánh giá: {str(e)}")
            return False

    @staticmethod
    def compute_prediction_accuracy(
        predictions: List[float], actuals: List[float]
    ) -> Dict[str, float]:
        """Tính toán độ chính xác của dự đoán số."""
        if len(predictions) != len(actuals):
            raise ValidationError("Số lượng dự đoán và thực tế không khớp")

        if not predictions:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mean_error_pct": 0.0}

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        # Tính sai số phần trăm trung bình
        mean_error_pct = 0.0
        count = 0
        for a, p in zip(actuals, predictions):
            if a != 0:
                mean_error_pct += 100.0 * abs(p - a) / abs(a)
                count += 1

        if count > 0:
            mean_error_pct /= count

        return {"mae": mae, "rmse": rmse, "r2": r2, "mean_error_pct": mean_error_pct}


def validate_anatomy_prediction_model(
    predictor: AnatomyPredictor,
    reference_date: datetime.datetime,
    test_images: List[Image],
    test_structures: List[Dict[str, Structure]],
    test_dates: List[datetime.datetime],
    output_dir: Optional[str] = None,
) -> PredictionMetrics:
    """
    Xác thực mô hình dự đoán thay đổi giải phẫu.

    Parameters
    ----------
    predictor : AnatomyPredictor
        Bộ dự đoán cần đánh giá
    reference_date : datetime.datetime
        Ngày tham chiếu
    test_images : List[Image]
        Danh sách hình ảnh kiểm tra
    test_structures : List[Dict[str, Structure]]
        Danh sách cấu trúc kiểm tra tương ứng với mỗi hình ảnh
    test_dates : List[datetime.datetime]
        Danh sách ngày tương ứng với mỗi hình ảnh/cấu trúc
    output_dir : Optional[str], optional
        Thư mục đầu ra, mặc định là None

    Returns
    -------
    PredictionMetrics
        Các chỉ số đánh giá độ chính xác
    """
    if len(test_images) != len(test_structures) or len(test_images) != len(test_dates):
        raise ValidationError("Số lượng hình ảnh, cấu trúc và ngày kiểm tra không khớp")

    # Tạo dự đoán cho các ngày kiểm tra
    patient = test_images[0].patient if test_images else None

    # Lọc dữ liệu lịch sử
    historical_indices = [
        i for i, date in enumerate(test_dates) if date < reference_date
    ]

    if not historical_indices:
        raise ValidationError("Không có dữ liệu lịch sử để huấn luyện mô hình dự đoán")

    historical_images = [test_images[i] for i in historical_indices]
    historical_structures = [test_structures[i] for i in historical_indices]
    historical_dates = [test_dates[i] for i in historical_indices]

    # Lọc dữ liệu kiểm tra
    test_indices = [i for i, date in enumerate(test_dates) if date >= reference_date]

    if not test_indices:
        raise ValidationError("Không có dữ liệu kiểm tra để đánh giá mô hình dự đoán")

    future_images = [test_images[i] for i in test_indices]
    future_structures = [test_structures[i] for i in test_indices]
    future_dates = [test_dates[i] for i in test_indices]

    # Tạo dự đoán
    prediction = predictor.predict_anatomy_changes(
        patient=patient,
        historical_images=historical_images,
        historical_structures=historical_structures,
        historical_dates=historical_dates,
        prediction_dates=future_dates,
        method=PredictionMethod.SPLINE,
    )

    # Đánh giá độ chính xác
    validator = ModelValidator()
    metrics = validator.validate_predictions(
        prediction=prediction,
        actual_images=future_images,
        actual_structures=future_structures,
        actual_dates=future_dates,
    )

    # Tạo báo cáo nếu cần
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        report_path = os.path.join(output_dir, "validation_report.csv")
        validator.generate_validation_report(report_path)

    return metrics


if __name__ == "__main__":
    # Mã để chạy thử và kiểm tra module
    logger.info("Kiểm tra module model_validator.py")
