#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module kiểm tra và đánh giá mô hình dự đoán trong QuangTPS.

Module này cung cấp các công cụ để kiểm tra tính hợp lệ và đánh giá hiệu suất
của các mô hình dự đoán thay đổi giải phẫu và các mô hình khác trong hệ thống.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass
import datetime
import json
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from quangtps.core.exceptions import ValidationError
from quangtps.core.types import Patient, Image, Structure
from quangtps.segmentation.contour.dice import calculate_dice_coefficient
from quangtps.common.timer import Timer

logger = logging.getLogger(__name__)


class ValidationMetric(Enum):
    """Các metric kiểm tra mô hình."""

    MSE = auto()  # Mean Squared Error
    MAE = auto()  # Mean Absolute Error
    RMSE = auto()  # Root Mean Squared Error
    R2 = auto()  # R-squared (coefficient of determination)
    EXPLAINED_VARIANCE = auto()  # Explained variance
    CORRELATION = auto()  # Pearson correlation
    DICE = auto()  # Dice coefficient
    JACCARD = auto()  # Jaccard index
    HAUSDORFF = auto()  # Hausdorff distance
    SSIM = auto()  # Structural Similarity Index
    VOLUME_DIFF = auto()  # Volume difference (%)
    CENTROID_DIST = auto()  # Centroid distance (mm)


@dataclass
class ValidationResult:
    """Kết quả của quá trình kiểm tra mô hình."""

    is_valid: bool = False
    confidence: float = 0.0
    message: str = ""
    metrics: Dict[str, float] = None
    timestamp: datetime.datetime = None

    def __post_init__(self):
        """Khởi tạo sau khi tạo instance."""
        if self.metrics is None:
            self.metrics = {}
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()

    def set_valid(self, is_valid: bool, confidence: float):
        """
        Thiết lập trạng thái hợp lệ và độ tin cậy.

        Parameters
        ----------
        is_valid : bool
            Trạng thái hợp lệ
        confidence : float
            Độ tin cậy từ 0 đến 1
        """
        self.is_valid = is_valid
        self.confidence = max(0.0, min(1.0, confidence))

    def set_message(self, message: str):
        """
        Thiết lập thông báo.

        Parameters
        ----------
        message : str
            Thông báo
        """
        self.message = message

    def add_metric(self, name: str, value: float):
        """
        Thêm metric đánh giá.

        Parameters
        ----------
        name : str
            Tên metric
        value : float
            Giá trị metric
        """
        self.metrics[name] = value

    def get_summary(self) -> Dict[str, Any]:
        """
        Lấy tóm tắt kết quả kiểm tra.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin tóm tắt
        """
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "message": self.message,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """
        Chuyển đổi kết quả sang chuỗi JSON.

        Returns
        -------
        str
            Chuỗi JSON chứa kết quả kiểm tra
        """
        summary = self.get_summary()
        return json.dumps(summary, ensure_ascii=False, indent=2)


class ModelValidator:
    """
    Lớp để kiểm tra hiệu suất và tính hợp lệ của các mô hình dự đoán.
    """

    def __init__(self):
        """Khởi tạo validator."""
        self.default_metrics = [ValidationMetric.MSE, ValidationMetric.SSIM]
        self.thresholds = {
            ValidationMetric.MSE.value: 0.05,
            ValidationMetric.MAE.value: 0.1,
            ValidationMetric.DICE.value: 0.8,
            ValidationMetric.JACCARD.value: 0.7,
            ValidationMetric.HAUSDORFF.value: 10.0,
            ValidationMetric.SSIM.value: 0.85,
            ValidationMetric.CORRELATION.value: 0.9,
        }
        self.metric_weights = {
            ValidationMetric.MSE.value: 1.0,
            ValidationMetric.MAE.value: 1.0,
            ValidationMetric.DICE.value: 2.0,
            ValidationMetric.JACCARD.value: 1.5,
            ValidationMetric.HAUSDORFF.value: 1.5,
            ValidationMetric.SSIM.value: 2.0,
            ValidationMetric.CORRELATION.value: 1.0,
        }
        self.validation_history = []

    def set_threshold(self, metric: Union[ValidationMetric, str], threshold: float):
        """
        Thiết lập ngưỡng cho một metric cụ thể.

        Parameters
        ----------
        metric : Union[ValidationMetric, str]
            Metric cần thiết lập ngưỡng.
        threshold : float
            Giá trị ngưỡng mới.
        """
        if isinstance(metric, ValidationMetric):
            metric = metric.value

        self.thresholds[metric] = threshold
        logger.debug(f"Đã thiết lập ngưỡng {threshold} cho metric {metric}")

    def set_metric_weight(self, metric: Union[ValidationMetric, str], weight: float):
        """
        Thiết lập trọng số cho một metric cụ thể.

        Parameters
        ----------
        metric : Union[ValidationMetric, str]
            Metric cần thiết lập trọng số.
        weight : float
            Giá trị trọng số mới.
        """
        if isinstance(metric, ValidationMetric):
            metric = metric.value

        self.metric_weights[metric] = weight
        logger.debug(f"Đã thiết lập trọng số {weight} cho metric {metric}")

    def set_default_metrics(self, metrics: List[ValidationMetric]):
        """
        Thiết lập danh sách metric mặc định.

        Parameters
        ----------
        metrics : List[ValidationMetric]
            Danh sách metric mặc định mới.
        """
        self.default_metrics = metrics
        logger.debug(f"Đã thiết lập metrics mặc định: {[m.name for m in metrics]}")

    def calculate_metric(
        self, metric: ValidationMetric, predicted: np.ndarray, reference: np.ndarray
    ) -> float:
        """
        Tính toán một metric cụ thể.

        Parameters
        ----------
        metric : ValidationMetric
            Metric cần tính.
        predicted : np.ndarray
            Dữ liệu dự đoán.
        reference : np.ndarray
            Dữ liệu tham chiếu.

        Returns
        -------
        float
            Giá trị metric.
        """
        try:
            # Chuẩn hóa dữ liệu về cùng kích thước
            if predicted.shape != reference.shape:
                logger.warning(
                    f"Kích thước khác nhau: predicted={predicted.shape}, reference={reference.shape}. Metric có thể không chính xác."
                )

            # Tính toán metric
            if metric == ValidationMetric.MSE:
                return mean_squared_error(reference.flatten(), predicted.flatten())
            elif metric == ValidationMetric.MAE:
                return mean_absolute_error(reference.flatten(), predicted.flatten())
            elif metric == ValidationMetric.RMSE:
                return np.sqrt(
                    mean_squared_error(reference.flatten(), predicted.flatten())
                )
            elif metric == ValidationMetric.R2:
                return r2_score(reference.flatten(), predicted.flatten())
            elif metric == ValidationMetric.EXPLAINED_VARIANCE:
                return explained_variance_score(
                    reference.flatten(), predicted.flatten()
                )
            elif metric == ValidationMetric.CORRELATION:
                corr, _ = pearsonr(reference.flatten(), predicted.flatten())
                return corr
            elif metric == ValidationMetric.DICE:
                return calculate_dice_coefficient(predicted, reference)
            elif metric == ValidationMetric.JACCARD:
                intersection = np.logical_and(predicted, reference)
                union = np.logical_or(predicted, reference)
                return np.sum(intersection) / np.sum(union)
            elif metric == ValidationMetric.SSIM:
                return ssim(predicted, reference)
            elif metric == ValidationMetric.VOLUME_DIFF:
                vol_pred = np.sum(predicted)
                vol_ref = np.sum(reference)
                return abs(vol_pred - vol_ref) / vol_ref * 100
            elif metric == ValidationMetric.CENTROID_DIST:
                # Tính tâm
                pred_coords = np.argwhere(predicted)
                ref_coords = np.argwhere(reference)
                if len(pred_coords) == 0 or len(ref_coords) == 0:
                    return float("inf")
                pred_centroid = np.mean(pred_coords, axis=0)
                ref_centroid = np.mean(ref_coords, axis=0)
                return np.linalg.norm(pred_centroid - ref_centroid)
            else:
                logger.error(f"Metric không được hỗ trợ: {metric}")
                return float("nan")

        except Exception as e:
            logger.error(f"Lỗi khi tính metric {metric}: {str(e)}")
            return float("nan")

    def validate_prediction(
        self, predicted, reference=None, metrics=None
    ) -> Tuple[bool, float]:
        """
        Kiểm tra tính hợp lệ của một dự đoán.

        Parameters
        ----------
        predicted : array_like
            Dữ liệu dự đoán.
        reference : array_like, optional
            Dữ liệu tham chiếu (nếu có).
        metrics : List[ValidationMetric], optional
            Danh sách các metric cần sử dụng.

        Returns
        -------
        Tuple[bool, float]
            Tuple chứa kết quả (hợp lệ hay không) và độ tin cậy.
        """
        # Tạo kết quả kiểm tra
        result = ValidationResult()

        # Sử dụng metrics mặc định nếu không cung cấp
        if metrics is None:
            metrics = self.default_metrics

        try:
            # Nếu không có tham chiếu, chỉ thực hiện các kiểm tra cơ bản
            if reference is None:
                # Kiểm tra giá trị NaN và Inf
                has_nan = np.any(np.isnan(predicted))
                has_inf = np.any(np.isinf(predicted))

                if has_nan or has_inf:
                    result.set_valid(False, 0.0)
                    result.set_message(
                        f"Dự đoán chứa giá trị không hợp lệ: NaN={has_nan}, Inf={has_inf}"
                    )
                else:
                    # Kiểm tra phạm vi giá trị hợp lý
                    min_val = np.min(predicted)
                    max_val = np.max(predicted)
                    is_reasonable_range = (
                        -1000 <= min_val and max_val <= 5000
                    )  # Giới hạn giả định cho dữ liệu Y tế

                    result.add_metric("min_value", min_val)
                    result.add_metric("max_value", max_val)

                    if is_reasonable_range:
                        result.set_valid(True, 0.8)
                        result.set_message("Dự đoán có phạm vi giá trị hợp lý")
                    else:
                        result.set_valid(False, 0.5)
                        result.set_message(
                            f"Phạm vi giá trị bất thường: [{min_val}, {max_val}]"
                        )
            else:
                # Đánh giá so với tham chiếu
                confidence_scores = []

                for metric in metrics:
                    metric_name = (
                        metric.name if hasattr(metric, "name") else str(metric)
                    )
                    metric_value = self.calculate_metric(metric, predicted, reference)
                    result.add_metric(metric_name, metric_value)

                    # So sánh với ngưỡng để xác định tính hợp lệ
                    threshold = self.thresholds.get(metric.value, None)
                    weight = self.metric_weights.get(metric.value, 1.0)

                    if threshold is not None:
                        # Tính độ tin cậy dựa trên sự khác biệt với ngưỡng
                        if metric in [
                            ValidationMetric.MSE,
                            ValidationMetric.MAE,
                            ValidationMetric.RMSE,
                            ValidationMetric.HAUSDORFF,
                            ValidationMetric.VOLUME_DIFF,
                            ValidationMetric.CENTROID_DIST,
                        ]:
                            # Các metric mà giá trị càng thấp càng tốt
                            confidence = max(0, 1 - (metric_value / threshold))
                        else:
                            # Các metric mà giá trị càng cao càng tốt
                            confidence = max(0, metric_value / threshold)

                        confidence = min(1.0, confidence)
                        confidence_scores.append(confidence * weight)

                # Tính điểm tin cậy trung bình
                if confidence_scores:
                    avg_confidence = sum(confidence_scores) / sum(
                        self.metric_weights.values()
                    )
                    result.set_valid(avg_confidence >= 0.7, avg_confidence)

                    if result.is_valid:
                        result.set_message("Dự đoán đạt tiêu chí kiểm tra")
                    else:
                        result.set_message(
                            f"Dự đoán không đạt tiêu chí kiểm tra, độ tin cậy = {avg_confidence:.4f}"
                        )
                else:
                    result.set_valid(False, 0.0)
                    result.set_message("Không thể tính các metric kiểm tra")

            # Lưu kết quả vào lịch sử
            self.validation_history.append(result)

            # Trả về kết quả chính
            return result.is_valid, result.confidence

        except Exception as e:
            logger.error(f"Lỗi trong quá trình kiểm tra: {str(e)}")
            result.set_valid(False, 0.0)
            result.set_message(f"Lỗi kiểm tra: {str(e)}")
            self.validation_history.append(result)
            return False, 0.0

    def validate_predictions(
        self, predictions, ground_truth=None, metrics=None
    ) -> Dict[str, Any]:
        """
        Đánh giá nhiều dự đoán cùng lúc.

        Parameters
        ----------
        predictions : Dict hoặc List
            Dự đoán cần đánh giá. Có thể là danh sách các dự đoán hoặc từ điển
            ánh xạ id dự đoán (ví dụ: thời điểm dự đoán) với dữ liệu dự đoán.
        ground_truth : Dict hoặc List, optional
            Dữ liệu thực tế tương ứng để so sánh. Nếu None, chỉ thực hiện
            kiểm tra tính hợp lệ nội tại của dự đoán.
        metrics : List[ValidationMetric], optional
            Danh sách các metric dùng để đánh giá. Nếu None, sẽ sử dụng
            metric mặc định của validator.

        Returns
        -------
        Dict[str, Any]
            Kết quả đánh giá, bao gồm kết quả cho từng dự đoán và thống kê tổng hợp.
        """
        results = {}
        overall_metrics = {}
        valid_count = 0
        total_count = 0

        try:
            # Chuyển đổi dự đoán thành từ điển nếu là danh sách
            pred_dict = predictions
            if isinstance(predictions, list):
                pred_dict = {i: pred for i, pred in enumerate(predictions)}

            # Chuyển đổi ground_truth thành từ điển nếu là danh sách
            gt_dict = ground_truth
            if ground_truth is not None and isinstance(ground_truth, list):
                gt_dict = {i: gt for i, gt in enumerate(ground_truth)}

            # Đánh giá từng dự đoán
            for pred_id, prediction in pred_dict.items():
                total_count += 1

                # Lấy ground truth tương ứng nếu có
                reference = None
                if gt_dict and pred_id in gt_dict:
                    reference = gt_dict[pred_id]

                # Thực hiện đánh giá
                is_valid, confidence = self.validate_prediction(
                    prediction, reference, metrics
                )

                # Tạo kết quả đánh giá
                validation_result = ValidationResult()
                validation_result.set_valid(is_valid, confidence)

                # Tính toán các metric cụ thể nếu có reference
                if reference is not None:
                    for metric in metrics or self.default_metrics:
                        try:
                            metric_value = self.calculate_metric(
                                metric, prediction, reference
                            )
                            validation_result.add_metric(
                                metric.name if hasattr(metric, "name") else str(metric),
                                metric_value,
                            )

                            # Thêm vào overall metrics
                            if metric not in overall_metrics:
                                overall_metrics[str(metric)] = []
                            overall_metrics[str(metric)].append(metric_value)
                        except Exception as e:
                            logger.error(
                                f"Lỗi khi tính metric {metric} cho dự đoán {pred_id}: {str(e)}"
                            )

                # Thêm vào kết quả
                results[str(pred_id)] = validation_result.get_summary()
                if is_valid:
                    valid_count += 1

            # Tính toán thống kê tổng hợp
            summary = {
                "total_count": total_count,
                "valid_count": valid_count,
                "valid_percentage": (valid_count / total_count * 100)
                if total_count > 0
                else 0,
            }

            # Thêm thống kê metric
            for metric_name, values in overall_metrics.items():
                if values:
                    summary[f"{metric_name}_mean"] = np.mean(values)
                    summary[f"{metric_name}_std"] = np.std(values)
                    summary[f"{metric_name}_min"] = np.min(values)
                    summary[f"{metric_name}_max"] = np.max(values)

            # Thêm thống kê tổng hợp vào kết quả
            results["summary"] = summary

            return results

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá nhiều dự đoán: {str(e)}")
            return {
                "error": str(e),
                "total_count": total_count,
                "valid_count": valid_count,
            }

    def get_last_validation_result(self) -> Optional[ValidationResult]:
        """
        Lấy kết quả kiểm tra gần nhất.

        Returns
        -------
        Optional[ValidationResult]
            Kết quả kiểm tra gần nhất hoặc None nếu không có.
        """
        if self.validation_history:
            return self.validation_history[-1]
        return None

    def get_validation_history(self) -> List[ValidationResult]:
        """
        Lấy lịch sử kiểm tra.

        Returns
        -------
        List[ValidationResult]
            Danh sách các kết quả kiểm tra.
        """
        return self.validation_history

    def plot_validation_history(self, metrics: List[str] = None, figsize=(10, 6)):
        """
        Vẽ biểu đồ lịch sử kiểm tra.

        Parameters
        ----------
        metrics : List[str], optional
            Danh sách các metric cần vẽ. Nếu None, vẽ tất cả các metric có sẵn.
        figsize : tuple, optional
            Kích thước hình vẽ.
        """
        if not self.validation_history:
            logger.warning("Không có lịch sử kiểm tra để vẽ")
            return

        # Tạo DataFrame từ lịch sử kiểm tra
        data = []
        for result in self.validation_history:
            row = {"timestamp": result.timestamp, "confidence": result.confidence}
            row.update(result.metrics)
            data.append(row)

        df = pd.DataFrame(data)

        # Nếu không chỉ định metrics, sử dụng tất cả các metric có sẵn
        if metrics is None:
            # Loại bỏ các cột không phải metric
            metrics = [
                col for col in df.columns if col not in ["timestamp", "confidence"]
            ]

        # Vẽ biểu đồ
        plt.figure(figsize=figsize)

        # Vẽ đường độ tin cậy
        plt.plot(df["timestamp"], df["confidence"], "k-", label="Độ tin cậy")

        # Vẽ các metric
        for metric in metrics:
            if metric in df.columns:
                # Chuẩn hóa giá trị metric để hiển thị trên cùng đồ thị với độ tin cậy
                values = df[metric].values
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                    plt.plot(
                        df["timestamp"], normalized, "--", label=f"{metric} (chuẩn hóa)"
                    )
                else:
                    logger.warning(
                        f"Metric {metric} có giá trị không đổi, không thể chuẩn hóa"
                    )

        plt.xlabel("Thời gian")
        plt.ylabel("Giá trị (chuẩn hóa)")
        plt.title("Lịch sử kiểm tra mô hình")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    def save_results(self, filepath: str) -> bool:
        """
        Lưu kết quả kiểm tra vào file.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến file lưu kết quả.

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không.
        """
        try:
            results = []
            for result in self.validation_history:
                results.append(result.get_summary())

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Đã lưu kết quả kiểm tra vào {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả kiểm tra: {str(e)}")
            return False

    def validate_structure_predictions(
        self,
        predicted_structures: Dict[str, Structure],
        reference_structures: Dict[str, Structure],
        metrics: List[ValidationMetric] = None,
    ) -> Dict[str, Any]:
        """
        Đánh giá dự đoán cấu trúc.

        Parameters
        ----------
        predicted_structures : Dict[str, Structure]
            Từ điển các cấu trúc dự đoán, với khóa là tên cấu trúc.
        reference_structures : Dict[str, Structure]
            Từ điển các cấu trúc tham chiếu, với khóa là tên cấu trúc.
        metrics : List[ValidationMetric], optional
            Danh sách các metric cần sử dụng, mặc định sử dụng DICE và Hausdorff.

        Returns
        -------
        Dict[str, Any]
            Kết quả đánh giá, bao gồm kết quả cho từng cấu trúc và thống kê tổng hợp.
        """
        if metrics is None:
            metrics = [
                ValidationMetric.DICE,
                ValidationMetric.HAUSDORFF,
                ValidationMetric.VOLUME_DIFF,
            ]

        results = {}
        structure_results = {}

        # Tìm các cấu trúc xuất hiện trong cả hai tập
        common_structures = set(predicted_structures.keys()) & set(
            reference_structures.keys()
        )

        for struct_name in common_structures:
            pred_struct = predicted_structures[struct_name]
            ref_struct = reference_structures[struct_name]

            # Lấy mask của cấu trúc
            pred_mask = pred_struct.mask
            ref_mask = ref_struct.mask

            # Đánh giá cho cấu trúc này
            struct_result = ValidationResult()
            confidence_scores = []

            for metric in metrics:
                try:
                    metric_value = self.calculate_metric(metric, pred_mask, ref_mask)
                    struct_result.add_metric(metric.name, metric_value)

                    # Tính độ tin cậy dựa trên ngưỡng
                    threshold = self.thresholds.get(metric.value, None)
                    weight = self.metric_weights.get(metric.value, 1.0)

                    if threshold is not None:
                        if metric in [
                            ValidationMetric.MSE,
                            ValidationMetric.MAE,
                            ValidationMetric.RMSE,
                            ValidationMetric.HAUSDORFF,
                            ValidationMetric.VOLUME_DIFF,
                            ValidationMetric.CENTROID_DIST,
                        ]:
                            # Các metric mà giá trị càng thấp càng tốt
                            confidence = max(0, 1 - (metric_value / threshold))
                        else:
                            # Các metric mà giá trị càng cao càng tốt
                            confidence = max(0, metric_value / threshold)

                        confidence = min(1.0, confidence)
                        confidence_scores.append(confidence * weight)

                except Exception as e:
                    logger.error(
                        f"Lỗi khi tính metric {metric} cho cấu trúc {struct_name}: {str(e)}"
                    )

            # Tính độ tin cậy trung bình
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / sum(
                    weight for weight in self.metric_weights.values() if weight > 0
                )
                struct_result.set_valid(avg_confidence >= 0.7, avg_confidence)

                if struct_result.is_valid:
                    struct_result.set_message(
                        f"Cấu trúc {struct_name} đạt tiêu chí kiểm tra"
                    )
                else:
                    struct_result.set_message(
                        f"Cấu trúc {struct_name} không đạt tiêu chí kiểm tra"
                    )
            else:
                struct_result.set_valid(False, 0.0)
                struct_result.set_message(
                    f"Không thể tính các metric cho cấu trúc {struct_name}"
                )

            # Lưu kết quả
            structure_results[struct_name] = struct_result.get_summary()

        # Tính toán thống kê tổng hợp
        valid_structures = sum(
            1 for result in structure_results.values() if result["is_valid"]
        )
        avg_confidence = np.mean(
            [result["confidence"] for result in structure_results.values()]
        )

        # Tạo kết quả tổng hợp
        results = {
            "structures": structure_results,
            "summary": {
                "total_structures": len(common_structures),
                "valid_structures": valid_structures,
                "valid_percentage": (valid_structures / len(common_structures)) * 100
                if common_structures
                else 0,
                "average_confidence": avg_confidence,
                "missing_in_prediction": list(
                    set(reference_structures.keys()) - set(predicted_structures.keys())
                ),
                "extra_in_prediction": list(
                    set(predicted_structures.keys()) - set(reference_structures.keys())
                ),
            },
        }

        return results
