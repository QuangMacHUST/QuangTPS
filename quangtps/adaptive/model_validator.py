#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module kiểm tra hiệu suất và tính hợp lệ của các mô hình dự đoán.

Module này cung cấp các lớp và hàm để đánh giá hiệu suất và tính hợp lệ
của các mô hình dự đoán được sử dụng trong hệ thống lập kế hoạch thích ứng.
"""

import logging
import numpy as np
from enum import Enum
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class ValidationMetric(Enum):
    """Các metric được hỗ trợ để đánh giá mô hình."""

    MSE = "mean_squared_error"
    MAE = "mean_absolute_error"
    DICE = "dice_coefficient"
    JACCARD = "jaccard_index"
    HAUSDORFF = "hausdorff_distance"
    SSIM = "structural_similarity"
    CORRELATION = "correlation_coefficient"


class ValidationResult:
    """Kết quả của quá trình kiểm tra mô hình."""

    def __init__(self):
        """Khởi tạo đối tượng kết quả kiểm tra."""
        self.is_valid = False
        self.confidence = 0.0
        self.metrics = {}
        self.timestamp = datetime.now()
        self.message = ""

    def set_valid(self, is_valid: bool, confidence: float):
        """
        Thiết lập trạng thái hợp lệ và độ tin cậy.

        Parameters
        ----------
        is_valid : bool
            Biểu thị kết quả dự đoán có hợp lệ không.
        confidence : float
            Độ tin cậy của kết quả kiểm tra (0.0 - 1.0).
        """
        self.is_valid = is_valid
        self.confidence = confidence

    def add_metric(self, name: str, value: float):
        """
        Thêm một metric vào kết quả.

        Parameters
        ----------
        name : str
            Tên của metric.
        value : float
            Giá trị của metric.
        """
        self.metrics[name] = value

    def set_message(self, message: str):
        """
        Thiết lập thông báo kết quả.

        Parameters
        ----------
        message : str
            Thông báo mô tả kết quả kiểm tra.
        """
        self.message = message

    def get_summary(self) -> Dict[str, Any]:
        """
        Tạo tóm tắt kết quả kiểm tra.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin tóm tắt kết quả.
        """
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }


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
        Thiết lập danh sách các metric mặc định.

        Parameters
        ----------
        metrics : List[ValidationMetric]
            Danh sách các metric mặc định.
        """
        self.default_metrics = metrics
        logger.debug(f"Đã thiết lập {len(metrics)} metric mặc định")

    def calculate_metric(self, metric: ValidationMetric, predicted, reference) -> float:
        """
        Tính giá trị của một metric cụ thể.

        Parameters
        ----------
        metric : ValidationMetric
            Metric cần tính.
        predicted : array_like
            Dữ liệu dự đoán.
        reference : array_like
            Dữ liệu tham chiếu.

        Returns
        -------
        float
            Giá trị của metric.
        """
        try:
            # Chuyển đổi sang numpy array nếu cần
            if not isinstance(predicted, np.ndarray):
                predicted = np.array(predicted)
            if not isinstance(reference, np.ndarray):
                reference = np.array(reference)

            # Kiểm tra kích thước
            if predicted.shape != reference.shape:
                logger.warning(
                    f"Kích thước dự đoán {predicted.shape} và tham chiếu {reference.shape} không khớp nhau"
                )
                return float("nan")

            # Tính metric dựa trên loại
            if metric == ValidationMetric.MSE:
                return float(np.mean((predicted - reference) ** 2))

            elif metric == ValidationMetric.MAE:
                return float(np.mean(np.abs(predicted - reference)))

            elif metric == ValidationMetric.DICE:
                # Giả định dữ liệu là nhị phân (masks)
                intersection = np.sum(predicted * reference)
                union = np.sum(predicted) + np.sum(reference)
                if union == 0:
                    return 1.0  # Cả hai mask đều trống
                return float(2.0 * intersection / union)

            elif metric == ValidationMetric.JACCARD:
                # Giả định dữ liệu là nhị phân (masks)
                intersection = np.sum(predicted * reference)
                union = np.sum(predicted) + np.sum(reference) - intersection
                if union == 0:
                    return 1.0  # Cả hai mask đều trống
                return float(intersection / union)

            elif metric == ValidationMetric.HAUSDORFF:
                # Để triển khai hausdorff distance hoàn chỉnh, cần thư viện chuyên biệt
                # Đây là mô phỏng đơn giản
                return float(np.max(np.abs(predicted - reference)))

            elif metric == ValidationMetric.SSIM:
                try:
                    # Thử import skimage để tính SSIM
                    from skimage.metrics import structural_similarity as ssim

                    return float(
                        ssim(
                            reference,
                            predicted,
                            data_range=reference.max() - reference.min(),
                        )
                    )
                except ImportError:
                    # Fallback nếu không có skimage
                    mean_pred = np.mean(predicted)
                    mean_ref = np.mean(reference)
                    var_pred = np.var(predicted)
                    var_ref = np.var(reference)
                    cov = np.mean((predicted - mean_pred) * (reference - mean_ref))
                    c1 = (0.01 * np.max(reference)) ** 2
                    c2 = (0.03 * np.max(reference)) ** 2
                    return float(
                        (2 * mean_pred * mean_ref + c1)
                        * (2 * cov + c2)
                        / (
                            (mean_pred**2 + mean_ref**2 + c1)
                            * (var_pred + var_ref + c2)
                        )
                    )

            elif metric == ValidationMetric.CORRELATION:
                # Hệ số tương quan Pearson
                pred_flat = predicted.flatten()
                ref_flat = reference.flatten()
                return float(np.corrcoef(pred_flat, ref_flat)[0, 1])

            else:
                logger.error(f"Không hỗ trợ metric {metric}")
                return float("nan")

        except Exception as e:
            logger.error(f"Lỗi khi tính metric {metric.value}: {str(e)}")
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
                # Thực hiện đánh giá đầy đủ với dữ liệu tham chiếu
                confidence_scores = []

                for metric in metrics:
                    # Tính giá trị metric
                    value = self.calculate_metric(metric, predicted, reference)
                    result.add_metric(metric.value, value)

                    # Xác định xem metric có vượt qua ngưỡng không
                    threshold = self.thresholds.get(metric.value, 0.5)
                    weight = self.metric_weights.get(metric.value, 1.0)

                    # Ngược lại nếu metric càng nhỏ càng tốt
                    if metric in [
                        ValidationMetric.MSE,
                        ValidationMetric.MAE,
                        ValidationMetric.HAUSDORFF,
                    ]:
                        passes_threshold = value <= threshold
                        # Tính điểm tin cậy (1.0 khi bằng 0, 0.0 khi bằng hoặc lớn hơn 2*threshold)
                        confidence_score = (
                            max(0.0, 1.0 - value / (2.0 * threshold)) * weight
                        )
                    else:
                        passes_threshold = value >= threshold
                        # Tính điểm tin cậy (1.0 khi bằng 1.0, 0.0 khi bằng hoặc nhỏ hơn threshold/2)
                        confidence_score = (
                            max(0.0, (value - threshold / 2) / (1.0 - threshold / 2))
                            * weight
                        )

                    confidence_scores.append(confidence_score)

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

            # Thêm thống kê vào kết quả
            results["summary"] = summary

            return results

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá nhiều dự đoán: {str(e)}")
            return {
                "error": str(e),
                "total_count": total_count,
                "valid_count": valid_count,
            }

    def _calculate_position_error(self, predicted_mask, reference_mask):
        """
        Tính toán sai số vị trí giữa hai mask.

        Phương thức này tính toán khoảng cách giữa tâm (centroid) của hai mask
        đã cho để đánh giá sai số vị trí của cấu trúc dự đoán.

        Parameters
        ----------
        predicted_mask : np.ndarray
            Mask nhị phân của cấu trúc được dự đoán.
        reference_mask : np.ndarray
            Mask nhị phân của cấu trúc tham chiếu.

        Returns
        -------
        float
            Khoảng cách Euclidean giữa tâm của hai mask.
        """
        try:
            # Xác minh rằng cả hai mask đều có thông tin
            if predicted_mask is None or reference_mask is None:
                logger.warning("Một hoặc cả hai mask là None")
                return float("inf")

            if predicted_mask.sum() == 0 or reference_mask.sum() == 0:
                logger.warning("Một hoặc cả hai mask không có pixel nào")
                return float("inf")

            # Tính tâm của các mask
            pred_centroid = self._get_centroid(predicted_mask)
            ref_centroid = self._get_centroid(reference_mask)

            # Tính khoảng cách Euclidean
            distance = np.sqrt(
                np.sum((np.array(pred_centroid) - np.array(ref_centroid)) ** 2)
            )
            return float(distance)

        except Exception as e:
            logger.error(f"Lỗi khi tính toán sai số vị trí: {str(e)}")
            return float("inf")

    def _get_centroid(self, mask):
        """
        Tính tâm (centroid) của mask nhị phân.

        Parameters
        ----------
        mask : np.ndarray
            Mask nhị phân 3D.

        Returns
        -------
        tuple
            Tọa độ (x, y, z) của tâm.
        """
        if mask.sum() == 0:
            return (0, 0, 0)

        # Tìm tọa độ các điểm không phải 0
        indices = np.where(mask > 0)

        # Tính trung bình các tọa độ để có tâm
        centroid = []
        for i in range(len(indices)):
            centroid.append(float(indices[i].mean()))

        return tuple(centroid)


if __name__ == "__main__":
    # Mã để chạy thử và kiểm tra module
    logger.info("Kiểm tra module model_validator.py")
