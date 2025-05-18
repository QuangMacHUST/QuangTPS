#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module tích hợp giữa dự đoán thay đổi giải phẫu và lập kế hoạch thích ứng.

Module này cung cấp các lớp và hàm để liên kết các module dự đoán thay đổi giải phẫu
với hệ thống lập kế hoạch thích ứng, cho phép tự động cập nhật kế hoạch dựa trên
dự đoán thay đổi giải phẫu theo thời gian.
"""

import logging
import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from quangtps.adaptive.prediction.anatomy_prediction import (
    AnatomyPredictor,
    AnatomyPrediction,
)
from quangtps.adaptive.model_validator import ModelValidator, PredictionMetrics
from quangtps.core.exceptions import IntegrationError
from quangtps.core.types import Patient, Image, Structure

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Chiến lược thích ứng để ứng phó với thay đổi giải phẫu."""

    ISOCENTER_SHIFT = auto()  # Dịch chuyển isocenter để bù đắp
    FLUENCE_ADJUST = auto()  # Điều chỉnh fluence để bù đắp
    MLC_ADJUST = auto()  # Điều chỉnh MLC để bù đắp
    REPLAN = auto()  # Lập kế hoạch lại hoàn toàn
    HYBRID = auto()  # Kết hợp nhiều chiến lược
    NO_ACTION = auto()  # Không cần thích ứng


class PredictionConfidenceLevel(Enum):
    """Mức độ tin cậy của dự đoán thay đổi giải phẫu."""

    VERY_LOW = auto()  # Rất thấp (< 60%)
    LOW = auto()  # Thấp (60-75%)
    MEDIUM = auto()  # Trung bình (75-85%)
    HIGH = auto()  # Cao (85-95%)
    VERY_HIGH = auto()  # Rất cao (> 95%)

    @staticmethod
    def from_dice_value(dice: float) -> "PredictionConfidenceLevel":
        """
        Chuyển đổi hệ số Dice thành mức độ tin cậy.

        Parameters
        ----------
        dice : float
            Hệ số Dice (0.0 - 1.0)

        Returns
        -------
        PredictionConfidenceLevel
            Mức độ tin cậy tương ứng
        """
        if dice < 0.6:
            return PredictionConfidenceLevel.VERY_LOW
        elif dice < 0.75:
            return PredictionConfidenceLevel.LOW
        elif dice < 0.85:
            return PredictionConfidenceLevel.MEDIUM
        elif dice < 0.95:
            return PredictionConfidenceLevel.HIGH
        else:
            return PredictionConfidenceLevel.VERY_HIGH


class AnatomyPredictionIntegrator:
    """
    Lớp tích hợp giữa dự đoán thay đổi giải phẫu và lập kế hoạch thích ứng.

    Lớp này đóng vai trò trung gian giữa bộ dự đoán thay đổi giải phẫu và hệ thống
    lập kế hoạch thích ứng, đảm bảo dữ liệu dự đoán được xác thực và chuyển đổi
    đúng định dạng để sử dụng trong quá trình lập kế hoạch thích ứng.
    """

    def __init__(
        self,
        predictor: Optional[AnatomyPredictor] = None,
        validator: Optional[ModelValidator] = None,
    ):
        """
        Khởi tạo tích hợp giữa dự đoán thay đổi giải phẫu và lập kế hoạch thích ứng.

        Parameters
        ----------
        predictor : AnatomyPredictor, optional
            Đối tượng dự đoán thay đổi giải phẫu
        validator : ModelValidator, optional
            Đối tượng xác thực mô hình dự đoán
        """
        self.predictor = predictor
        self.validator = validator
        self.planner = None
        self.latest_prediction = None
        self.latest_metrics = None
        self.prediction_callbacks = []
        self.validation_callbacks = []
        self.auto_update = False

        # Logging khởi tạo
        logger.info("Khởi tạo AnatomyPredictionIntegrator")
        if predictor is None:
            logger.warning("Khởi tạo integrator mà không có predictor")
        if validator is None:
            logger.warning("Khởi tạo integrator mà không có validator")

    def set_predictor(self, predictor: AnatomyPredictor) -> None:
        """
        Thiết lập bộ dự đoán thay đổi giải phẫu.

        Parameters
        ----------
        predictor : AnatomyPredictor
            Đối tượng dự đoán thay đổi giải phẫu
        """
        self.predictor = predictor
        logger.info(f"Đã thiết lập predictor: {predictor.__class__.__name__}")

    def set_validator(self, validator: ModelValidator) -> None:
        """
        Thiết lập bộ xác thực mô hình dự đoán.

        Parameters
        ----------
        validator : ModelValidator
            Đối tượng xác thực mô hình dự đoán
        """
        self.validator = validator
        logger.info(f"Đã thiết lập validator: {validator.__class__.__name__}")

    def connect_planner(self, planner: Any) -> None:
        """
        Kết nối với hệ thống lập kế hoạch thích ứng.

        Parameters
        ----------
        planner : Any
            Đối tượng lập kế hoạch thích ứng
        """
        self.planner = planner

        # Đảm bảo planner có các phương thức cần thiết
        required_methods = ["update_with_prediction", "get_adaptation_options"]
        for method in required_methods:
            if not hasattr(self.planner, method):
                logger.warning(f"Planner không có phương thức {method}")

        logger.info(f"Đã kết nối với planner: {planner.__class__.__name__}")

    def register_prediction_callback(
        self, callback: Callable[[AnatomyPrediction], None]
    ) -> None:
        """
        Đăng ký callback được gọi sau khi có dự đoán mới.

        Parameters
        ----------
        callback : Callable[[AnatomyPrediction], None]
            Hàm callback nhận dự đoán làm đầu vào
        """
        self.prediction_callbacks.append(callback)

    def register_validation_callback(
        self, callback: Callable[[PredictionMetrics], None]
    ) -> None:
        """
        Đăng ký callback được gọi sau khi có kết quả đánh giá dự đoán mới.

        Parameters
        ----------
        callback : Callable[[PredictionMetrics], None]
            Hàm callback nhận kết quả đánh giá làm đầu vào
        """
        self.validation_callbacks.append(callback)

    def set_auto_update(self, enabled: bool = True) -> None:
        """
        Thiết lập chế độ tự động cập nhật kế hoạch dựa trên dự đoán mới.

        Parameters
        ----------
        enabled : bool, optional
            Bật/tắt chế độ tự động cập nhật, mặc định là True
        """
        self.auto_update = enabled
        logger.info(f"Đã {'bật' if enabled else 'tắt'} chế độ tự động cập nhật")

    def predict_and_update(
        self, date: Union[datetime.datetime, str], validate: bool = True
    ) -> Tuple[AnatomyPrediction, Optional[PredictionMetrics]]:
        """
        Dự đoán thay đổi giải phẫu và cập nhật kế hoạch thích ứng.

        Parameters
        ----------
        date : Union[datetime.datetime, str]
            Ngày cần dự đoán thay đổi giải phẫu
        validate : bool, optional
            Xác thực dự đoán trước khi cập nhật kế hoạch, mặc định là True

        Returns
        -------
        Tuple[AnatomyPrediction, Optional[PredictionMetrics]]
            Dự đoán thay đổi giải phẫu và kết quả xác thực (nếu có)

        Raises
        ------
        IntegrationError
            Nếu không có predictor hoặc planner
        """
        if self.predictor is None:
            raise IntegrationError("Không thể dự đoán: predictor chưa được thiết lập")

        if self.planner is None:
            logger.warning("Planner chưa được kết nối, sẽ không cập nhật kế hoạch")

        # Chuyển đổi ngày nếu cần
        if isinstance(date, str):
            try:
                date = datetime.datetime.fromisoformat(date)
            except ValueError:
                raise ValueError(f"Định dạng ngày không hợp lệ: {date}")

        # Dự đoán thay đổi giải phẫu
        try:
            prediction = self.predictor.predict_anatomy_at_date(date)
            self.latest_prediction = prediction

            # Thông báo cho các callback
            for callback in self.prediction_callbacks:
                try:
                    callback(prediction)
                except Exception as e:
                    logger.error(f"Lỗi khi gọi prediction callback: {str(e)}")

            logger.info(f"Đã dự đoán thay đổi giải phẫu cho ngày {date}")
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán thay đổi giải phẫu: {str(e)}")
            raise IntegrationError(f"Lỗi khi dự đoán thay đổi giải phẫu: {str(e)}")

        # Xác thực dự đoán nếu có validator
        metrics = None
        if validate and self.validator is not None:
            try:
                # Thực hiện xác thực dự đoán
                # Lưu ý: Cần có dữ liệu thực tế để xác thực
                # Trong trường hợp này, giả định xác thực với dữ liệu sẵn có
                metrics = self._validate_prediction(prediction)
                self.latest_metrics = metrics

                # Thông báo cho các callback
                for callback in self.validation_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Lỗi khi gọi validation callback: {str(e)}")

                logger.info(
                    f"Đã xác thực dự đoán với độ chính xác: {metrics.get_accuracy_summary()}"
                )
            except Exception as e:
                logger.error(f"Lỗi khi xác thực dự đoán: {str(e)}")

        # Cập nhật kế hoạch nếu auto_update được bật
        if self.auto_update and self.planner is not None:
            try:
                if hasattr(self.planner, "update_with_prediction"):
                    self.planner.update_with_prediction(prediction, metrics)
                    logger.info("Đã tự động cập nhật kế hoạch với dự đoán mới")
                else:
                    logger.warning("Planner không hỗ trợ update_with_prediction")
            except Exception as e:
                logger.error(f"Lỗi khi cập nhật kế hoạch: {str(e)}")

        return prediction, metrics

    def _validate_prediction(self, prediction: AnatomyPrediction) -> PredictionMetrics:
        """
        Xác thực dự đoán thay đổi giải phẫu nếu có dữ liệu thực tế.

        Parameters
        ----------
        prediction : AnatomyPrediction
            Dự đoán cần xác thực

        Returns
        -------
        PredictionMetrics
            Kết quả xác thực dự đoán

        Raises
        ------
        IntegrationError
            Nếu không có validator
        """
        if self.validator is None:
            raise IntegrationError("Không thể xác thực: validator chưa được thiết lập")

        # Trong thực tế, cần có dữ liệu thực tế để xác thực
        # Đây chỉ là mẫu giả định
        try:
            # Giả định có các phương thức để lấy dữ liệu thực tế
            if hasattr(self.predictor, "get_validation_data"):
                actual_images, actual_structures, actual_dates = (
                    self.predictor.get_validation_data()
                )

                # Xác thực dự đoán
                return self.validator.validate_predictions(
                    prediction=prediction,
                    actual_images=actual_images,
                    actual_structures=actual_structures,
                    actual_dates=actual_dates,
                )
            else:
                # Tạo metrics giả nếu không có dữ liệu xác thực
                logger.warning("Không có dữ liệu xác thực, tạo metrics giả")
                return PredictionMetrics()
        except Exception as e:
            logger.error(f"Lỗi khi xác thực dự đoán: {str(e)}")
            raise IntegrationError(f"Lỗi khi xác thực dự đoán: {str(e)}")

    def get_adaptation_options(self) -> Dict[str, Any]:
        """
        Lấy các tùy chọn thích ứng từ planner.

        Returns
        -------
        Dict[str, Any]
            Các tùy chọn thích ứng có sẵn

        Raises
        ------
        IntegrationError
            Nếu không có planner hoặc planner không hỗ trợ phương thức này
        """
        if self.planner is None:
            raise IntegrationError("Planner chưa được kết nối")

        if not hasattr(self.planner, "get_adaptation_options"):
            raise IntegrationError("Planner không hỗ trợ get_adaptation_options")

        try:
            return self.planner.get_adaptation_options()
        except Exception as e:
            logger.error(f"Lỗi khi lấy tùy chọn thích ứng: {str(e)}")
            raise IntegrationError(f"Lỗi khi lấy tùy chọn thích ứng: {str(e)}")

    def apply_adaptation(self, option_id: str, **kwargs) -> bool:
        """
        Áp dụng tùy chọn thích ứng đã chọn.

        Parameters
        ----------
        option_id : str
            ID của tùy chọn thích ứng cần áp dụng
        **kwargs
            Các tham số bổ sung cho tùy chọn thích ứng

        Returns
        -------
        bool
            True nếu áp dụng thành công, False nếu thất bại

        Raises
        ------
        IntegrationError
            Nếu không có planner hoặc planner không hỗ trợ phương thức này
        """
        if self.planner is None:
            raise IntegrationError("Planner chưa được kết nối")

        if not hasattr(self.planner, "apply_adaptation"):
            raise IntegrationError("Planner không hỗ trợ apply_adaptation")

        try:
            result = self.planner.apply_adaptation(option_id, **kwargs)
            if result:
                logger.info(f"Đã áp dụng thành công tùy chọn thích ứng: {option_id}")
            else:
                logger.warning(f"Không thể áp dụng tùy chọn thích ứng: {option_id}")
            return result
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng tùy chọn thích ứng: {str(e)}")
            raise IntegrationError(f"Lỗi khi áp dụng tùy chọn thích ứng: {str(e)}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Lấy tóm tắt về trạng thái hiện tại của tích hợp.

        Returns
        -------
        Dict[str, Any]
            Thông tin tóm tắt về trạng thái tích hợp
        """
        has_predictor = self.predictor is not None
        has_validator = self.validator is not None
        has_planner = self.planner is not None

        summary = {
            "has_predictor": has_predictor,
            "has_validator": has_validator,
            "has_planner": has_planner,
            "auto_update": self.auto_update,
            "has_latest_prediction": self.latest_prediction is not None,
            "has_latest_metrics": self.latest_metrics is not None,
        }

        if has_predictor:
            summary["predictor_type"] = self.predictor.__class__.__name__

        if has_validator:
            summary["validator_type"] = self.validator.__class__.__name__

        if has_planner:
            summary["planner_type"] = self.planner.__class__.__name__

        if self.latest_prediction is not None:
            summary["prediction_date"] = (
                self.latest_prediction.get_prediction_dates()[-1].isoformat()
                if self.latest_prediction.get_prediction_dates()
                else None
            )

        if self.latest_metrics is not None:
            summary["metrics_summary"] = self.latest_metrics.get_accuracy_summary()

        return summary


def create_prediction_metrics() -> PredictionMetrics:
    """
    Tạo đối tượng PredictionMetrics mới để lưu trữ kết quả đánh giá dự đoán.

    Returns
    -------
    PredictionMetrics
        Đối tượng mới để lưu trữ kết quả đánh giá
    """
    return PredictionMetrics()


def create_anatomy_prediction_integrator(
    predictor: Optional[AnatomyPredictor] = None,
    validator: Optional[ModelValidator] = None,
    auto_update: bool = False,
) -> AnatomyPredictionIntegrator:
    """
    Tạo đối tượng AnatomyPredictionIntegrator mới với các tham số tùy chọn.

    Parameters
    ----------
    predictor : AnatomyPredictor, optional
        Đối tượng dự đoán thay đổi giải phẫu, mặc định là None
    validator : ModelValidator, optional
        Đối tượng xác thực mô hình dự đoán, mặc định là None
    auto_update : bool, optional
        Bật/tắt chế độ tự động cập nhật, mặc định là False

    Returns
    -------
    AnatomyPredictionIntegrator
        Đối tượng tích hợp mới đã được cấu hình
    """
    integrator = AnatomyPredictionIntegrator(predictor=predictor, validator=validator)
    integrator.set_auto_update(auto_update)
    return integrator
