#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module lập kế hoạch thích ứng và tối ưu hóa.

Module này cung cấp các lớp và phương thức để tích hợp giữa các thành phần dự đoán
thay đổi giải phẫu và lập kế hoạch tự động thích ứng với các thay đổi đó.
"""

import logging
from typing import Optional, Dict, Any, List, Type, Union

logger = logging.getLogger(__name__)

# Hỗ trợ import các module cần thiết với xử lý ngoại lệ và cơ chế dự phòng
try:
    from quangtps.adaptive.prediction.deformable_anatomy_predictor import (
        DeformableAnatomyPredictor,
    )

    has_anatomy_predictor = True
    logger.info("Đã import DeformableAnatomyPredictor thành công")
except ImportError as e:
    has_anatomy_predictor = False
    logger.warning(f"Không thể import DeformableAnatomyPredictor: {str(e)}")

    # Tạo lớp giả cho DeformableAnatomyPredictor
    class DeformableAnatomyPredictor:
        """Lớp giả cho DeformableAnatomyPredictor khi module thực không khả dụng."""

        def __init__(self, **kwargs):
            logger.warning("Sử dụng lớp giả DeformableAnatomyPredictor")
            self.is_dummy = True

        def predict_multiple_timepoints(self, **kwargs):
            logger.warning("Gọi phương thức giả predict_multiple_timepoints")
            return {}

        def set_validator(self, validator):
            logger.warning("Gọi phương thức giả set_validator")
            pass


try:
    from quangtps.adaptive.robust_adaptive_planning import RobustAdaptivePlanner

    has_adaptive_planner = True
    logger.info("Đã import RobustAdaptivePlanner thành công")
except ImportError as e:
    has_adaptive_planner = False
    logger.warning(f"Không thể import RobustAdaptivePlanner: {str(e)}")

    # Tạo lớp giả cho RobustAdaptivePlanner
    class RobustAdaptivePlanner:
        """Lớp giả cho RobustAdaptivePlanner khi module thực không khả dụng."""

        def __init__(self, **kwargs):
            logger.warning("Sử dụng lớp giả RobustAdaptivePlanner")
            self.is_dummy = True

        def set_anatomy_predictor(self, predictor):
            logger.warning("Gọi phương thức giả set_anatomy_predictor")
            pass

        def generate_adaptive_plans(self, **kwargs):
            logger.warning("Gọi phương thức giả generate_adaptive_plans")
            return {}

        def set_validator(self, validator):
            logger.warning("Gọi phương thức giả set_validator")
            pass


try:
    from quangtps.adaptive.model_validator import ModelValidator

    has_model_validator = True
    logger.info("Đã import ModelValidator thành công")
except ImportError as e:
    has_model_validator = False
    logger.warning(f"Không thể import ModelValidator: {str(e)}")

    # Tạo lớp giả cho ModelValidator
    class ModelValidator:
        """Lớp giả cho ModelValidator khi module thực không khả dụng."""

        def __init__(self, **kwargs):
            logger.warning("Sử dụng lớp giả ModelValidator")
            self.is_dummy = True

        def validate_predictions(self, **kwargs):
            logger.warning("Gọi phương thức giả validate_predictions")
            return {}


class AnatomyPredictionIntegrator:
    """
    Lớp tích hợp giữa dự đoán thay đổi giải phẫu và lập kế hoạch thích ứng.

    Lớp này cung cấp giao diện giữa các module dự đoán thay đổi giải phẫu và
    module lập kế hoạch thích ứng, đảm bảo luồng dữ liệu và thông tin nhất quán
    giữa các thành phần khác nhau của hệ thống thích ứng.
    """

    def __init__(self):
        """Khởi tạo tích hợp giữa dự đoán và lập kế hoạch thích ứng."""
        self.anatomy_predictor = None
        self.adaptive_planner = None
        self.model_validator = None
        self.is_initialized = False
        self.config = {}

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """
        Khởi tạo các thành phần dự đoán và lập kế hoạch thích ứng.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Cấu hình cho các thành phần, bao gồm cài đặt cho predictor và planner.

        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu thất bại.
        """
        if config is None:
            config = {}

        self.config = config
        success = True

        # Khởi tạo anatomy predictor nếu có
        try:
            predictor_config = config.get("predictor", {})
            self.anatomy_predictor = DeformableAnatomyPredictor(**predictor_config)
            logger.info("Đã khởi tạo DeformableAnatomyPredictor thành công")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo DeformableAnatomyPredictor: {str(e)}")
            success = False

        # Khởi tạo adaptive planner nếu có
        try:
            planner_config = config.get("planner", {})
            self.adaptive_planner = RobustAdaptivePlanner(**planner_config)
            logger.info("Đã khởi tạo RobustAdaptivePlanner thành công")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo RobustAdaptivePlanner: {str(e)}")
            success = False

        # Khởi tạo model validator nếu có
        try:
            validator_config = config.get("validator", {})
            self.model_validator = ModelValidator(**validator_config)
            logger.info("Đã khởi tạo ModelValidator thành công")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo ModelValidator: {str(e)}")
            # Đây không phải là thành phần quan trọng, nên không đánh dấu là thất bại

        # Thiết lập liên kết giữa các thành phần nếu chúng đều tồn tại
        if success and self.anatomy_predictor and self.adaptive_planner:
            try:
                self._setup_component_links()
                logger.info(
                    "Đã thiết lập liên kết giữa các thành phần thích ứng thành công"
                )
            except Exception as e:
                logger.error(
                    f"Lỗi khi thiết lập liên kết giữa các thành phần: {str(e)}"
                )
                success = False

        self.is_initialized = success
        return success

    def _setup_component_links(self):
        """
        Thiết lập liên kết giữa các thành phần dự đoán và lập kế hoạch thích ứng.

        Method này đảm bảo các thành phần có thể giao tiếp và trao đổi dữ liệu đúng cách.
        """
        if self.anatomy_predictor and self.adaptive_planner:
            # Kết nối dự đoán với lập kế hoạch
            if hasattr(self.adaptive_planner, "set_anatomy_predictor"):
                self.adaptive_planner.set_anatomy_predictor(self.anatomy_predictor)
                logger.debug("Đã kết nối anatomy_predictor với adaptive_planner")

            # Kết nối validator với predictor nếu cả hai tồn tại
            if self.model_validator and hasattr(
                self.anatomy_predictor, "set_validator"
            ):
                self.anatomy_predictor.set_validator(self.model_validator)
                logger.debug("Đã kết nối model_validator với anatomy_predictor")

            # Kết nối validator với planner nếu cần thiết
            if self.model_validator and hasattr(self.adaptive_planner, "set_validator"):
                self.adaptive_planner.set_validator(self.model_validator)
                logger.debug("Đã kết nối model_validator với adaptive_planner")

    def predict_and_plan(
        self,
        initial_images: List[Any],
        initial_structures: List[Any],
        time_points: List[float],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Dự đoán thay đổi giải phẫu và tạo kế hoạch thích ứng cho một loạt thời điểm.

        Parameters
        ----------
        initial_images : List[Any]
            Danh sách hình ảnh ban đầu được sử dụng cho dự đoán.
        initial_structures : List[Any]
            Danh sách cấu trúc ban đầu được sử dụng cho dự đoán.
        time_points : List[float]
            Các thời điểm (tính bằng ngày) cần dự đoán và lập kế hoạch.
        **kwargs
            Các tham số bổ sung cho dự đoán và lập kế hoạch.

        Returns
        -------
        Dict[str, Any]
            Kết quả chứa các dự đoán giải phẫu và kế hoạch thích ứng.
        """
        if not self.is_initialized:
            logger.error(
                "Tích hợp chưa được khởi tạo, không thể dự đoán và lập kế hoạch"
            )
            return {"success": False, "message": "Tích hợp chưa được khởi tạo"}

        result = {"success": False, "predictions": {}, "plans": {}, "metrics": {}}

        # Thực hiện dự đoán giải phẫu nếu có thể
        if self.anatomy_predictor:
            try:
                logger.info(
                    f"Bắt đầu dự đoán giải phẫu cho {len(time_points)} thời điểm"
                )
                predictions = self.anatomy_predictor.predict_multiple_timepoints(
                    initial_images=initial_images,
                    initial_structures=initial_structures,
                    time_points=time_points,
                    **kwargs,
                )
                result["predictions"] = predictions
                result["success"] = True
                logger.info(
                    f"Đã dự đoán thành công giải phẫu cho {len(predictions)} thời điểm"
                )
            except Exception as e:
                logger.error(f"Lỗi khi dự đoán giải phẫu: {str(e)}")
                return {"success": False, "message": f"Lỗi dự đoán: {str(e)}"}
        else:
            logger.error("Không có anatomy_predictor khả dụng")
            return {"success": False, "message": "Không có anatomy_predictor khả dụng"}

        # Lập kế hoạch thích ứng dựa trên dự đoán nếu có thể
        if self.adaptive_planner and result["success"]:
            try:
                logger.info("Bắt đầu lập kế hoạch thích ứng dựa trên dự đoán")
                plans = self.adaptive_planner.generate_adaptive_plans(
                    predictions=result["predictions"], **kwargs
                )
                result["plans"] = plans
                logger.info(f"Đã tạo thành công {len(plans)} kế hoạch thích ứng")
            except Exception as e:
                logger.error(f"Lỗi khi lập kế hoạch thích ứng: {str(e)}")
                result["message"] = (
                    f"Dự đoán thành công nhưng lập kế hoạch thất bại: {str(e)}"
                )

        # Đánh giá kết quả nếu có validator
        if self.model_validator and result["success"]:
            try:
                metrics = self.model_validator.validate_predictions(
                    predictions=result["predictions"],
                    ground_truth=kwargs.get("ground_truth", None),
                )
                result["metrics"] = metrics
                logger.info("Đã đánh giá kết quả dự đoán thành công")
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá kết quả: {str(e)}")
                # Không ảnh hưởng đến kết quả tổng thể

        return result


def create_integrated_adaptive_system(
    config: Dict[str, Any] = None,
    predictor_class: Optional[Type] = None,
    planner_class: Optional[Type] = None,
    validator_class: Optional[Type] = None,
    backup_predictor: bool = True,
    backup_planner: bool = True,
    backup_validator: bool = True,
) -> Optional[AnatomyPredictionIntegrator]:
    """
    Tạo hệ thống tích hợp lập kế hoạch thích ứng.

    Parameters
    ----------
    config : Dict[str, Any], optional
        Cấu hình cho hệ thống tích hợp, mặc định là None
    predictor_class : Optional[Type], optional
        Lớp dự đoán thay đổi giải phẫu tùy chỉnh, mặc định là None
    planner_class : Optional[Type], optional
        Lớp lập kế hoạch thích ứng tùy chỉnh, mặc định là None
    validator_class : Optional[Type], optional
        Lớp validator tùy chỉnh, mặc định là None
    backup_predictor : bool, optional
        Cho phép sử dụng predictor dự phòng nếu predictor chính không khả dụng, mặc định là True
    backup_planner : bool, optional
        Cho phép sử dụng planner dự phòng nếu planner chính không khả dụng, mặc định là True
    backup_validator : bool, optional
        Cho phép sử dụng validator dự phòng nếu validator chính không khả dụng, mặc định là True

    Returns
    -------
    Optional[AnatomyPredictionIntegrator]
        Đối tượng tích hợp đã được khởi tạo hoặc None nếu không thể tạo
    """
    try:
        # Tạo integrator
        integrator = AnatomyPredictionIntegrator()

        # Khởi tạo với cấu hình mặc định hoặc cấu hình được cung cấp
        if config is None:
            config = {}

        # Cập nhật config với các lớp tùy chỉnh nếu được cung cấp
        if predictor_class:
            config["predictor_class"] = predictor_class
        if planner_class:
            config["planner_class"] = planner_class
        if validator_class:
            config["validator_class"] = validator_class

        # Thêm cấu hình dự phòng
        config["backup_options"] = {
            "backup_predictor": backup_predictor,
            "backup_planner": backup_planner,
            "backup_validator": backup_validator,
        }

        # Khởi tạo hệ thống
        success = integrator.initialize(config)

        if success:
            logger.info(
                "Đã tạo và khởi tạo hệ thống tích hợp lập kế hoạch thích ứng thành công"
            )
            return integrator
        else:
            logger.error("Không thể khởi tạo hệ thống tích hợp lập kế hoạch thích ứng")
            return None
    except Exception as e:
        logger.error(f"Lỗi khi tạo hệ thống tích hợp lập kế hoạch thích ứng: {str(e)}")
        return None


# Export các lớp và hàm
__all__ = [
    "AnatomyPredictionIntegrator",
    "create_integrated_adaptive_system",
]

# Nếu có DeformableAnatomyPredictor, thêm vào __all__
if has_anatomy_predictor:
    __all__.append("DeformableAnatomyPredictor")

# Nếu có RobustAdaptivePlanner, thêm vào __all__
if has_adaptive_planner:
    __all__.append("RobustAdaptivePlanner")

# Nếu có ModelValidator, thêm vào __all__
if has_model_validator:
    __all__.append("ModelValidator")
