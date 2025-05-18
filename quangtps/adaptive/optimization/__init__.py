#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module xử lý tối ưu hóa kế hoạch thích ứng trong QuangTPS.

Module này cung cấp các lớp và hàm để tối ưu hóa kế hoạch thích ứng dựa trên
sự thay đổi giải phẫu của bệnh nhân theo thời gian, đảm bảo kế hoạch xạ trị
luôn phù hợp với giải phẫu hiện tại của bệnh nhân.
"""

from quangtps.adaptive.optimization.real_time_adaptive_planning import (
    RealTimeAdaptivePlanner,
    RealTimeAdaptiveSession,
    AdaptationPriority,
    AdaptationStatus,
    create_real_time_adaptive_planner,
)

from quangtps.adaptive.optimization.anatomy_prediction_integration import (
    AnatomyPredictionIntegrator,
    AdaptationStrategy,
    PredictionConfidenceLevel,
)


# Tạo hàm tiện ích để tạo real-time adaptive planner
def create_real_time_adaptive_planner():
    """
    Tạo hệ thống lập kế hoạch thích ứng thời gian thực.

    Hàm này tạo và cấu hình đối tượng RealTimeAdaptivePlanner để hỗ trợ
    lập kế hoạch thích ứng nhanh chóng trong quá trình điều trị.

    Returns
    -------
    RealTimeAdaptivePlanner hoặc object tương tự
        Đối tượng lập kế hoạch thích ứng thời gian thực
    """
    try:
        # Thử import RealTimeAdaptivePlanner từ module chính
        from quangtps.adaptive.optimization.real_time_adaptive_planning import (
            RealTimeAdaptivePlanner,
        )

        # Tạo đối tượng với cấu hình mặc định
        planner = RealTimeAdaptivePlanner()

        # Cấu hình các thông số phù hợp
        planner.set_max_optimization_time(60)  # 60 giây tối đa cho tối ưu hóa
        planner.set_adaptation_strategies(
            ["ISOCENTER_SHIFT", "FLUENCE_ADJUST", "REPLAN"]
        )

        return planner

    except ImportError as e:
        import logging

        logging.warning(f"Không thể import RealTimeAdaptivePlanner: {str(e)}")

        # Tạo đối tượng giả khi không thể tạo đối tượng thật
        class DummyRealTimeAdaptivePlanner:
            """Đối tượng giả cho RealTimeAdaptivePlanner khi không tồn tại."""

            def __init__(self):
                self.name = "DummyRealTimeAdaptivePlanner"
                self.strategies = []
                self.max_optimization_time = 60
                logging.warning("Đang sử dụng DummyRealTimeAdaptivePlanner thay thế")

            def set_predictor(self, predictor):
                """Thiết lập predictor giả."""
                logging.info(
                    f"Thiết lập predictor {predictor.__class__.__name__} cho dummy planner"
                )

            def set_max_optimization_time(self, time_seconds):
                """Thiết lập thời gian tối ưu hóa tối đa."""
                self.max_optimization_time = time_seconds

            def set_adaptation_strategies(self, strategies):
                """Thiết lập các chiến lược thích ứng."""
                self.strategies = strategies

            def update_with_prediction(self, prediction, metrics=None):
                """Giả lập cập nhật với dự đoán."""
                logging.info("Giả lập cập nhật kế hoạch với dự đoán mới")
                return True

            def get_adaptation_options(self):
                """Trả về các tùy chọn thích ứng giả."""
                return {
                    "isocenter_shift": {
                        "id": "isocenter_shift",
                        "name": "Dịch chuyển isocenter",
                        "description": "Dịch chuyển isocenter để bù đắp thay đổi vị trí cấu trúc",
                    },
                    "fluence_adjust": {
                        "id": "fluence_adjust",
                        "name": "Điều chỉnh fluence",
                        "description": "Điều chỉnh fluence để bù đắp thay đổi hình dạng và vị trí",
                    },
                    "replan": {
                        "id": "replan",
                        "name": "Lập kế hoạch lại",
                        "description": "Tạo kế hoạch hoàn toàn mới",
                    },
                }

            def apply_adaptation(self, option_id, **kwargs):
                """Giả lập áp dụng phương án thích ứng."""
                logging.info(f"Giả lập áp dụng phương án thích ứng: {option_id}")
                return True

        return DummyRealTimeAdaptivePlanner()


# Tạo hàm tiện ích để khởi tạo hệ thống thích ứng tích hợp
def create_integrated_adaptive_system(patient=None, reference_image=None):
    """
    Tạo hệ thống thích ứng tích hợp với dự đoán thay đổi giải phẫu.

    Hàm này tạo và tích hợp các thành phần cần thiết cho hệ thống lập kế hoạch
    thích ứng thời gian thực, bao gồm dự đoán thay đổi giải phẫu và tối ưu hóa
    kế hoạch thích ứng.

    Parameters
    ----------
    patient : Patient, optional
        Đối tượng bệnh nhân cần lập kế hoạch thích ứng
    reference_image : Image, optional
        Hình ảnh tham chiếu cho quá trình thích ứng

    Returns
    -------
    dict
        Dictionary chứa các thành phần của hệ thống thích ứng tích hợp
    """
    try:
        # Import các module cần thiết
        from quangtps.adaptive.prediction import DeformableAnatomyPredictor
        from quangtps.adaptive.model_validator import ModelValidator
        import logging

        # Khởi tạo các thành phần với tham số tùy chọn
        anatomy_predictor = None
        if patient is not None and reference_image is not None:
            try:
                anatomy_predictor = DeformableAnatomyPredictor(
                    patient=patient, reference_image=reference_image
                )
            except Exception as e:
                logging.warning(
                    f"Không thể khởi tạo DeformableAnatomyPredictor: {str(e)}"
                )
                # Tạo đối tượng rỗng nếu không thể khởi tạo với tham số
                anatomy_predictor = DeformableAnatomyPredictor.__new__(
                    DeformableAnatomyPredictor
                )
        else:
            # Tạo đối tượng rỗng nếu chưa có tham số
            anatomy_predictor = DeformableAnatomyPredictor.__new__(
                DeformableAnatomyPredictor
            )

        # Khởi tạo real-time planner
        realtime_planner = create_real_time_adaptive_planner()
        model_validator = ModelValidator()

        # Tạo tích hợp với AnatomyPredictionIntegrator
        try:
            from quangtps.adaptive.optimization.anatomy_prediction_integration import (
                AnatomyPredictionIntegrator,
            )

            prediction_integrator = AnatomyPredictionIntegrator(
                predictor=anatomy_predictor, validator=model_validator
            )
        except (ImportError, Exception) as e:
            logging.warning(f"Không thể tạo AnatomyPredictionIntegrator: {str(e)}")
            prediction_integrator = None

        # Tích hợp các thành phần
        integrated_system = {
            "predictor": anatomy_predictor,
            "realtime_planner": realtime_planner,
            "validator": model_validator,
            "prediction_integrator": prediction_integrator,
        }

        # Thiết lập các liên kết giữa các thành phần
        if anatomy_predictor is not None:
            try:
                # Thử thiết lập validator nếu phương thức tồn tại
                if hasattr(anatomy_predictor, "set_validator"):
                    anatomy_predictor.set_validator(model_validator)
            except (AttributeError, TypeError) as e:
                logging.warning(f"Không thể thiết lập validator: {str(e)}")

        if realtime_planner is not None:
            try:
                # Thử thiết lập predictor nếu phương thức tồn tại
                if hasattr(realtime_planner, "set_predictor"):
                    realtime_planner.set_predictor(anatomy_predictor)
            except (AttributeError, TypeError) as e:
                logging.warning(f"Không thể thiết lập predictor: {str(e)}")

        # Thiết lập tích hợp giữa prediction_integrator và realtime_planner
        if prediction_integrator is not None and realtime_planner is not None:
            try:
                if hasattr(prediction_integrator, "connect_planner"):
                    prediction_integrator.connect_planner(realtime_planner)
            except Exception as e:
                logging.warning(
                    f"Không thể kết nối prediction_integrator với planner: {str(e)}"
                )

        return integrated_system
    except ImportError as e:
        import logging

        logging.error(f"Không thể tạo hệ thống thích ứng tích hợp: {str(e)}")
        return None


__all__ = [
    "RealTimeAdaptivePlanner",
    "RealTimeAdaptiveSession",
    "AdaptationPriority",
    "AdaptationStatus",
    "create_real_time_adaptive_planner",
    "AnatomyPredictionIntegrator",
    "AdaptationStrategy",
    "PredictionConfidenceLevel",
    "create_integrated_adaptive_system",
]
