#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module tích hợp mô hình dự đoán thay đổi giải phẫu với lập kế hoạch thích ứng.

Module này cung cấp các công cụ và lớp để tích hợp mô hình dự đoán thay đổi giải phẫu
với quá trình lập kế hoạch thích ứng, nhằm cải thiện quá trình điều trị bằng cách
dự đoán và chuẩn bị trước cho các thay đổi giải phẫu có thể xảy ra.
"""

import os
import time
import logging
import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum, auto

from quangtps.core.types import Patient, Image, Structure, Plan, Dose
from quangtps.core.exceptions import PredictionIntegrationError
from quangtps.planning.plan import TreatmentPlan
from quangtps.adaptive.prediction.anatomy_prediction import (
    AnatomyPredictor,
    AnatomyPrediction,
    PredictionMethod,
)
from quangtps.adaptive.prediction.deformable_anatomy_predictor import (
    DeformableAnatomyPredictor,
)
from quangtps.adaptive.model_validator import ModelValidator
from quangtps.adaptive.adaptive_planning import AdaptivePlanner, PlanAdaptationSession
from quangtps.adaptive.optimization.real_time_adaptive_planning import (
    RealTimeAdaptivePlanner,
    RealTimeAdaptiveSession,
)
from quangtps.optimization.mco.pareto_navigator import ParetoNavigator
from quangtps.adaptive.temporal_analysis import TemporalAnalyzer

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Các chiến lược thích ứng khác nhau cho mô hình dự đoán."""

    REACTIVE = auto()  # Phản ứng: thích ứng khi phát hiện thay đổi
    PROACTIVE = auto()  # Chủ động: dự đoán trước và chuẩn bị kế hoạch
    LIBRARY = auto()  # Thư viện: tạo thư viện kế hoạch sẵn sàng
    ONLINE = auto()  # Trực tuyến: thích ứng trong thời gian thực


class PredictionConfidenceLevel(Enum):
    """Các mức độ tin cậy của dự đoán."""

    LOW = auto()  # Độ tin cậy thấp
    MEDIUM = auto()  # Độ tin cậy trung bình
    HIGH = auto()  # Độ tin cậy cao


class AnatomyPredictionIntegrator:
    """Lớp tích hợp mô hình dự đoán thay đổi giải phẫu với lập kế hoạch thích ứng."""

    def __init__(
        self,
        anatomy_predictor: Optional[AnatomyPredictor] = None,
        adaptive_planner: Optional[AdaptivePlanner] = None,
        realtime_planner: Optional[RealTimeAdaptivePlanner] = None,
        model_validator: Optional[ModelValidator] = None,
    ):
        """
        Khởi tạo tích hợp mô hình dự đoán với lập kế hoạch thích ứng.

        Parameters
        ----------
        anatomy_predictor : Optional[AnatomyPredictor], optional
            Bộ dự đoán thay đổi giải phẫu, mặc định là None
        adaptive_planner : Optional[AdaptivePlanner], optional
            Bộ lập kế hoạch thích ứng, mặc định là None
        realtime_planner : Optional[RealTimeAdaptivePlanner], optional
            Bộ lập kế hoạch thích ứng thời gian thực, mặc định là None
        model_validator : Optional[ModelValidator], optional
            Bộ kiểm tra mô hình, mặc định là None
        """
        # Khởi tạo các thành phần
        self.anatomy_predictor = anatomy_predictor or AnatomyPredictor()
        self.adaptive_planner = adaptive_planner or AdaptivePlanner()
        self.realtime_planner = realtime_planner or RealTimeAdaptivePlanner()
        self.model_validator = model_validator or ModelValidator()

        # Thiết lập chiến lược mặc định
        self.default_strategy = AdaptationStrategy.PROACTIVE
        self.prediction_timepoints = [7, 14, 21, 28]  # Dự đoán cho 7, 14, 21, 28 ngày

        # Lưu trữ kế hoạch dự đoán
        self.predicted_plans = {}  # Dict[patient_id, Dict[date, Plan]]
        self.predicted_doses = {}  # Dict[patient_id, Dict[date, Dose]]
        self.prediction_confidences = {}  # Dict[patient_id, Dict[date, confidence_level]]

        logger.info(
            "Khởi tạo tích hợp mô hình dự đoán thay đổi giải phẫu với lập kế hoạch thích ứng"
        )

    def set_prediction_timepoints(self, days: List[int]):
        """
        Thiết lập các mốc thời gian dự đoán.

        Parameters
        ----------
        days : List[int]
            Danh sách các ngày cần dự đoán (tính từ ngày tham chiếu)
        """
        self.prediction_timepoints = sorted(days)
        logger.info(
            f"Đã thiết lập các mốc thời gian dự đoán: {self.prediction_timepoints}"
        )

    def create_predicted_plans(
        self,
        patient: Patient,
        reference_plan: TreatmentPlan,
        strategy: Optional[AdaptationStrategy] = None,
    ) -> Dict[datetime.datetime, TreatmentPlan]:
        """
        Tạo các kế hoạch dự đoán cho các mốc thời gian tương lai.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        reference_plan : TreatmentPlan
            Kế hoạch tham chiếu
        strategy : Optional[AdaptationStrategy], optional
            Chiến lược thích ứng, mặc định là None (sử dụng chiến lược mặc định)

        Returns
        -------
        Dict[datetime.datetime, TreatmentPlan]
            Từ điển các kế hoạch dự đoán theo ngày
        """
        try:
            logger.info(f"Bắt đầu tạo kế hoạch dự đoán cho bệnh nhân {patient.id}")

            # Sử dụng chiến lược mặc định nếu không được chỉ định
            strategy = strategy or self.default_strategy

            # Lấy hình ảnh và cấu trúc tham chiếu
            reference_image = reference_plan.image
            reference_structures = reference_plan.structures
            reference_date = reference_plan.creation_date

            # Lấy lịch sử điều trị của bệnh nhân
            temporal_analyzer = TemporalAnalyzer()
            historical_data = temporal_analyzer.analyze_patient(patient)

            historical_images = historical_data.get("images", [])
            historical_structures = historical_data.get("structures", [])
            historical_dates = historical_data.get("dates", [])

            # Tạo dự đoán thay đổi giải phẫu
            prediction_dates = [
                reference_date + datetime.timedelta(days=days)
                for days in self.prediction_timepoints
            ]

            anatomy_prediction = self.anatomy_predictor.predict_anatomy_changes(
                patient=patient,
                historical_images=historical_images + [reference_image],
                historical_structures=historical_structures + [reference_structures],
                historical_dates=historical_dates + [reference_date],
                prediction_dates=prediction_dates,
                method=PredictionMethod.SPLINE,
            )

            # Tạo kế hoạch cho từng mốc thời gian dự đoán
            predicted_plans = {}

            for pred_date in prediction_dates:
                # Lấy cấu trúc và hình ảnh dự đoán cho ngày cụ thể
                pred_structures = anatomy_prediction.get_structure_at_date(pred_date)
                pred_image = anatomy_prediction.get_image_at_date(pred_date)
                confidence = anatomy_prediction.get_confidence_at_date(pred_date)

                # Bỏ qua nếu không có dữ liệu dự đoán
                if not pred_structures or not pred_image:
                    logger.warning(f"Không có dữ liệu dự đoán cho ngày {pred_date}")
                    continue

                # Tạo kế hoạch thích ứng dựa trên chiến lược
                if strategy == AdaptationStrategy.PROACTIVE:
                    # Tạo kế hoạch thích ứng chủ động
                    pred_plan = self._create_proactive_plan(
                        patient=patient,
                        reference_plan=reference_plan,
                        pred_structures=pred_structures,
                        pred_image=pred_image,
                        pred_date=pred_date,
                    )

                elif strategy == AdaptationStrategy.LIBRARY:
                    # Tạo thư viện kế hoạch
                    pred_plan = self._create_library_plan(
                        patient=patient,
                        reference_plan=reference_plan,
                        pred_structures=pred_structures,
                        pred_image=pred_image,
                        pred_date=pred_date,
                    )

                else:  # Mặc định là PROACTIVE
                    pred_plan = self._create_proactive_plan(
                        patient=patient,
                        reference_plan=reference_plan,
                        pred_structures=pred_structures,
                        pred_image=pred_image,
                        pred_date=pred_date,
                    )

                if pred_plan:
                    predicted_plans[pred_date] = pred_plan

                    # Lưu độ tin cậy của dự đoán
                    if patient.id not in self.prediction_confidences:
                        self.prediction_confidences[patient.id] = {}

                    # Phân loại độ tin cậy
                    if confidence > 0.8:
                        conf_level = PredictionConfidenceLevel.HIGH
                    elif confidence > 0.5:
                        conf_level = PredictionConfidenceLevel.MEDIUM
                    else:
                        conf_level = PredictionConfidenceLevel.LOW

                    self.prediction_confidences[patient.id][pred_date] = conf_level

            # Lưu các kế hoạch dự đoán
            self.predicted_plans[patient.id] = predicted_plans

            logger.info(
                f"Đã tạo {len(predicted_plans)} kế hoạch dự đoán cho bệnh nhân {patient.id}"
            )
            return predicted_plans

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch dự đoán: {str(e)}", exc_info=True)
            raise PredictionIntegrationError(f"Lỗi khi tạo kế hoạch dự đoán: {str(e)}")

    def _create_proactive_plan(
        self,
        patient: Patient,
        reference_plan: TreatmentPlan,
        pred_structures: Dict[str, Structure],
        pred_image: Image,
        pred_date: datetime.datetime,
    ) -> Optional[TreatmentPlan]:
        """
        Tạo kế hoạch thích ứng chủ động dựa trên dữ liệu dự đoán.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        reference_plan : TreatmentPlan
            Kế hoạch tham chiếu
        pred_structures : Dict[str, Structure]
            Cấu trúc dự đoán
        pred_image : Image
            Hình ảnh dự đoán
        pred_date : datetime.datetime
            Ngày dự đoán

        Returns
        -------
        Optional[TreatmentPlan]
            Kế hoạch thích ứng chủ động hoặc None nếu thất bại
        """
        try:
            logger.info(f"Tạo kế hoạch thích ứng chủ động cho ngày {pred_date}")

            # Tạo phiên thích ứng
            session = PlanAdaptationSession(
                patient=patient, original_plan=reference_plan, new_image=pred_image
            )

            # Thực hiện thích ứng kế hoạch
            adapted_plan = self.adaptive_planner.adapt_plan(
                session=session,
                action_type=None,  # Tự động xác định
            )

            if not adapted_plan:
                return None

            # Thêm thông tin dự đoán vào kế hoạch
            adapted_plan.metadata["prediction_date"] = pred_date.isoformat()
            adapted_plan.metadata["is_predicted"] = True
            adapted_plan.metadata["reference_plan_id"] = reference_plan.id

            # Đặt ID cho kế hoạch dự đoán
            days_from_ref = (pred_date - reference_plan.creation_date).days
            adapted_plan.set_id(f"{reference_plan.id}_pred_{days_from_ref}d")

            return adapted_plan

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch thích ứng chủ động: {str(e)}")
            return None

    def _create_library_plan(
        self,
        patient: Patient,
        reference_plan: TreatmentPlan,
        pred_structures: Dict[str, Structure],
        pred_image: Image,
        pred_date: datetime.datetime,
    ) -> Optional[TreatmentPlan]:
        """
        Tạo kế hoạch cho thư viện kế hoạch dựa trên dữ liệu dự đoán.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        reference_plan : TreatmentPlan
            Kế hoạch tham chiếu
        pred_structures : Dict[str, Structure]
            Cấu trúc dự đoán
        pred_image : Image
            Hình ảnh dự đoán
        pred_date : datetime.datetime
            Ngày dự đoán

        Returns
        -------
        Optional[TreatmentPlan]
            Kế hoạch thư viện hoặc None nếu thất bại
        """
        try:
            logger.info(f"Tạo kế hoạch thư viện cho ngày {pred_date}")

            # Tạo trình khám phá Pareto để tạo các kế hoạch đa mục tiêu
            pareto_navigator = ParetoNavigator()

            # Tạo kế hoạch thư viện cho nhiều điểm trên mặt Pareto
            library_plans = pareto_navigator.create_plan_library(
                patient=patient,
                reference_plan=reference_plan,
                new_structures=pred_structures,
                new_image=pred_image,
                options={"num_plans": 5},  # Tạo 5 kế hoạch khác nhau
            )

            if not library_plans:
                return None

            # Chọn kế hoạch tốt nhất theo tiêu chí mặc định
            best_plan = library_plans[0]

            # Thêm thông tin dự đoán vào kế hoạch
            best_plan.metadata["prediction_date"] = pred_date.isoformat()
            best_plan.metadata["is_predicted"] = True
            best_plan.metadata["reference_plan_id"] = reference_plan.id
            best_plan.metadata["library_size"] = len(library_plans)

            # Đặt ID cho kế hoạch dự đoán
            days_from_ref = (pred_date - reference_plan.creation_date).days
            best_plan.set_id(f"{reference_plan.id}_lib_{days_from_ref}d")

            return best_plan

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch thư viện: {str(e)}")
            return None

    def select_plan_for_treatment(
        self,
        patient_id: str,
        treatment_date: datetime.datetime,
        verification_image: Optional[Image] = None,
    ) -> Optional[TreatmentPlan]:
        """
        Chọn kế hoạch phù hợp nhất từ các kế hoạch dự đoán sẵn có cho một ngày điều trị.

        Parameters
        ----------
        patient_id : str
            ID bệnh nhân
        treatment_date : datetime.datetime
            Ngày điều trị
        verification_image : Optional[Image], optional
            Hình ảnh xác minh (CBCT, v.v.), mặc định là None

        Returns
        -------
        Optional[TreatmentPlan]
            Kế hoạch phù hợp nhất hoặc None nếu không có kế hoạch nào phù hợp
        """
        try:
            # Kiểm tra xem có kế hoạch dự đoán cho bệnh nhân này không
            if patient_id not in self.predicted_plans:
                logger.warning(f"Không có kế hoạch dự đoán cho bệnh nhân {patient_id}")
                return None

            plans = self.predicted_plans[patient_id]

            # Nếu không có hình ảnh xác minh, chọn kế hoạch gần nhất với ngày điều trị
            if not verification_image:
                closest_date = min(
                    plans.keys(),
                    key=lambda d: abs((d - treatment_date).total_seconds()),
                )

                # Kiểm tra xem ngày gần nhất có quá xa không (> 7 ngày)
                if abs((closest_date - treatment_date).days) > 7:
                    logger.warning(
                        f"Kế hoạch dự đoán gần nhất cách ngày điều trị {abs((closest_date - treatment_date).days)} ngày"
                    )

                logger.info(f"Đã chọn kế hoạch dự đoán cho ngày {closest_date}")
                return plans[closest_date]

            # Nếu có hình ảnh xác minh, so sánh với hình ảnh dự đoán để chọn kế hoạch phù hợp nhất
            best_match = None
            best_similarity = -1

            for pred_date, plan in plans.items():
                # Tính độ tương đồng giữa hình ảnh xác minh và hình ảnh dự đoán
                similarity = self._calculate_image_similarity(
                    verification_image, plan.image
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = plan

            if best_match and best_similarity > 0.8:  # Ngưỡng tương đồng
                logger.info(
                    f"Đã chọn kế hoạch phù hợp nhất với độ tương đồng {best_similarity:.2f}"
                )
                return best_match

            logger.warning(
                f"Không tìm thấy kế hoạch dự đoán phù hợp (độ tương đồng tốt nhất: {best_similarity:.2f})"
            )
            return None

        except Exception as e:
            logger.error(f"Lỗi khi chọn kế hoạch điều trị: {str(e)}")
            return None

    def _calculate_image_similarity(self, image1: Image, image2: Image) -> float:
        """
        Tính độ tương đồng giữa hai hình ảnh.

        Parameters
        ----------
        image1 : Image
            Hình ảnh thứ nhất
        image2 : Image
            Hình ảnh thứ hai

        Returns
        -------
        float
            Độ tương đồng từ 0 đến 1
        """
        try:
            # Lấy dữ liệu từ hai hình ảnh
            data1 = image1.get_array()
            data2 = image2.get_array()

            # Chuẩn hóa kích thước nếu cần
            if data1.shape != data2.shape:
                # Cần có xử lý phức tạp hơn, nhưng đơn giản hóa ở đây
                return 0.0

            # Tính hệ số tương quan
            correlation = np.corrcoef(data1.flatten(), data2.flatten())[0, 1]

            # Chuyển hệ số tương quan từ [-1, 1] sang [0, 1]
            similarity = (correlation + 1) / 2

            return max(0, min(1, similarity))  # Đảm bảo giá trị từ 0 đến 1

        except Exception as e:
            logger.error(f"Lỗi khi tính độ tương đồng giữa hai hình ảnh: {str(e)}")
            return 0.0

    def integrate_with_realtime_planning(
        self, patient: Patient, reference_plan: TreatmentPlan, verification_image: Image
    ) -> RealTimeAdaptiveSession:
        """
        Tích hợp mô hình dự đoán với lập kế hoạch thích ứng thời gian thực.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        reference_plan : TreatmentPlan
            Kế hoạch tham chiếu
        verification_image : Image
            Hình ảnh xác minh

        Returns
        -------
        RealTimeAdaptiveSession
            Phiên lập kế hoạch thích ứng thời gian thực
        """
        # Tạo phiên thích ứng thời gian thực
        session = self.realtime_planner.start_adaptation_session(
            patient=patient,
            original_plan=reference_plan,
            new_image=verification_image,
            auto_start=False,  # Không bắt đầu ngay lập tức
        )

        # Thử chọn kế hoạch dự đoán phù hợp
        predicted_plan = self.select_plan_for_treatment(
            patient_id=patient.id,
            treatment_date=datetime.datetime.now(),
            verification_image=verification_image,
        )

        if predicted_plan:
            # Nếu có kế hoạch dự đoán phù hợp, sử dụng nó làm khởi tạo ban đầu
            session.log(
                f"Sử dụng kế hoạch dự đoán ID={predicted_plan.id} làm khởi tạo ban đầu"
            )
            session.prediction_used = True
            session.prediction_plan_id = predicted_plan.id

            # Điều chỉnh kế hoạch dự đoán tương ứng với hình ảnh xác minh
            # Quá trình này sẽ nhanh hơn vì đã có kế hoạch dự đoán sẵn
            pass

        # Bắt đầu phiên thích ứng
        self.realtime_planner._start_session_thread(session)

        return session

    def evaluate_prediction_accuracy(
        self,
        patient: Patient,
        predicted_plans: Dict[datetime.datetime, TreatmentPlan],
        actual_images: List[Image],
        actual_structures: List[Dict[str, Structure]],
        actual_dates: List[datetime.datetime],
    ) -> Dict[str, Any]:
        """
        Đánh giá độ chính xác của các dự đoán.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        predicted_plans : Dict[datetime.datetime, TreatmentPlan]
            Từ điển các kế hoạch dự đoán theo ngày
        actual_images : List[Image]
            Danh sách hình ảnh thực tế
        actual_structures : List[Dict[str, Structure]]
            Danh sách cấu trúc thực tế
        actual_dates : List[datetime.datetime]
            Danh sách ngày thực tế

        Returns
        -------
        Dict[str, Any]
            Kết quả đánh giá
        """
        try:
            logger.info(f"Đánh giá độ chính xác dự đoán cho bệnh nhân {patient.id}")

            # Chuyển đổi kế hoạch dự đoán thành AnatomyPrediction
            pred_structures = {}
            pred_images = {}

            for pred_date, plan in predicted_plans.items():
                pred_structures[pred_date] = plan.structures
                pred_images[pred_date] = plan.image

            # Tạo đối tượng AnatomyPrediction
            reference_date = min(predicted_plans.keys())
            prediction = AnatomyPrediction(reference_date, patient.id)

            for pred_date in sorted(predicted_plans.keys()):
                prediction.add_prediction_timepoint(
                    date=pred_date,
                    structures=pred_structures[pred_date],
                    image=pred_images[pred_date],
                )

            # Sử dụng ModelValidator để đánh giá độ chính xác
            validator = ModelValidator()
            metrics = validator.validate_predictions(
                prediction=prediction,
                actual_images=actual_images,
                actual_structures=actual_structures,
                actual_dates=actual_dates,
            )

            # Tạo báo cáo đánh giá
            output_dir = os.path.join("reports", "prediction_evaluation", patient.id)
            validator.generate_validation_report(
                os.path.join(output_dir, "validation_report.csv")
            )

            # Chuyển đổi kết quả thành dict
            result = {
                "patient_id": patient.id,
                "number_of_predictions": len(predicted_plans),
                "number_of_actual_datapoints": len(actual_dates),
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "structure_metrics": metrics.structure_metrics,
            }

            logger.info(
                f"Đã hoàn thành đánh giá độ chính xác dự đoán cho bệnh nhân {patient.id}"
            )
            return result

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá độ chính xác dự đoán: {str(e)}")
            return {"patient_id": patient.id, "error": str(e)}


if __name__ == "__main__":
    # Mã để chạy thử và kiểm tra module
    logging.basicConfig(level=logging.INFO)
    logger.info("Kiểm tra module anatomy_prediction_integration.py")

    # Tạo đối tượng tích hợp
    integrator = AnatomyPredictionIntegrator()
    logger.info("Đã tạo đối tượng tích hợp dự đoán thay đổi giải phẫu thành công")
