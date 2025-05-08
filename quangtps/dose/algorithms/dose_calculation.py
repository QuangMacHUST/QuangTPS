#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tính toán liều.

Module này cung cấp giao diện chung cho các thuật toán tính toán liều
khác nhau trong hệ thống lập kế hoạch xạ trị.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Type
import numpy as np
import threading
import traceback
from enum import Enum, auto

try:
    import SimpleITK as sitk

    HAS_SITK = True
except ImportError:
    HAS_SITK = False

    # Tạo lớp mô phỏng nếu không có SimpleITK
    class sitk:
        class Image:
            pass

        @staticmethod
        def GetArrayFromImage(image):
            return np.array([])

        @staticmethod
        def GetImageFromArray(array):
            return sitk.Image()


from quangtps.dose.algorithms.base import (
    DoseCalculationImplementer,
    DoseCalculationAlgorithm,
    ValidationError,
    AlgorithmError,
)
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)


class DoseCalculationEngine:
    """
    Công cụ tính toán liều chính.

    Lớp này quản lý các thuật toán tính toán liều khác nhau và cung cấp
    giao diện thống nhất để sử dụng các thuật toán trong hệ thống.
    """

    def __init__(self):
        """Khởi tạo DoseCalculationEngine."""
        self.implementers = {}
        self.current_algorithm = None
        self.status_callback = None
        self.calculation_thread = None
        self.calculation_in_progress = False
        self.calculation_cancelled = False
        self._register_default_implementers()

    def _register_default_implementers(self):
        """Đăng ký các implementer mặc định."""
        # Import các implementer
        try:
            from quangtps.dose.algorithms.aaa import AAADoseCalculation

            self.register_implementer(AAADoseCalculation())
            logger.info("Đã đăng ký thuật toán AAA")
        except ImportError:
            logger.warning("Không thể import thuật toán AAA")
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký thuật toán AAA: {str(e)}")

        try:
            from quangtps.dose.algorithms.ccc import CollapsedConeImplementer

            self.register_implementer(CollapsedConeImplementer())
            logger.info("Đã đăng ký thuật toán CCC")
        except ImportError:
            logger.warning("Không thể import thuật toán CCC")
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký thuật toán CCC: {str(e)}")

        try:
            from quangtps.dose.algorithms.acuros import AcurosXBImplementer

            self.register_implementer(AcurosXBImplementer())
            logger.info("Đã đăng ký thuật toán Acuros XB")
        except ImportError:
            logger.warning("Không thể import thuật toán Acuros XB")
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký thuật toán Acuros XB: {str(e)}")

        try:
            from quangtps.dose.algorithms.gbbs import GBBSImplementer

            self.register_implementer(GBBSImplementer())
            logger.info("Đã đăng ký thuật toán GBBS")
        except ImportError:
            logger.warning("Không thể import thuật toán GBBS")
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký thuật toán GBBS: {str(e)}")

        # Kiểm tra Monte Carlo plugin
        try:
            from quangtps.plugins.montecarlo_dose.monte_carlo import (
                MonteCarloImplementer,
            )

            self.register_implementer(MonteCarloImplementer())
            logger.info("Đã đăng ký thuật toán Monte Carlo")
        except ImportError:
            logger.warning(
                "Không thể import thuật toán Monte Carlo (plugin không được cài đặt)"
            )
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký thuật toán Monte Carlo: {str(e)}")

    def register_implementer(self, implementer: DoseCalculationImplementer):
        """
        Đăng ký một implementer mới.

        Args:
            implementer: Đối tượng implementer cần đăng ký
        """
        if not implementer:
            logger.error("Không thể đăng ký implementer None")
            return

        try:
            supported_algorithms = implementer.supported_algorithms()
            if not supported_algorithms:
                logger.warning(
                    f"Implementer {implementer.__class__.__name__} không hỗ trợ thuật toán nào"
                )
                return

            for algorithm in supported_algorithms:
                if algorithm not in self.implementers:
                    self.implementers[algorithm] = []

                # Kiểm tra xem implementer đã được đăng ký chưa
                already_registered = False
                for existing_impl in self.implementers[algorithm]:
                    if existing_impl.__class__ == implementer.__class__:
                        already_registered = True
                        break

                if not already_registered:
                    self.implementers[algorithm].append(implementer)
                    logger.info(f"Đã đăng ký thuật toán {algorithm}")
                else:
                    logger.debug(f"Thuật toán {algorithm} đã được đăng ký trước đó")

            # Thiết lập thuật toán hiện tại nếu chưa có
            if self.current_algorithm is None and supported_algorithms:
                self.current_algorithm = supported_algorithms[0]
                logger.info(
                    f"Đã thiết lập thuật toán mặc định: {self.current_algorithm}"
                )

        except Exception as e:
            logger.error(f"Lỗi khi đăng ký implementer: {str(e)}")
            logger.debug(traceback.format_exc())

    def set_algorithm(self, algorithm: DoseCalculationAlgorithm) -> bool:
        """
        Thiết lập thuật toán tính toán liều.

        Args:
            algorithm: Thuật toán cần sử dụng

        Returns:
            bool: True nếu thiết lập thành công, False nếu không
        """
        if algorithm not in self.implementers:
            logger.error(f"Thuật toán {algorithm} không được hỗ trợ")
            return False

        if not self.implementers[algorithm]:
            logger.error(f"Không có implementer nào cho thuật toán {algorithm}")
            return False

        self.current_algorithm = algorithm
        logger.info(f"Đã thiết lập thuật toán: {algorithm}")
        return True

    def get_available_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Lấy danh sách các thuật toán khả dụng.

        Returns:
            List[DoseCalculationAlgorithm]: Danh sách các thuật toán
        """
        return list(self.implementers.keys())

    def get_current_algorithm(self) -> Optional[DoseCalculationAlgorithm]:
        """
        Lấy thuật toán hiện tại.

        Returns:
            Optional[DoseCalculationAlgorithm]: Thuật toán hiện tại hoặc None
        """
        return self.current_algorithm

    def _get_current_implementer(self) -> Optional[DoseCalculationImplementer]:
        """
        Lấy implementer hiện tại dựa trên thuật toán đã chọn.

        Returns:
            Optional[DoseCalculationImplementer]: Implementer hiện tại hoặc None
        """
        if (
            self.current_algorithm is None
            or self.current_algorithm not in self.implementers
        ):
            logger.error("Không có thuật toán nào được thiết lập")
            return None

        implementers = self.implementers[self.current_algorithm]
        if not implementers:
            logger.error(
                f"Không có implementer nào cho thuật toán {self.current_algorithm}"
            )
            return None

        # Trả về implementer đầu tiên (có thể thêm logic chọn implementer tốt nhất sau này)
        return implementers[0]

    def set_status_callback(self, callback):
        """
        Thiết lập callback để báo cáo tiến độ tính toán.

        Args:
            callback: Hàm callback nhận tham số là tỷ lệ hoàn thành (0-1) và mô tả
        """
        self.status_callback = callback

        # Đồng bộ callback với các implementer
        for algorithm, implementers in self.implementers.items():
            for implementer in implementers:
                if hasattr(implementer, "register_status_callback"):
                    try:
                        implementer.register_status_callback(callback)
                    except Exception as e:
                        logger.warning(
                            f"Không thể đăng ký callback với implementer: {str(e)}"
                        )
                elif hasattr(implementer, "set_status_callback"):
                    try:
                        implementer.set_status_callback(callback)
                    except Exception as e:
                        logger.warning(
                            f"Không thể thiết lập callback cho implementer: {str(e)}"
                        )

    def calculate_async(
        self,
        patient_data: Dict[str, Any],
        beam_data: Dict[str, Any],
        callback=None,
        calculation_options: Dict[str, Any] = None,
    ):
        """
        Tính toán liều bất đồng bộ.

        Args:
            patient_data: Dữ liệu bệnh nhân
            beam_data: Dữ liệu chùm tia
            callback: Hàm callback được gọi khi tính toán hoàn thành
            calculation_options: Tùy chọn tính toán

        Returns:
            threading.Thread: Luồng tính toán
        """
        if self.calculation_in_progress:
            logger.warning("Đang có phiên tính toán khác đang chạy")
            return None

        self.calculation_in_progress = True
        self.calculation_cancelled = False

        def calculation_thread():
            result = None
            error = None

            try:
                result = self.calculate(
                    patient_data, beam_data, calculation_options=calculation_options
                )
            except Exception as e:
                error = e
                logger.error(f"Lỗi trong tính toán bất đồng bộ: {str(e)}")
                logger.debug(traceback.format_exc())

            self.calculation_in_progress = False

            if callback and not self.calculation_cancelled:
                try:
                    callback(result, error)
                except Exception as e:
                    logger.error(f"Lỗi trong callback: {str(e)}")

        # Khởi tạo và bắt đầu luồng
        self.calculation_thread = threading.Thread(target=calculation_thread)
        self.calculation_thread.daemon = True
        self.calculation_thread.start()

        return self.calculation_thread

    def cancel_calculation(self):
        """Hủy phiên tính toán hiện tại."""
        if (
            self.calculation_in_progress
            and self.calculation_thread
            and self.calculation_thread.is_alive()
        ):
            logger.info("Đang hủy phiên tính toán...")
            self.calculation_cancelled = True
            # Không thể thực sự hủy luồng trong Python
            # Chỉ có thể đánh dấu và chờ luồng tự kết thúc
            self.calculation_thread.join(0.1)  # Không chờ đợi quá lâu
            return True
        return False

    def is_calculation_in_progress(self) -> bool:
        """
        Kiểm tra xem có phiên tính toán nào đang chạy không.

        Returns:
            bool: True nếu đang tính toán, False nếu không
        """
        return self.calculation_in_progress

    def calculate(
        self,
        patient_data: Dict[str, Any],
        beam_data: Dict[str, Any],
        calculation_options: Dict[str, Any] = None,
    ) -> Optional[DoseGrid]:
        """
        Tính toán phân bố liều.

        Args:
            patient_data: Dữ liệu bệnh nhân
            beam_data: Dữ liệu chùm tia
            calculation_options: Tùy chọn tính toán

        Returns:
            Optional[DoseGrid]: Đối tượng DoseGrid hoặc None nếu có lỗi

        Raises:
            ValidationError: Nếu dữ liệu đầu vào không hợp lệ
            AlgorithmError: Nếu có lỗi trong quá trình tính toán
        """
        start_time = time.time()

        # Trích xuất implementer
        implementer = self._get_current_implementer()
        if implementer is None:
            raise ValidationError("Không tìm thấy implementer phù hợp")

        # Thiết lập callback nếu có
        if hasattr(implementer, "register_status_callback") and self.status_callback:
            implementer.register_status_callback(self.status_callback)
        elif hasattr(implementer, "set_status_callback") and self.status_callback:
            implementer.set_status_callback(self.status_callback)

        # Báo cáo trạng thái ban đầu
        if self.status_callback:
            algorithm_name = (
                self.current_algorithm.name
                if hasattr(self.current_algorithm, "name")
                else str(self.current_algorithm)
            )
            self.status_callback(
                0.0, f"Bắt đầu tính toán với thuật toán {algorithm_name}"
            )

        # Kiểm tra hợp lệ của dữ liệu đầu vào
        self._validate_input(patient_data, beam_data)

        try:
            # Thực hiện tính toán
            if self.status_callback:
                self.status_callback(0.05, "Đang chuẩn bị dữ liệu")

            # Gọi phương thức calculate của implementer
            dose_grid = implementer.calculate(
                patient_data=patient_data,
                beam_data=beam_data,
                calculation_options=calculation_options,
            )

            # Báo cáo kết quả
            end_time = time.time()
            calculation_time = end_time - start_time
            logger.info(f"Tính toán liều hoàn thành trong {calculation_time:.2f} giây")

            if self.status_callback:
                self.status_callback(1.0, "Tính toán liều hoàn thành")

            return dose_grid

        except ValidationError as e:
            if self.status_callback:
                self.status_callback(1.0, f"Lỗi: {str(e)}")
            logger.error(f"Lỗi xác thực dữ liệu: {str(e)}")
            raise

        except AlgorithmError as e:
            if self.status_callback:
                self.status_callback(1.0, f"Lỗi thuật toán: {str(e)}")
            logger.error(f"Lỗi thuật toán: {str(e)}")
            raise

        except Exception as e:
            if self.status_callback:
                self.status_callback(1.0, f"Lỗi không xác định: {str(e)}")
            logger.error(f"Lỗi không xác định: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AlgorithmError(f"Lỗi không xác định: {str(e)}")

    def _validate_input(self, patient_data: Dict[str, Any], beam_data: Dict[str, Any]):
        """
        Kiểm tra tính hợp lệ của dữ liệu đầu vào.

        Args:
            patient_data: Dữ liệu bệnh nhân
            beam_data: Dữ liệu chùm tia

        Raises:
            ValidationError: Nếu dữ liệu không hợp lệ
        """
        # Kiểm tra dữ liệu bệnh nhân
        if not patient_data:
            raise ValidationError("Dữ liệu bệnh nhân trống")

        # Kiểm tra dữ liệu CT
        if "ct_data" not in patient_data:
            raise ValidationError("Thiếu dữ liệu CT trong patient_data")

        ct_data = patient_data["ct_data"]
        if not isinstance(ct_data, np.ndarray) and not (
            HAS_SITK and isinstance(ct_data, sitk.Image)
        ):
            raise ValidationError("Dữ liệu CT phải là ndarray hoặc SimpleITK.Image")

        # Kiểm tra dữ liệu chùm tia
        if not beam_data:
            raise ValidationError("Dữ liệu chùm tia trống")

        # Kiểm tra các trường bắt buộc trong beam_data
        required_fields = [
            "energy",
            "fluence",
            "gantry_angle",
            "collimator_angle",
            "couch_angle",
        ]
        missing_fields = [field for field in required_fields if field not in beam_data]

        if missing_fields:
            raise ValidationError(
                f"Thiếu các trường sau trong beam_data: {', '.join(missing_fields)}"
            )

        # Kiểm tra thông tin vị trí và kích thước
        if "spacing" not in patient_data:
            logger.warning(
                "Thiếu thông tin spacing trong patient_data, sẽ sử dụng giá trị mặc định (1,1,1)"
            )
            patient_data["spacing"] = (1.0, 1.0, 1.0)

        if "origin" not in patient_data:
            logger.warning(
                "Thiếu thông tin origin trong patient_data, sẽ sử dụng giá trị mặc định (0,0,0)"
            )
            patient_data["origin"] = (0.0, 0.0, 0.0)


# Tạo một instance toàn cục để sử dụng trong toàn bộ hệ thống
default_engine = DoseCalculationEngine()
