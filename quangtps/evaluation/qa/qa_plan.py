#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo và quản lý kế hoạch QA.

Module này cung cấp các công cụ để tạo kế hoạch QA từ kế hoạch điều trị lâm sàng,
hỗ trợ nhiều loại phantom khác nhau và phân tích kết quả QA.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import copy
import datetime

from quangtps.planning.plan import Plan
from quangtps.structures.roi import ROI
from quangtps.core.patient.patient import Patient
from quangtps.dose.dose_grid import DoseGrid
from quangtps.structures.structure_set import StructureSet
from quangtps.evaluation.qa.phantom import PhantomLibrary, Phantom
from quangtps.evaluation.qa.dose_comparison import DoseComparison
from quangtps.evaluation.metrics.gamma_index import GammaIndex, GammaParameters

logger = logging.getLogger(__name__)


class QAPlanType:
    """Loại kế hoạch QA."""

    PHANTOM = "phantom"  # QA trên phantom
    HYBRID = "hybrid"  # QA hybrid (dose on patient CT + measurement)
    MACHINE = "machine"  # QA máy điều trị (không có bệnh nhân)


class QAPlan:
    """
    Lớp biểu diễn kế hoạch QA.

    Lớp này cung cấp các công cụ để tạo, quản lý và phân tích kế hoạch QA
    từ kế hoạch điều trị lâm sàng.
    """

    def __init__(
        self,
        name: str,
        qa_type: str = QAPlanType.PHANTOM,
        clinical_plan: Optional[Plan] = None,
        patient: Optional[Patient] = None,
        phantom: Optional[Phantom] = None,
    ):
        """
        Khởi tạo kế hoạch QA.

        Parameters
        ----------
        name : str
            Tên kế hoạch QA
        qa_type : str, optional
            Loại kế hoạch QA, mặc định là QAPlanType.PHANTOM
        clinical_plan : Optional[Plan], optional
            Kế hoạch lâm sàng gốc, mặc định là None
        patient : Optional[Patient], optional
            Bệnh nhân, mặc định là None
        phantom : Optional[Phantom], optional
            Phantom sử dụng cho QA, mặc định là None
        """
        self.name = name
        self.qa_type = qa_type
        self.clinical_plan = clinical_plan
        self.patient = patient
        self.phantom = phantom

        # Kế hoạch QA (sẽ được tạo dựa trên kế hoạch lâm sàng)
        self.plan = None

        # Dữ liệu QA
        self.measured_dose = None  # Liều đo được
        self.calculated_dose = None  # Liều tính toán
        self.analysis_results = {}  # Kết quả phân tích

        # Thông tin QA
        self.qa_date = None
        self.qa_device = None
        self.qa_operator = None
        self.qa_notes = ""
        self.qa_status = "Chưa thực hiện"

        # Lịch sử QA
        self.qa_history = []

    def create_phantom_plan(
        self,
        phantom: Optional[Phantom] = None,
        copy_beams: bool = True,
        adjust_for_phantom: bool = True,
        beam_mapping: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Tạo kế hoạch QA trên phantom từ kế hoạch lâm sàng.

        Parameters
        ----------
        phantom : Optional[Phantom], optional
            Phantom sử dụng, mặc định là None (sử dụng phantom đã cài đặt)
        copy_beams : bool, optional
            Sao chép cấu hình chùm tia, mặc định là True
        adjust_for_phantom : bool, optional
            Điều chỉnh chùm tia cho phantom, mặc định là True
        beam_mapping : Optional[Dict[str, str]], optional
            Ánh xạ giữa chùm tia lâm sàng và chùm tia phantom, mặc định là None

        Returns
        -------
        bool
            True nếu tạo thành công, False nếu không
        """
        if not self.clinical_plan:
            logger.error("Không thể tạo kế hoạch QA: Không có kế hoạch lâm sàng")
            return False

        if phantom:
            self.phantom = phantom
        elif not self.phantom:
            # Sử dụng phantom mặc định nếu không được chỉ định
            default_phantom = PhantomLibrary.get_default_phantom()
            if not default_phantom:
                logger.error("Không thể tạo kế hoạch QA: Không có phantom")
                return False
            self.phantom = default_phantom

        try:
            # Tạo kế hoạch mới
            self.plan = Plan(
                name=f"QA_{self.clinical_plan.name}",
                patient=self.phantom.patient
                if hasattr(self.phantom, "patient")
                else None,
            )

            # Sao chép các thuộc tính cơ bản
            self.plan.technique = self.clinical_plan.technique
            self.plan.modality = self.clinical_plan.modality
            self.plan.prescription = copy.deepcopy(self.clinical_plan.prescription)

            # Sao chép và điều chỉnh chùm tia nếu cần
            if copy_beams and hasattr(self.clinical_plan, "beams"):
                for i, beam in enumerate(self.clinical_plan.beams):
                    # Sao chép chùm tia
                    new_beam = copy.deepcopy(beam)

                    # Điều chỉnh cho phantom nếu cần
                    if adjust_for_phantom:
                        # Điều chỉnh isocentre để trỏ vào trung tâm phantom
                        if hasattr(self.phantom, "center"):
                            new_beam.isocenter = self.phantom.center

                        # Các điều chỉnh khác cho phantom...

                    # Thêm chùm tia vào kế hoạch QA
                    self.plan.add_beam(new_beam)

            # Thiết lập cấu trúc QA từ phantom
            if hasattr(self.phantom, "structures"):
                self.plan.structures = self.phantom.structures

            logger.info(f"Đã tạo kế hoạch QA trên phantom: {self.name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch QA trên phantom: {e}")
            import traceback

            traceback.print_exc()
            return False

    def import_measurement(
        self,
        measured_data_path: str,
        data_format: str = "dicom",
        qa_device: Optional[str] = None,
        qa_date: Optional[datetime.datetime] = None,
        qa_operator: Optional[str] = None,
    ) -> bool:
        """
        Nhập dữ liệu đo từ file.

        Parameters
        ----------
        measured_data_path : str
            Đường dẫn đến file dữ liệu đo
        data_format : str, optional
            Định dạng dữ liệu, mặc định là "dicom"
        qa_device : Optional[str], optional
            Thiết bị QA, mặc định là None
        qa_date : Optional[datetime.datetime], optional
            Ngày thực hiện QA, mặc định là None (sử dụng ngày hiện tại)
        qa_operator : Optional[str], optional
            Người thực hiện QA, mặc định là None

        Returns
        -------
        bool
            True nếu nhập thành công, False nếu không
        """
        try:
            # Kiểm tra sự tồn tại của file
            if not os.path.exists(measured_data_path):
                logger.error(f"Không tìm thấy file dữ liệu đo: {measured_data_path}")
                return False

            # Xác định định dạng và nhập dữ liệu
            if data_format.lower() == "dicom":
                # Nhập dữ liệu từ DICOM
                from quangtps.dicom.dicom_reader import DicomReader

                reader = DicomReader()
                dose_data = reader.read_dose_file(measured_data_path)

                if dose_data is None:
                    logger.error(f"Không thể đọc file DICOM: {measured_data_path}")
                    return False

                self.measured_dose = dose_data

            elif data_format.lower() == "csv":
                # Nhập dữ liệu từ CSV
                import pandas as pd

                try:
                    df = pd.read_csv(measured_data_path)
                    # Xử lý dữ liệu CSV và chuyển thành DoseGrid
                    # ...
                    self.measured_dose = self._convert_csv_to_dose(df)
                except Exception as e:
                    logger.error(f"Lỗi khi đọc file CSV: {e}")
                    return False

            else:
                logger.error(f"Định dạng không được hỗ trợ: {data_format}")
                return False

            # Cập nhật thông tin QA
            self.qa_device = qa_device
            self.qa_date = qa_date if qa_date else datetime.datetime.now()
            self.qa_operator = qa_operator

            # Cập nhật trạng thái
            self.qa_status = "Đã nhập dữ liệu đo"

            # Thêm vào lịch sử
            self._add_to_history(
                "Nhập dữ liệu đo",
                {
                    "file_path": measured_data_path,
                    "format": data_format,
                    "device": qa_device,
                    "date": self.qa_date,
                    "operator": qa_operator,
                },
            )

            logger.info(f"Đã nhập dữ liệu đo từ: {measured_data_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi nhập dữ liệu đo: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _convert_csv_to_dose(self, df):
        """
        Chuyển đổi dữ liệu CSV thành đối tượng DoseGrid.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame chứa dữ liệu liều đo được

        Returns
        -------
        DoseGrid
            Đối tượng DoseGrid chứa dữ liệu liều
        """
        # TODO: Triển khai chuyển đổi dữ liệu CSV sang DoseGrid
        # Đây là phương thức giả định cần được triển khai dựa trên
        # cấu trúc thực tế của dữ liệu CSV
        return None

    def calculate_qa_dose(self) -> bool:
        """
        Tính toán liều cho kế hoạch QA.

        Returns
        -------
        bool
            True nếu tính toán thành công, False nếu không
        """
        if not self.plan:
            logger.error("Không thể tính toán liều: Chưa có kế hoạch QA")
            return False

        try:
            # Tính toán liều
            from quangtps.dose.dose_calculation import DoseCalculator

            calculator = DoseCalculator()
            result = calculator.calculate_dose(self.plan)

            if not result:
                logger.error("Không thể tính toán liều cho kế hoạch QA")
                return False

            self.calculated_dose = result.dose_grid

            # Cập nhật trạng thái
            self.qa_status = "Đã tính toán liều"

            # Thêm vào lịch sử
            self._add_to_history(
                "Tính toán liều QA",
                {
                    "algorithm": calculator.algorithm
                    if hasattr(calculator, "algorithm")
                    else "unknown",
                    "timestamp": datetime.datetime.now(),
                },
            )

            logger.info(f"Đã tính toán liều cho kế hoạch QA: {self.name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều cho kế hoạch QA: {e}")
            import traceback

            traceback.print_exc()
            return False

    def analyze_qa_results(
        self,
        methods: Optional[List[str]] = None,
        gamma_params: Optional[GammaParameters] = None,
    ) -> Dict[str, Any]:
        """
        Phân tích kết quả QA bằng cách so sánh liều tính toán và đo được.

        Parameters
        ----------
        methods : Optional[List[str]], optional
            Danh sách phương pháp phân tích, mặc định là None (tất cả phương pháp)
        gamma_params : Optional[GammaParameters], optional
            Tham số cho phân tích gamma, mặc định là None (sử dụng tham số mặc định)

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích
        """
        if not self.calculated_dose or not self.measured_dose:
            logger.error("Không thể phân tích kết quả: Thiếu dữ liệu liều")
            return {}

        # Nếu không chỉ định phương pháp, sử dụng tất cả
        if not methods:
            methods = ["gamma", "difference", "dta", "profiles"]

        results = {}

        try:
            # Khởi tạo đối tượng so sánh liều
            comparison = DoseComparison(self.calculated_dose, self.measured_dose)

            # Phân tích gamma
            if "gamma" in methods:
                gamma_params = gamma_params or GammaParameters()
                gamma_result = comparison.calculate_gamma_index(gamma_params)

                results["gamma"] = {
                    "passing_rate": gamma_result.passing_rate,
                    "gamma_map": gamma_result.gamma_map,
                    "parameters": gamma_params.__dict__,
                    "max_gamma": gamma_result.max_gamma,
                    "mean_gamma": gamma_result.mean_gamma,
                }

            # Phân tích sự khác biệt liều
            if "difference" in methods:
                diff_result = comparison.calculate_dose_difference()

                results["difference"] = {
                    "mean_difference": diff_result.mean_difference,
                    "max_difference": diff_result.max_difference,
                    "difference_map": diff_result.difference_map,
                    "histogram": diff_result.histogram,
                }

            # Phân tích DTA (Distance To Agreement)
            if "dta" in methods:
                dta_result = comparison.calculate_dta()

                results["dta"] = {
                    "mean_dta": dta_result.mean_dta,
                    "max_dta": dta_result.max_dta,
                    "dta_map": dta_result.dta_map,
                    "passing_rate": dta_result.passing_rate,
                }

            # Phân tích profile
            if "profiles" in methods:
                profile_results = comparison.extract_dose_profiles()

                results["profiles"] = profile_results

            # Lưu kết quả phân tích
            self.analysis_results = results

            # Cập nhật trạng thái
            self.qa_status = "Đã phân tích kết quả"

            # Thêm vào lịch sử
            self._add_to_history(
                "Phân tích kết quả QA",
                {
                    "methods": methods,
                    "timestamp": datetime.datetime.now(),
                    "results_summary": {
                        k: {
                            key: val
                            for key, val in v.items()
                            if key
                            not in [
                                "gamma_map",
                                "difference_map",
                                "dta_map",
                                "histogram",
                            ]
                        }
                        for k, v in results.items()
                    },
                },
            )

            logger.info(f"Đã phân tích kết quả QA cho kế hoạch: {self.name}")

            return results

        except Exception as e:
            logger.error(f"Lỗi khi phân tích kết quả QA: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def generate_qa_report(self, output_path: Optional[str] = None) -> str:
        """
        Tạo báo cáo QA từ kết quả phân tích.

        Parameters
        ----------
        output_path : Optional[str], optional
            Đường dẫn lưu báo cáo, mặc định là None (tự động tạo)

        Returns
        -------
        str
            Đường dẫn đến báo cáo QA
        """
        if not self.analysis_results:
            logger.error("Không thể tạo báo cáo: Chưa có kết quả phân tích")
            return ""

        try:
            from quangtps.reporting.qa_report_generator import QAReportGenerator

            # Tạo đường dẫn mặc định nếu không được chỉ định
            if not output_path:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"qa_report_{self.name}_{timestamp}.pdf"

            # Tạo báo cáo
            generator = QAReportGenerator()
            report_path = generator.generate_report(
                qa_plan=self,
                output_path=output_path,
                include_images=True,
                include_profiles=True,
                include_histograms=True,
            )

            # Thêm vào lịch sử
            self._add_to_history(
                "Tạo báo cáo QA",
                {
                    "report_path": report_path,
                    "timestamp": datetime.datetime.now(),
                },
            )

            logger.info(f"Đã tạo báo cáo QA: {report_path}")

            return report_path

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo QA: {e}")
            import traceback

            traceback.print_exc()
            return ""

    def _add_to_history(self, action: str, details: Dict[str, Any]) -> None:
        """
        Thêm một mục vào lịch sử QA.

        Parameters
        ----------
        action : str
            Hành động thực hiện
        details : Dict[str, Any]
            Chi tiết về hành động
        """
        history_entry = {
            "timestamp": datetime.datetime.now(),
            "action": action,
            "details": details,
        }

        self.qa_history.append(history_entry)

    def save(self, file_path: str) -> bool:
        """
        Lưu kế hoạch QA vào file.

        Parameters
        ----------
        file_path : str
            Đường dẫn file lưu

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không
        """
        try:
            import pickle

            with open(file_path, "wb") as f:
                pickle.dump(self, f)

            logger.info(f"Đã lưu kế hoạch QA vào: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi lưu kế hoạch QA: {e}")
            return False

    @classmethod
    def load(cls, file_path: str) -> Optional["QAPlan"]:
        """
        Tải kế hoạch QA từ file.

        Parameters
        ----------
        file_path : str
            Đường dẫn file

        Returns
        -------
        Optional[QAPlan]
            Kế hoạch QA nếu tải thành công, None nếu không
        """
        try:
            import pickle

            with open(file_path, "rb") as f:
                qa_plan = pickle.load(f)

            logger.info(f"Đã tải kế hoạch QA từ: {file_path}")
            return qa_plan

        except Exception as e:
            logger.error(f"Lỗi khi tải kế hoạch QA: {e}")
            return None
