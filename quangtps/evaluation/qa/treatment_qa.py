#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý đảm bảo chất lượng điều trị (Treatment Quality Assurance).

Module này cung cấp các lớp và phương thức để đảm bảo chất lượng và tính an toàn
của các kế hoạch điều trị xạ trị, bao gồm kiểm tra tính hợp lệ, kiểm tra độ chính xác
của liều lượng, và đánh giá chất lượng kế hoạch.
"""

import logging
import uuid
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

from quangtps.treatment.plan import TreatmentPlan
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.stereotactic import SRS, SBRT
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)


class QATestType(str, Enum):
    """Enum đại diện cho các loại kiểm tra QA."""
    PRE_TREATMENT = "Pre-Treatment"
    IN_VIVO = "In-Vivo"
    POST_TREATMENT = "Post-Treatment"
    MACHINE_QA = "Machine QA"
    PATIENT_SPECIFIC = "Patient-Specific"


class QAStatus(str, Enum):
    """Enum đại diện cho trạng thái kiểm tra QA."""
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    PASSED = "Passed"
    FAILED = "Failed"
    REVIEW_REQUIRED = "Review Required"


class QAProtocol(str, Enum):
    """Enum đại diện cho các giao thức QA phổ biến."""
    AAPM_TG119 = "AAPM TG-119"
    AAPM_TG142 = "AAPM TG-142"
    AAPM_TG218 = "AAPM TG-218"
    ICRU_83 = "ICRU 83"
    ASTRO_GUIDELINES = "ASTRO Guidelines"
    CUSTOM = "Custom"


class MetricResult:
    """
    Lớp đại diện cho kết quả của một chỉ số đánh giá.
    
    Lớp này chứa thông tin về giá trị đo được, giá trị tham chiếu,
    và đánh giá tính chấp nhận được của chỉ số.
    """
    
    def __init__(
        self,
        name: str,
        value: float,
        reference: float,
        tolerance: float,
        unit: str = "",
        description: str = ""
    ):
        """
        Khởi tạo một kết quả chỉ số đánh giá.
        
        Parameters
        ----------
        name : str
            Tên chỉ số
        value : float
            Giá trị đo được
        reference : float
            Giá trị tham chiếu
        tolerance : float
            Dung sai cho phép
        unit : str, optional
            Đơn vị đo
        description : str, optional
            Mô tả chỉ số
        """
        self.name = name
        self.value = value
        self.reference = reference
        self.tolerance = tolerance
        self.unit = unit
        self.description = description
        
        # Tính sai số
        self.error = value - reference
        self.percent_error = (self.error / reference) * 100 if reference != 0 else float('inf')
        
        # Đánh giá tính chấp nhận được
        self.is_acceptable = abs(self.error) <= tolerance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi kết quả thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kết quả
        """
        return {
            "name": self.name,
            "value": self.value,
            "reference": self.reference,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "description": self.description,
            "error": self.error,
            "percent_error": self.percent_error,
            "is_acceptable": self.is_acceptable
        }


class TreatmentQATest:
    """
    Lớp đại diện cho một bài kiểm tra QA điều trị.
    
    Lớp này chứa thông tin về một bài kiểm tra QA cụ thể, bao gồm
    loại kiểm tra, kế hoạch điều trị liên quan, và kết quả kiểm tra.
    """
    
    def __init__(
        self,
        test_name: str,
        test_type: QATestType,
        protocol: QAProtocol,
        plan_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        machine_id: Optional[str] = None,
        description: str = ""
    ):
        """
        Khởi tạo một bài kiểm tra QA.
        
        Parameters
        ----------
        test_name : str
            Tên bài kiểm tra
        test_type : QATestType
            Loại kiểm tra
        protocol : QAProtocol
            Giao thức QA
        plan_id : str, optional
            ID của kế hoạch điều trị liên quan
        patient_id : str, optional
            ID của bệnh nhân
        machine_id : str, optional
            ID của máy xạ trị
        description : str, optional
            Mô tả bài kiểm tra
        """
        self.test_id = str(uuid.uuid4())
        self.test_name = test_name
        self.test_type = test_type
        self.protocol = protocol
        self.plan_id = plan_id
        self.patient_id = patient_id
        self.machine_id = machine_id
        self.description = description
        
        self.created_date = datetime.datetime.now()
        self.scheduled_date = None
        self.performed_date = None
        self.status = QAStatus.PENDING
        
        self.metrics: List[MetricResult] = []
        self.overall_result: Optional[bool] = None
        self.notes = ""
        self.performed_by = ""
        self.reviewer = ""
        self.reviewed_date = None
        
        self.metadata: Dict[str, Any] = {}
    
    def add_metric(self, metric: MetricResult) -> None:
        """
        Thêm một chỉ số đánh giá vào bài kiểm tra.
        
        Parameters
        ----------
        metric : MetricResult
            Chỉ số đánh giá
        """
        self.metrics.append(metric)
    
    def evaluate(self) -> bool:
        """
        Đánh giá kết quả tổng thể của bài kiểm tra.
        
        Returns
        -------
        bool
            True nếu tất cả các chỉ số đều chấp nhận được, False nếu không
        """
        if not self.metrics:
            return False
        
        self.overall_result = all(metric.is_acceptable for metric in self.metrics)
        self.status = QAStatus.PASSED if self.overall_result else QAStatus.FAILED
        
        return self.overall_result
    
    def set_status(self, status: QAStatus, notes: str = "") -> None:
        """
        Thiết lập trạng thái của bài kiểm tra.
        
        Parameters
        ----------
        status : QAStatus
            Trạng thái mới
        notes : str, optional
            Ghi chú
        """
        self.status = status
        if notes:
            self.notes += f"\n{datetime.datetime.now().isoformat()}: {notes}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin bài kiểm tra thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin bài kiểm tra
        """
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "test_type": self.test_type,
            "protocol": self.protocol,
            "plan_id": self.plan_id,
            "patient_id": self.patient_id,
            "machine_id": self.machine_id,
            "description": self.description,
            "created_date": self.created_date.isoformat(),
            "scheduled_date": self.scheduled_date.isoformat() if self.scheduled_date else None,
            "performed_date": self.performed_date.isoformat() if self.performed_date else None,
            "status": self.status,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "overall_result": self.overall_result,
            "notes": self.notes,
            "performed_by": self.performed_by,
            "reviewer": self.reviewer,
            "reviewed_date": self.reviewed_date.isoformat() if self.reviewed_date else None,
            "metadata": self.metadata
        }


class TreatmentQAManager:
    """
    Lớp quản lý đảm bảo chất lượng điều trị.
    
    Lớp này cung cấp các phương thức để tạo, quản lý, và đánh giá
    các bài kiểm tra QA cho các kế hoạch điều trị xạ trị.
    """
    
    def __init__(self):
        """Khởi tạo trình quản lý QA."""
        self.qa_tests: Dict[str, TreatmentQATest] = {}
        self.plan_qa_map: Dict[str, List[str]] = {}  # plan_id -> list of test_ids
        self.patient_qa_map: Dict[str, List[str]] = {}  # patient_id -> list of test_ids
        self.machine_qa_map: Dict[str, List[str]] = {}  # machine_id -> list of test_ids
    
    def create_test(
        self,
        test_name: str,
        test_type: QATestType,
        protocol: QAProtocol,
        plan_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        machine_id: Optional[str] = None,
        description: str = ""
    ) -> str:
        """
        Tạo một bài kiểm tra QA mới.
        
        Parameters
        ----------
        test_name : str
            Tên bài kiểm tra
        test_type : QATestType
            Loại kiểm tra
        protocol : QAProtocol
            Giao thức QA
        plan_id : str, optional
            ID của kế hoạch điều trị liên quan
        patient_id : str, optional
            ID của bệnh nhân
        machine_id : str, optional
            ID của máy xạ trị
        description : str, optional
            Mô tả bài kiểm tra
            
        Returns
        -------
        str
            ID của bài kiểm tra
        """
        test = TreatmentQATest(
            test_name=test_name,
            test_type=test_type,
            protocol=protocol,
            plan_id=plan_id,
            patient_id=patient_id,
            machine_id=machine_id,
            description=description
        )
        
        self.qa_tests[test.test_id] = test
        
        # Cập nhật các bản đồ tham chiếu
        if plan_id:
            if plan_id not in self.plan_qa_map:
                self.plan_qa_map[plan_id] = []
            self.plan_qa_map[plan_id].append(test.test_id)
        
        if patient_id:
            if patient_id not in self.patient_qa_map:
                self.patient_qa_map[patient_id] = []
            self.patient_qa_map[patient_id].append(test.test_id)
        
        if machine_id:
            if machine_id not in self.machine_qa_map:
                self.machine_qa_map[machine_id] = []
            self.machine_qa_map[machine_id].append(test.test_id)
        
        logger.info(f"Created QA test: {test_name} (ID: {test.test_id})")
        
        return test.test_id
    
    def get_test(self, test_id: str) -> Optional[TreatmentQATest]:
        """
        Lấy thông tin về một bài kiểm tra QA.
        
        Parameters
        ----------
        test_id : str
            ID của bài kiểm tra
            
        Returns
        -------
        Optional[TreatmentQATest]
            Bài kiểm tra nếu tồn tại, None nếu không
        """
        return self.qa_tests.get(test_id)
    
    def update_test(self, test_id: str, **kwargs) -> bool:
        """
        Cập nhật thông tin của một bài kiểm tra QA.
        
        Parameters
        ----------
        test_id : str
            ID của bài kiểm tra
        **kwargs : dict
            Các thông số cần cập nhật
            
        Returns
        -------
        bool
            True nếu cập nhật thành công, False nếu không
        """
        test = self.get_test(test_id)
        if not test:
            return False
        
        # Cập nhật các thuộc tính
        for key, value in kwargs.items():
            if hasattr(test, key):
                setattr(test, key, value)
        
        return True
    
    def add_metric_to_test(
        self,
        test_id: str,
        metric_name: str,
        value: float,
        reference: float,
        tolerance: float,
        unit: str = "",
        description: str = ""
    ) -> bool:
        """
        Thêm một chỉ số đánh giá vào bài kiểm tra.
        
        Parameters
        ----------
        test_id : str
            ID của bài kiểm tra
        metric_name : str
            Tên chỉ số
        value : float
            Giá trị đo được
        reference : float
            Giá trị tham chiếu
        tolerance : float
            Dung sai cho phép
        unit : str, optional
            Đơn vị đo
        description : str, optional
            Mô tả chỉ số
            
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu không
        """
        test = self.get_test(test_id)
        if not test:
            return False
        
        metric = MetricResult(
            name=metric_name,
            value=value,
            reference=reference,
            tolerance=tolerance,
            unit=unit,
            description=description
        )
        
        test.add_metric(metric)
        return True
    
    def evaluate_test(self, test_id: str) -> Optional[bool]:
        """
        Đánh giá kết quả của một bài kiểm tra.
        
        Parameters
        ----------
        test_id : str
            ID của bài kiểm tra
            
        Returns
        -------
        Optional[bool]
            Kết quả đánh giá nếu thành công, None nếu không
        """
        test = self.get_test(test_id)
        if not test:
            return None
        
        result = test.evaluate()
        return result
    
    def get_plan_qa_tests(self, plan_id: str) -> List[TreatmentQATest]:
        """
        Lấy danh sách các bài kiểm tra QA cho một kế hoạch.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch
            
        Returns
        -------
        List[TreatmentQATest]
            Danh sách các bài kiểm tra
        """
        test_ids = self.plan_qa_map.get(plan_id, [])
        return [self.qa_tests[test_id] for test_id in test_ids if test_id in self.qa_tests]
    
    def get_patient_qa_tests(self, patient_id: str) -> List[TreatmentQATest]:
        """
        Lấy danh sách các bài kiểm tra QA cho một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        List[TreatmentQATest]
            Danh sách các bài kiểm tra
        """
        test_ids = self.patient_qa_map.get(patient_id, [])
        return [self.qa_tests[test_id] for test_id in test_ids if test_id in self.qa_tests]
    
    def get_machine_qa_tests(self, machine_id: str) -> List[TreatmentQATest]:
        """
        Lấy danh sách các bài kiểm tra QA cho một máy xạ trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
            
        Returns
        -------
        List[TreatmentQATest]
            Danh sách các bài kiểm tra
        """
        test_ids = self.machine_qa_map.get(machine_id, [])
        return [self.qa_tests[test_id] for test_id in test_ids if test_id in self.qa_tests]
    
    def create_standard_qa_tests_for_plan(
        self,
        plan: TreatmentPlan,
        technique: Any
    ) -> List[str]:
        """
        Tạo các bài kiểm tra QA tiêu chuẩn cho một kế hoạch điều trị.
        
        Parameters
        ----------
        plan : TreatmentPlan
            Kế hoạch điều trị
        technique : Any
            Kỹ thuật xạ trị
            
        Returns
        -------
        List[str]
            Danh sách các ID của bài kiểm tra đã tạo
        """
        test_ids = []
        
        # Xác định loại kỹ thuật và tạo các bài kiểm tra phù hợp
        if isinstance(technique, IMRT):
            # Bài kiểm tra QA cho IMRT
            test_id = self.create_test(
                test_name=f"IMRT QA - {plan.plan_name}",
                test_type=QATestType.PRE_TREATMENT,
                protocol=QAProtocol.AAPM_TG119,
                plan_id=plan.plan_id,
                patient_id=plan.patient_id,
                machine_id=technique.linac.machine_id if hasattr(technique, 'linac') and technique.linac else None,
                description="Pre-treatment QA for IMRT plan"
            )
            test_ids.append(test_id)
            
            # Thêm các chỉ số đánh giá tiêu chuẩn cho IMRT
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Gamma Pass Rate (3%/3mm)",
                value=0.0,  # Giá trị ban đầu, sẽ được cập nhật sau khi đo
                reference=95.0,
                tolerance=5.0,
                unit="%",
                description="Tỷ lệ điểm vượt qua phân tích gamma với tiêu chí 3%/3mm"
            )
            
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Point Dose Difference",
                value=0.0,
                reference=2.0,
                tolerance=3.0,
                unit="%",
                description="Sai số liều lượng tại điểm tham chiếu"
            )
            
        elif isinstance(technique, VMAT):
            # Bài kiểm tra QA cho VMAT
            test_id = self.create_test(
                test_name=f"VMAT QA - {plan.plan_name}",
                test_type=QATestType.PRE_TREATMENT,
                protocol=QAProtocol.AAPM_TG119,
                plan_id=plan.plan_id,
                patient_id=plan.patient_id,
                machine_id=technique.treatment_machine.machine_id if hasattr(technique, 'treatment_machine') and technique.treatment_machine else None,
                description="Pre-treatment QA for VMAT plan"
            )
            test_ids.append(test_id)
            
            # Thêm các chỉ số đánh giá tiêu chuẩn cho VMAT
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Gamma Pass Rate (3%/3mm)",
                value=0.0,
                reference=95.0,
                tolerance=5.0,
                unit="%",
                description="Tỷ lệ điểm vượt qua phân tích gamma với tiêu chí 3%/3mm"
            )
            
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Point Dose Difference",
                value=0.0,
                reference=2.0,
                tolerance=3.0,
                unit="%",
                description="Sai số liều lượng tại điểm tham chiếu"
            )
            
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Delivery Time",
                value=0.0,
                reference=technique.arcs[0].get("expected_delivery_time", 0.0) if hasattr(technique, 'arcs') and technique.arcs else 0.0,
                tolerance=30.0,
                unit="s",
                description="Thời gian thực hiện điều trị"
            )
            
        elif isinstance(technique, (SRS, SBRT)):
            # Bài kiểm tra QA cho SRS/SBRT
            technique_type = "SRS" if isinstance(technique, SRS) else "SBRT"
            test_id = self.create_test(
                test_name=f"{technique_type} QA - {plan.plan_name}",
                test_type=QATestType.PRE_TREATMENT,
                protocol=QAProtocol.AAPM_TG218,
                plan_id=plan.plan_id,
                patient_id=plan.patient_id,
                machine_id=technique.treatment_machine.machine_id if hasattr(technique, 'treatment_machine') and technique.treatment_machine else None,
                description=f"Pre-treatment QA for {technique_type} plan"
            )
            test_ids.append(test_id)
            
            # Thêm các chỉ số đánh giá tiêu chuẩn cho SRS/SBRT
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Gamma Pass Rate (2%/2mm)",
                value=0.0,
                reference=97.0,
                tolerance=3.0,
                unit="%",
                description="Tỷ lệ điểm vượt qua phân tích gamma với tiêu chí 2%/2mm"
            )
            
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Point Dose Difference",
                value=0.0,
                reference=1.0,
                tolerance=2.0,
                unit="%",
                description="Sai số liều lượng tại điểm tham chiếu"
            )
            
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Isocenter Position",
                value=0.0,
                reference=0.0,
                tolerance=1.0,
                unit="mm",
                description="Sai số vị trí tâm đồng vị"
            )
            
        else:
            # Bài kiểm tra QA chung cho các kỹ thuật khác
            test_id = self.create_test(
                test_name=f"Plan QA - {plan.plan_name}",
                test_type=QATestType.PRE_TREATMENT,
                protocol=QAProtocol.CUSTOM,
                plan_id=plan.plan_id,
                patient_id=plan.patient_id,
                description="Pre-treatment QA for plan"
            )
            test_ids.append(test_id)
            
            # Thêm các chỉ số đánh giá cơ bản
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Point Dose Difference",
                value=0.0,
                reference=3.0,
                tolerance=5.0,
                unit="%",
                description="Sai số liều lượng tại điểm tham chiếu"
            )
        
        # Thêm bài kiểm tra QA IGRT nếu cần
        if hasattr(plan, 'requires_igrt') and plan.requires_igrt:
            test_id = self.create_test(
                test_name=f"IGRT QA - {plan.plan_name}",
                test_type=QATestType.PRE_TREATMENT,
                protocol=QAProtocol.AAPM_TG142,
                plan_id=plan.plan_id,
                patient_id=plan.patient_id,
                description="IGRT verification for treatment plan"
            )
            test_ids.append(test_id)
            
            self.add_metric_to_test(
                test_id=test_id,
                metric_name="Geometric Alignment",
                value=0.0,
                reference=0.0,
                tolerance=1.0,
                unit="mm",
                description="Sai số canh chỉnh hình học"
            )
        
        return test_ids


# Cấu trúc dữ liệu cho báo cáo QA
class TreatmentQAResult:
    """
    Lớp đại diện cho kết quả của một bài kiểm tra QA.
    
    Lớp này chứa thông tin về kết quả của một bài kiểm tra QA,
    bao gồm các chỉ số đo được và đánh giá.
    """
    
    def __init__(self, test_id: str, test_name: str, overall_result: bool, metrics: List[Dict[str, Any]]):
        """
        Khởi tạo kết quả QA.
        
        Parameters
        ----------
        test_id : str
            ID của bài kiểm tra
        test_name : str
            Tên bài kiểm tra
        overall_result : bool
            Kết quả tổng thể (đạt/không đạt)
        metrics : List[Dict[str, Any]]
            Danh sách các chỉ số đánh giá
        """
        self.test_id = test_id
        self.test_name = test_name
        self.overall_result = overall_result
        self.metrics = metrics
        self.timestamp = datetime.datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi kết quả thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kết quả
        """
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "overall_result": self.overall_result,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


class TreatmentQAReport:
    """
    Lớp đại diện cho báo cáo QA của một kế hoạch điều trị.
    
    Lớp này chứa thông tin về các kết quả QA của một kế hoạch điều trị,
    bao gồm các bài kiểm tra và đánh giá tổng thể.
    """
    
    def __init__(self, plan_id: str, plan_name: str):
        """
        Khởi tạo báo cáo QA.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch
        plan_name : str
            Tên kế hoạch
        """
        self.plan_id = plan_id
        self.plan_name = plan_name
        self.results: List[TreatmentQAResult] = []
        self.created_date = datetime.datetime.now()
        self.summary = ""
        self.recommendations = []
        
    def add_result(self, result: TreatmentQAResult) -> None:
        """
        Thêm một kết quả vào báo cáo.
        
        Parameters
        ----------
        result : TreatmentQAResult
            Kết quả QA
        """
        self.results.append(result)
        
    def generate_summary(self) -> str:
        """
        Tạo tóm tắt cho báo cáo.
        
        Returns
        -------
        str
            Tóm tắt báo cáo
        """
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.overall_result)
        
        summary = f"Summary for {self.plan_name}: {passed_tests}/{total_tests} tests passed.\n"
        
        if passed_tests == total_tests:
            summary += "All QA tests have passed. The plan is ready for treatment."
        elif passed_tests / total_tests >= 0.75:
            summary += "Most QA tests have passed. Review the failed tests before proceeding with treatment."
        else:
            summary += "Several QA tests have failed. The plan may need to be modified before treatment."
            
        self.summary = summary
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi báo cáo thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin báo cáo
        """
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "results": [result.to_dict() for result in self.results],
            "created_date": self.created_date.isoformat(),
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class PlanQualityMetrics:
    """
    Lớp đại diện cho các chỉ số chất lượng của kế hoạch điều trị.
    
    Lớp này chứa các chỉ số như chỉ số phù hợp, độ đồng nhất, chỉ số gradient,
    và các chỉ số DVH để đánh giá chất lượng kế hoạch.
    """
    
    def __init__(self, plan_id: str):
        """
        Khởi tạo các chỉ số chất lượng.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch
        """
        self.plan_id = plan_id
        self.conformity_index = 0.0
        self.homogeneity_index = 0.0
        self.gradient_index = 0.0
        self.dose_spillage = 0.0
        self.target_coverage = 0.0
        self.oar_constraints_met = 0.0  # Tỷ lệ các chỉ số giới hạn của cơ quan nguy cấp được đáp ứng
        self.dvh_metrics = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi các chỉ số thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin các chỉ số
        """
        return {
            "plan_id": self.plan_id,
            "conformity_index": self.conformity_index,
            "homogeneity_index": self.homogeneity_index,
            "gradient_index": self.gradient_index,
            "dose_spillage": self.dose_spillage,
            "target_coverage": self.target_coverage,
            "oar_constraints_met": self.oar_constraints_met,
            "dvh_metrics": self.dvh_metrics
        }


# Hàm tiện ích
def perform_treatment_qa(test: TreatmentQATest, measurements: Dict[str, float]) -> TreatmentQAResult:
    """
    Thực hiện kiểm tra QA và cập nhật kết quả.
    
    Parameters
    ----------
    test : TreatmentQATest
        Bài kiểm tra QA
    measurements : Dict[str, float]
        Các giá trị đo được
        
    Returns
    -------
    TreatmentQAResult
        Kết quả của bài kiểm tra
    """
    # Cập nhật các giá trị đo được cho các chỉ số
    for metric in test.metrics:
        if metric.name in measurements:
            metric.value = measurements[metric.name]
            
    # Đánh giá kết quả
    test.evaluate()
    test.performed_date = datetime.datetime.now()
    
    # Tạo kết quả
    result = TreatmentQAResult(
        test_id=test.test_id,
        test_name=test.test_name,
        overall_result=test.overall_result,
        metrics=[metric.to_dict() for metric in test.metrics]
    )
    
    return result


def evaluate_plan_quality(plan: TreatmentPlan, dose_distribution: np.ndarray, structures: Dict[str, np.ndarray]) -> PlanQualityMetrics:
    """
    Đánh giá chất lượng của kế hoạch điều trị.
    
    Parameters
    ----------
    plan : TreatmentPlan
        Kế hoạch điều trị
    dose_distribution : np.ndarray
        Phân bố liều
    structures : Dict[str, np.ndarray]
        Các cấu trúc giải phẫu
        
    Returns
    -------
    PlanQualityMetrics
        Các chỉ số chất lượng của kế hoạch
    """
    metrics = PlanQualityMetrics(plan.plan_id)
    
    # Tính toán các chỉ số chất lượng
    if "PTV" in structures:
        ptv = structures["PTV"]
        
        # Tính chỉ số phù hợp (CI)
        prescribed_dose = plan.prescribed_dose
        volume_receiving_prescribed_dose = np.sum(dose_distribution >= prescribed_dose)
        ptv_volume = np.sum(ptv)
        if ptv_volume > 0:
            metrics.conformity_index = (volume_receiving_prescribed_dose / ptv_volume)
        
        # Tính chỉ số đồng nhất (HI)
        if np.sum(ptv) > 0:
            d2 = np.percentile(dose_distribution[ptv > 0], 2)
            d98 = np.percentile(dose_distribution[ptv > 0], 98)
            if d2 > 0:
                metrics.homogeneity_index = (d2 - d98) / prescribed_dose
        
        # Tính độ phủ mục tiêu
        metrics.target_coverage = np.sum((dose_distribution >= prescribed_dose) & (ptv > 0)) / np.sum(ptv)
    
    # Tính chỉ số gradient (GI) nếu có các cấu trúc liên quan
    if "PTV" in structures and "BODY" in structures:
        ptv = structures["PTV"]
        body = structures["BODY"]
        
        half_prescribed_dose = plan.prescribed_dose / 2
        volume_receiving_half_prescribed_dose = np.sum((dose_distribution >= half_prescribed_dose) & (body > 0))
        volume_receiving_prescribed_dose = np.sum((dose_distribution >= plan.prescribed_dose) & (body > 0))
        
        if volume_receiving_prescribed_dose > 0:
            metrics.gradient_index = volume_receiving_half_prescribed_dose / volume_receiving_prescribed_dose
    
    # Các chỉ số DVH có thể được tính toán từ phân bố liều và cấu trúc
    # Đây là chỉ một ví dụ đơn giản
    
    return metrics
