#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý lập kế hoạch điều trị (Treatment Planner).

Module này cung cấp các lớp và phương thức để tạo và quản lý kế hoạch điều trị,
tích hợp giữa contour và các kỹ thuật xạ trị khác nhau.
"""

import os
import uuid
import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from quangtps.planning.plan import Plan, PlanType, PlanStatus
from quangtps.planning.beam import BeamArrangement
from quangtps.planning.prescription import Prescription
from quangtps.planning.optimization import OptimizationSettings
from quangtps.planning.evaluation import PlanEvaluation
from quangtps.planning.comparison import PlanComparison

# Import for different treatment techniques
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.imrt import IMRT, StaticIMRT, DynamicIMRT
from quangtps.treatment.techniques.conformal import Conformal3DRT
from quangtps.treatment.techniques.stereotactic import SRS, SBRT
from quangtps.treatment.techniques.proton import ProtonTherapy
from quangtps.treatment.techniques.carbon import CarbonIonTherapy
from quangtps.treatment.techniques.electron import ElectronTherapy
from quangtps.treatment.techniques.flash import FLASHTherapy
from quangtps.treatment.techniques.tbi import TBI
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.mlc.mlc_model import MLCModel

# Import contour module
from quangtps.segmentation.contour.contour_manager import ContourSet, ContourManager

logger = logging.getLogger(__name__)


class TreatmentTechnique(str, Enum):
    """Enum cho các kỹ thuật điều trị xạ trị."""

    CONFORMAL_3D = "3DCRT"  # Xạ trị hình dạng 3D
    IMRT = "IMRT"  # Xạ trị điều biến cường độ
    VMAT = "VMAT"  # Xạ trị cung điều biến thể tích
    SRS = "SRS"  # Xạ phẫu thần kinh
    SBRT = "SBRT"  # Xạ trị thân định vị
    PROTON = "Proton"  # Xạ trị Proton
    CARBON = "Carbon"  # Xạ trị Ion Carbon
    ELECTRON = "Electron"  # Xạ trị Electron
    TBI = "TBI"  # Xạ trị toàn thân
    FLASH = "FLASH"  # Xạ trị FLASH (liều cao, thời gian ngắn)
    BNCT = "BNCT"  # Xạ trị bắt neutron boron
    ADAPTIVE = "Adaptive"  # Xạ trị thích ứng
    IGRT = "IGRT"  # Xạ trị điều khiển bằng hình ảnh
    CUSTOM = "Custom"  # Kỹ thuật tùy chỉnh


class TreatmentPlanner:
    """
    Lớp quản lý việc lập kế hoạch điều trị xạ trị.

    Lớp này giúp tạo và quản lý kế hoạch điều trị dựa trên các contour đã định nghĩa
    và các kỹ thuật xạ trị khác nhau.
    """

    def __init__(
        self, patient_id: str, contour_manager: Optional[ContourManager] = None
    ):
        """
        Khởi tạo trình quản lý kế hoạch điều trị.

        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        contour_manager : ContourManager, optional
            Đối tượng quản lý contour của bệnh nhân
        """
        self.patient_id = patient_id
        self.contour_manager = contour_manager or ContourManager()
        self.plans: Dict[str, Plan] = {}
        self.treatment_machines: Dict[str, Linac] = {}
        self.mlc_models: Dict[str, MLCModel] = {}

        # Thêm các máy điều trị xạ trị mặc định
        self._initialize_default_machines()

        logger.info(f"Khởi tạo TreatmentPlanner cho bệnh nhân {patient_id}")

    def _initialize_default_machines(self):
        """Thiết lập các máy xạ trị mặc định"""
        # TrueBeam STx
        truebeam = Linac(
            name="TrueBeam STx",
            max_field_size=(40, 40),
            available_energies=["6MV", "10MV", "15MV", "6FFF", "10FFF"],
            max_dose_rate=1400,
            has_cone_beam_ct=True,
        )
        self.add_treatment_machine(truebeam)

        # Varian Halcyon
        halcyon = Linac(
            name="Halcyon",
            max_field_size=(28, 28),
            available_energies=["6MV"],
            max_dose_rate=800,
            has_cone_beam_ct=True,
        )
        self.add_treatment_machine(halcyon)

        # Elekta Versa HD
        versa = Linac(
            name="Versa HD",
            max_field_size=(40, 40),
            available_energies=["6MV", "10MV", "15MV", "6FFF", "10FFF"],
            max_dose_rate=1400,
            has_cone_beam_ct=True,
        )
        self.add_treatment_machine(versa)

    def add_treatment_machine(self, machine: Linac):
        """
        Thêm máy điều trị xạ trị.

        Parameters
        ----------
        machine : Linac
            Máy điều trị xạ trị cần thêm
        """
        self.treatment_machines[machine.name] = machine
        logger.info(f"Đã thêm máy điều trị {machine.name}")

    def create_plan(
        self,
        plan_name: str,
        contour_set_id: str,
        technique: TreatmentTechnique,
        machine_name: str,
        plan_type: PlanType = PlanType.DEFINITIVE,
    ) -> str:
        """
        Tạo kế hoạch điều trị mới.

        Parameters
        ----------
        plan_name : str
            Tên kế hoạch điều trị
        contour_set_id : str
            ID của bộ contour sử dụng cho kế hoạch
        technique : TreatmentTechnique
            Kỹ thuật xạ trị
        machine_name : str
            Tên máy điều trị
        plan_type : PlanType, optional
            Loại kế hoạch điều trị, mặc định là DEFINITIVE

        Returns
        -------
        str
            ID của kế hoạch điều trị được tạo

        Raises
        ------
        ValueError
            Nếu contour set không tồn tại hoặc máy điều trị không tồn tại
        """
        # Kiểm tra bộ contour tồn tại
        if not self.contour_manager.has_contour_set(contour_set_id):
            raise ValueError(f"Bộ contour với ID {contour_set_id} không tồn tại")

        # Kiểm tra máy điều trị tồn tại
        if machine_name not in self.treatment_machines:
            raise ValueError(f"Máy điều trị {machine_name} không tồn tại")

        # Tạo kế hoạch điều trị mới
        plan = Plan(plan_name, self.patient_id, plan_type=plan_type)
        plan.technique = technique.value
        plan.machine_id = machine_name

        # Lưu tham chiếu đến bộ contour
        plan.contour_set_id = contour_set_id

        # Lưu kế hoạch và trả về ID
        self.plans[plan.plan_id] = plan
        logger.info(
            f"Đã tạo kế hoạch điều trị {plan_name} (ID: {plan.plan_id}) với kỹ thuật {technique.value}"
        )

        return plan.plan_id

    def create_vmat_plan(
        self,
        plan_name: str,
        contour_set_id: str,
        machine_name: str = "TrueBeam STx",
        energy: str = "6MV",
        plan_type: PlanType = PlanType.DEFINITIVE,
    ) -> Tuple[str, VMAT]:
        """
        Tạo kế hoạch điều trị VMAT.

        Parameters
        ----------
        plan_name : str
            Tên kế hoạch điều trị
        contour_set_id : str
            ID của bộ contour sử dụng cho kế hoạch
        machine_name : str, optional
            Tên máy điều trị, mặc định là TrueBeam STx
        energy : str, optional
            Năng lượng sử dụng, mặc định là 6MV
        plan_type : PlanType, optional
            Loại kế hoạch điều trị, mặc định là DEFINITIVE

        Returns
        -------
        Tuple[str, VMAT]
            ID của kế hoạch điều trị và đối tượng VMAT tương ứng
        """
        # Tạo kế hoạch cơ bản
        plan_id = self.create_plan(
            plan_name, contour_set_id, TreatmentTechnique.VMAT, machine_name, plan_type
        )

        # Tạo kế hoạch VMAT cụ thể
        vmat_plan = VMAT(plan_name, plan_id)
        vmat_plan.set_treatment_machine(self.treatment_machines[machine_name])

        # Liên kết kế hoạch VMAT với kế hoạch cơ bản
        self.plans[plan_id].treatment_plan = vmat_plan

        return plan_id, vmat_plan

    def create_imrt_plan(
        self,
        plan_name: str,
        contour_set_id: str,
        machine_name: str = "TrueBeam STx",
        energy: str = "6MV",
        plan_type: PlanType = PlanType.DEFINITIVE,
    ) -> Tuple[str, IMRT]:
        """
        Tạo kế hoạch điều trị IMRT.

        Parameters
        ----------
        plan_name : str
            Tên kế hoạch điều trị
        contour_set_id : str
            ID của bộ contour sử dụng cho kế hoạch
        machine_name : str, optional
            Tên máy điều trị, mặc định là TrueBeam STx
        energy : str, optional
            Năng lượng sử dụng, mặc định là 6MV
        plan_type : PlanType, optional
            Loại kế hoạch điều trị, mặc định là DEFINITIVE

        Returns
        -------
        Tuple[str, IMRT]
            ID của kế hoạch điều trị và đối tượng IMRT tương ứng
        """
        # Tạo kế hoạch cơ bản
        plan_id = self.create_plan(
            plan_name, contour_set_id, TreatmentTechnique.IMRT, machine_name, plan_type
        )

        # Tạo kế hoạch IMRT cụ thể
        imrt_plan = IMRT(plan_name, plan_id)
        imrt_plan.set_treatment_machine(self.treatment_machines[machine_name])

        # Liên kết kế hoạch IMRT với kế hoạch cơ bản
        self.plans[plan_id].treatment_plan = imrt_plan

        return plan_id, imrt_plan

    def set_prescription(
        self,
        plan_id: str,
        target_structure: str,
        total_dose: float,
        num_fractions: int,
        dose_per_fraction: Optional[float] = None,
    ) -> None:
        """
        Thiết lập đơn điều trị cho kế hoạch.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
        target_structure : str
            Tên cấu trúc đích (PTV)
        total_dose : float
            Tổng liều điều trị (Gy)
        num_fractions : int
            Số đợt điều trị
        dose_per_fraction : float, optional
            Liều mỗi đợt (Gy). Nếu không cung cấp, sẽ tính từ tổng liều và số đợt
        """
        if plan_id not in self.plans:
            raise ValueError(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")

        plan = self.plans[plan_id]

        # Tính liều mỗi đợt nếu không được cung cấp
        if dose_per_fraction is None:
            dose_per_fraction = total_dose / num_fractions

        # Tạo đối tượng fractionation
        fractionation = Fractionation(
            num_fractions=num_fractions,
            dose_per_fraction=dose_per_fraction,
            total_dose=total_dose,
        )

        # Tạo đơn điều trị
        prescription = Prescription(target_structure, total_dose)
        prescription.set_fractionation(fractionation)

        # Liên kết đơn điều trị với kế hoạch
        plan.set_prescription(prescription)

        # Cập nhật cả đối tượng kỹ thuật cụ thể nếu có
        if hasattr(plan, "treatment_plan") and plan.treatment_plan is not None:
            plan.treatment_plan.set_fractionation(fractionation)

        logger.info(
            f"Đã thiết lập đơn điều trị cho kế hoạch {plan.plan_name}: "
            f"{total_dose:.1f} Gy trong {num_fractions} đợt "
            f"({dose_per_fraction:.2f} Gy/đợt)"
        )

    def add_optimization_objective(
        self,
        plan_id: str,
        structure_name: str,
        objective_type: str,
        dose: float,
        volume: Optional[float] = None,
        weight: float = 1.0,
    ) -> None:
        """
        Thêm mục tiêu tối ưu hóa cho cấu trúc.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
        structure_name : str
            Tên cấu trúc
        objective_type : str
            Loại mục tiêu ("min_dose", "max_dose", "min_dvh", "max_dvh", "uniform_dose", "mean_dose")
        dose : float
            Liều tham chiếu (Gy hoặc %)
        volume : float, optional
            Thể tích tham chiếu (%) cho mục tiêu DVH
        weight : float, optional
            Trọng số của mục tiêu, mặc định là
        """
        if plan_id not in self.plans:
            raise ValueError(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")

        plan = self.plans[plan_id]

        # Tạo mục tiêu tối ưu hóa
        objective = {
            "structure": structure_name,
            "type": objective_type,
            "dose": dose,
            "volume": volume,
            "weight": weight,
        }

        # Thêm vào thiết lập tối ưu hóa
        if plan.optimization_settings is None:
            plan.optimization_settings = OptimizationSettings()

        plan.optimization_settings.add_objective(objective)

        # Cập nhật cả đối tượng kỹ thuật cụ thể nếu có
        if hasattr(plan, "treatment_plan") and plan.treatment_plan is not None:
            plan.treatment_plan.add_optimization_objective(
                structure_name, objective_type, dose, volume, weight
            )

        logger.info(
            f"Đã thêm mục tiêu tối ưu hóa cho cấu trúc {structure_name} trong kế hoạch {plan.plan_name}"
        )

    def add_constraint(
        self,
        plan_id: str,
        structure_name: str,
        constraint_type: str,
        dose: float,
        volume: Optional[float] = None,
    ) -> None:
        """
        Thêm ràng buộc cho cấu trúc.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
        structure_name : str
            Tên cấu trúc
        constraint_type : str
            Loại ràng buộc ("max_dose", "max_dvh", "mean_dose")
        dose : float
            Liều tham chiếu (Gy hoặc %)
        volume : float, optional
            Thể tích tham chiếu (%) cho ràng buộc DVH
        """
        if plan_id not in self.plans:
            raise ValueError(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")

        plan = self.plans[plan_id]

        # Tạo ràng buộc
        constraint = {
            "structure": structure_name,
            "type": constraint_type,
            "dose": dose,
            "volume": volume,
        }

        # Thêm vào thiết lập tối ưu hóa
        if plan.optimization_settings is None:
            plan.optimization_settings = OptimizationSettings()

        plan.optimization_settings.add_constraint(constraint)

        # Cập nhật cả đối tượng kỹ thuật cụ thể nếu có
        if hasattr(plan, "treatment_plan") and plan.treatment_plan is not None:
            plan.treatment_plan.add_optimization_constraint(
                structure_name, constraint_type, dose, volume
            )

        logger.info(
            f"Đã thêm ràng buộc cho cấu trúc {structure_name} trong kế hoạch {plan.plan_name}"
        )

    def optimize_plan(self, plan_id: str, max_iterations: int = 100) -> bool:
        """
        Tối ưu hóa kế hoạch điều trị.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
        max_iterations : int, optional
            Số lần lặp tối đa cho quá trình tối ưu hóa

        Returns
        -------
        bool
            True nếu tối ưu hóa thành công, False nếu không
        """
        if plan_id not in self.plans:
            raise ValueError(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")

        plan = self.plans[plan_id]

        # Kiểm tra các điều kiện cần thiết
        if plan.prescription is None:
            logger.error(
                f"Không thể tối ưu hóa kế hoạch {plan.plan_name} - đơn điều trị chưa được thiết lập"
            )
            return False

        if (
            plan.optimization_settings is None
            or not plan.optimization_settings.has_objectives()
        ):
            logger.error(
                f"Không thể tối ưu hóa kế hoạch {plan.plan_name} - chưa có mục tiêu tối ưu hóa"
            )
            return False

        # Cập nhật trạng thái
        plan.set_status(PlanStatus.OPTIMIZATION)

        # Tối ưu hóa kế hoạch dựa trên kỹ thuật cụ thể
        if hasattr(plan, "treatment_plan") and plan.treatment_plan is not None:
            try:
                if hasattr(plan.treatment_plan, "optimize"):
                    success = plan.treatment_plan.optimize(max_iterations)
                    if success:
                        plan.set_status(PlanStatus.CALCULATION)
                        return True
                    else:
                        logger.error(f"Tối ưu hóa kế hoạch {plan.plan_name} thất bại")
                        return False
                else:
                    logger.error(f"Đối tượng kỹ thuật không hỗ trợ tối ưu hóa")
                    return False
            except Exception as e:
                logger.error(f"Lỗi khi tối ưu hóa kế hoạch {plan.plan_name}: {str(e)}")
                return False
        else:
            logger.error(
                f"Kế hoạch {plan.plan_name} không có đối tượng kỹ thuật cụ thể"
            )
            return False

    def calculate_dose(self, plan_id: str) -> bool:
        """
        Tính toán phân bố liều cho kế hoạch.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị

        Returns
        -------
        bool
            True nếu tính toán thành công, False nếu không
        """
        if plan_id not in self.plans:
            raise ValueError(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")

        plan = self.plans[plan_id]
        return plan.calculate_dose()

    def evaluate_plan(self, plan_id: str) -> Optional[PlanEvaluation]:
        """
        Đánh giá kế hoạch điều trị.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị

        Returns
        -------
        Optional[PlanEvaluation]
            Đối tượng đánh giá kế hoạch nếu thành công, None nếu không
        """
        if plan_id not in self.plans:
            raise ValueError(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")

        plan = self.plans[plan_id]
        return plan.evaluate()

    def compare_plans(self, plan_ids: List[str]) -> Optional[PlanComparison]:
        """
        So sánh nhiều kế hoạch điều trị.

        Parameters
        ----------
        plan_ids : List[str]
            Danh sách ID của các kế hoạch điều trị cần so sánh

        Returns
        -------
        Optional[PlanComparison]
            Đối tượng so sánh kế hoạch nếu thành công, None nếu không
        """
        # Kiểm tra các kế hoạch tồn tại
        plans = []
        for plan_id in plan_ids:
            if plan_id not in self.plans:
                logger.error(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")
                return None
            plans.append(self.plans[plan_id])

        # Kiểm tra các kế hoạch đã được tính toán liều
        for plan in plans:
            if not plan.calculation_complete:
                logger.error(f"Kế hoạch {plan.plan_name} chưa được tính toán liều")
                return None

        # Tạo đối tượng so sánh
        comparison = PlanComparison(plans)
        comparison.calculate_comparison_metrics()

        return comparison

    def export_plan(self, plan_id: str, output_dir: str) -> bool:
        """
        Xuất kế hoạch điều trị ra file.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
        output_dir : str
            Thư mục đầu ra

        Returns
        -------
        bool
            True nếu xuất thành công, False nếu không
        """
        if plan_id not in self.plans:
            raise ValueError(f"Kế hoạch điều trị với ID {plan_id} không tồn tại")

        plan = self.plans[plan_id]

        try:
            # Tạo thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Chuyển đổi kế hoạch thành dictionary
            plan_dict = plan.to_dict()

            # Lưu file JSON
            import json

            output_file = os.path.join(
                output_dir, f"{plan.plan_name}_{plan.plan_id}.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(plan_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Đã xuất kế hoạch {plan.plan_name} ra file {output_file}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xuất kế hoạch {plan.plan_name}: {str(e)}")
            return False

    def import_plan(self, input_file: str) -> Optional[str]:
        """
        Nhập kế hoạch điều trị từ file.

        Parameters
        ----------
        input_file : str
            Đường dẫn đến file kế hoạch

        Returns
        -------
        Optional[str]
            ID của kế hoạch điều trị được nhập nếu thành công, None nếu không
        """
        try:
            # Đọc file JSON
            import json

            with open(input_file, "r", encoding="utf-8") as f:
                plan_dict = json.load(f)

            # Tạo đối tượng Plan từ dictionary
            plan = Plan.from_dict(plan_dict)

            # Thêm vào danh sách kế hoạch
            self.plans[plan.plan_id] = plan

            logger.info(f"Đã nhập kế hoạch {plan.plan_name} từ file {input_file}")
            return plan.plan_id
        except Exception as e:
            logger.error(f"Lỗi khi nhập kế hoạch từ file {input_file}: {str(e)}")
            return None
