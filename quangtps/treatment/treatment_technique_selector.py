#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module lựa chọn kỹ thuật xạ trị (Treatment Technique Selector).

Module này cung cấp các lớp và phương thức để chọn và tích hợp các kỹ thuật xạ trị phù hợp
cho mỗi bệnh nhân dựa trên các chỉ định lâm sàng và đặc điểm của bệnh.
"""

import logging
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple

from quangtps.treatment.techniques.dcat import DCAT
from quangtps.treatment.techniques.imrt import IMRT, IMRTOptimizationType, IMRTDeliveryType
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.stereotactic import SRS, SBRT
from quangtps.treatment.techniques.conformal import Conformal3D
from quangtps.treatment.techniques.electron import ElectronTherapy
from quangtps.treatment.techniques.proton import ProtonTherapy
from quangtps.treatment.techniques.carbon import CarbonTherapy
from quangtps.treatment.techniques.flash import FLASHTherapy
from quangtps.treatment.techniques.igrt import IGRT
from quangtps.treatment.techniques.adaptive import AdaptiveRadiotherapy
from quangtps.treatment.techniques.bnct import BNCT
from quangtps.treatment.techniques.tbi import TBI

from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.plan import TreatmentPlan, PlanType
from quangtps.treatment.machine.treatment_machine import TreatmentMachine

logger = logging.getLogger(__name__)


class TechniqueCategory(str, Enum):
    """Enum đại diện cho các loại kỹ thuật xạ trị."""
    PHOTON_EXTERNAL = "Photon External Beam"
    ELECTRON = "Electron"
    PARTICLE = "Particle Therapy"
    SPECIALIZED = "Specialized Technique"
    ADAPTIVE = "Adaptive Radiotherapy"


class TreatmentSite(str, Enum):
    """Enum đại diện cho các vị trí điều trị."""
    BRAIN = "Brain"
    HEAD_NECK = "Head and Neck"
    LUNG = "Lung"
    BREAST = "Breast"
    ABDOMEN = "Abdomen"
    PELVIS = "Pelvis"
    PROSTATE = "Prostate"
    SPINE = "Spine"
    EXTREMITY = "Extremity"
    WHOLE_BODY = "Whole Body"


class TreatmentComplexity(str, Enum):
    """Enum đại diện cho mức độ phức tạp của điều trị."""
    SIMPLE = "Simple"
    INTERMEDIATE = "Intermediate"
    COMPLEX = "Complex"
    VERY_COMPLEX = "Very Complex"


class TreatmentTechniqueSelector:
    """
    Lớp lựa chọn kỹ thuật xạ trị.
    
    Lớp này cung cấp các phương thức để chọn và tích hợp kỹ thuật xạ trị phù hợp
    cho mỗi bệnh nhân dựa trên các chỉ định lâm sàng.
    """
    
    def __init__(self):
        """Khởi tạo bộ lựa chọn kỹ thuật xạ trị."""
        # Ánh xạ từ loại kỹ thuật tới lớp kỹ thuật
        self.technique_map = {
            "3D-CRT": Conformal3D,
            "IMRT": IMRT,
            "VMAT": VMAT,
            "DCAT": DCAT,
            "SRS": SRS,
            "SBRT": SBRT,
            "ELECTRON": ElectronTherapy,
            "PROTON": ProtonTherapy,
            "CARBON": CarbonTherapy,
            "FLASH": FLASHTherapy,
            "IGRT": IGRT,
            "ADAPTIVE": AdaptiveRadiotherapy,
            "BNCT": BNCT,
            "TBI": TBI
        }
        
        # Hướng dẫn kỹ thuật cho mỗi vị trí điều trị và mức độ phức tạp
        self.site_complexity_technique_map = {
            TreatmentSite.BRAIN: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["IMRT", "VMAT"],
                TreatmentComplexity.COMPLEX: ["SRS", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["SRS", "PROTON"]
            },
            TreatmentSite.HEAD_NECK: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["3D-CRT", "IMRT"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["IMRT", "VMAT", "ADAPTIVE"]
            },
            TreatmentSite.LUNG: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["3D-CRT", "IMRT"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["SBRT", "VMAT"]
            },
            TreatmentSite.BREAST: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["3D-CRT", "IMRT"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["IMRT", "VMAT", "ADAPTIVE"]
            },
            TreatmentSite.ABDOMEN: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["3D-CRT", "IMRT"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["SBRT", "VMAT"]
            },
            TreatmentSite.PELVIS: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["3D-CRT", "IMRT"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["IMRT", "VMAT", "ADAPTIVE"]
            },
            TreatmentSite.PROSTATE: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["IMRT"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["SBRT", "VMAT", "PROTON"]
            },
            TreatmentSite.SPINE: {
                TreatmentComplexity.SIMPLE: ["3D-CRT"],
                TreatmentComplexity.INTERMEDIATE: ["IMRT"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["SBRT", "SRS"]
            },
            TreatmentSite.EXTREMITY: {
                TreatmentComplexity.SIMPLE: ["3D-CRT", "ELECTRON"],
                TreatmentComplexity.INTERMEDIATE: ["3D-CRT", "IMRT", "ELECTRON"],
                TreatmentComplexity.COMPLEX: ["IMRT", "VMAT"],
                TreatmentComplexity.VERY_COMPLEX: ["IMRT", "VMAT", "ELECTRON"]
            },
            TreatmentSite.WHOLE_BODY: {
                TreatmentComplexity.SIMPLE: ["TBI"],
                TreatmentComplexity.INTERMEDIATE: ["TBI"],
                TreatmentComplexity.COMPLEX: ["TBI"],
                TreatmentComplexity.VERY_COMPLEX: ["TBI"]
            }
        }
    
    def get_recommended_techniques(
        self,
        site: Union[TreatmentSite, str],
        complexity: Union[TreatmentComplexity, str]
    ) -> List[str]:
        """
        Lấy danh sách các kỹ thuật xạ trị được khuyến nghị cho một vị trí và mức độ phức tạp.
        
        Parameters
        ----------
        site : Union[TreatmentSite, str]
            Vị trí điều trị
        complexity : Union[TreatmentComplexity, str]
            Mức độ phức tạp
            
        Returns
        -------
        List[str]
            Danh sách các kỹ thuật được khuyến nghị
        """
        # Chuyển đổi kiểu nếu cần
        if isinstance(site, str):
            site = TreatmentSite(site)
        
        if isinstance(complexity, str):
            complexity = TreatmentComplexity(complexity)
        
        # Lấy danh sách kỹ thuật được khuyến nghị
        techniques = self.site_complexity_technique_map.get(site, {}).get(complexity, [])
        
        if not techniques:
            logger.warning(f"No recommended techniques found for site={site}, complexity={complexity}")
            return ["3D-CRT"]  # Mặc định 3D-CRT nếu không tìm thấy khuyến nghị
        
        return techniques
    
    def create_technique_instance(
        self,
        technique_name: str,
        plan_name: str,
        plan_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Tạo một thể hiện của kỹ thuật xạ trị.
        
        Parameters
        ----------
        technique_name : str
            Tên kỹ thuật xạ trị
        plan_name : str
            Tên kế hoạch
        plan_id : str, optional
            ID kế hoạch
        **kwargs : dict
            Các tham số bổ sung
            
        Returns
        -------
        Any
            Thể hiện của kỹ thuật xạ trị
            
        Raises
        ------
        ValueError
            Nếu kỹ thuật không được hỗ trợ
        """
        # Kiểm tra xem kỹ thuật có được hỗ trợ không
        if technique_name not in self.technique_map:
            raise ValueError(f"Unsupported technique: {technique_name}")
        
        # Tạo thể hiện của kỹ thuật
        technique_class = self.technique_map[technique_name]
        
        # Xử lý một số trường hợp đặc biệt
        if technique_name == "IMRT":
            optimization_type = kwargs.get('optimization_type', IMRTOptimizationType.FLUENCE_MAP)
            delivery_type = kwargs.get('delivery_type', IMRTDeliveryType.STEP_AND_SHOOT)
            return technique_class(name=plan_name, optimization_type=optimization_type, delivery_type=delivery_type)
        
        # Trường hợp tổng quát
        return technique_class(plan_name=plan_name, plan_id=plan_id)
    
    def create_plan_with_technique(
        self,
        technique_name: str,
        plan_name: str,
        patient_id: str,
        site: Union[TreatmentSite, str],
        fractionation: Fractionation,
        machine: Optional[TreatmentMachine] = None,
        **technique_params
    ) -> Tuple[TreatmentPlan, Any]:
        """
        Tạo một kế hoạch điều trị với kỹ thuật xạ trị.
        
        Parameters
        ----------
        technique_name : str
            Tên kỹ thuật xạ trị
        plan_name : str
            Tên kế hoạch
        patient_id : str
            ID bệnh nhân
        site : Union[TreatmentSite, str]
            Vị trí điều trị
        fractionation : Fractionation
            Thông tin phân đoạn
        machine : TreatmentMachine, optional
            Máy xạ trị
        **technique_params : dict
            Các tham số kỹ thuật bổ sung
            
        Returns
        -------
        Tuple[TreatmentPlan, Any]
            Kế hoạch điều trị và thể hiện kỹ thuật xạ trị
        """
        # Tạo ID kế hoạch
        plan_id = str(uuid.uuid4())
        
        # Tạo kế hoạch điều trị
        plan = TreatmentPlan(
            plan_name=plan_name,
            plan_id=plan_id,
            patient_id=patient_id,
            description=f"{technique_name} plan for {site}"
        )
        
        # Thiết lập thông tin cơ bản
        if isinstance(site, str):
            site = TreatmentSite(site)
        
        plan.site = site.value
        plan.set_fractionation(fractionation)
        
        # Thiết lập loại kế hoạch dựa trên kỹ thuật
        if technique_name in ["IMRT", "VMAT"]:
            plan.set_plan_type(PlanType.EXTERNAL_BEAM)
        elif technique_name in ["SRS", "SBRT"]:
            plan.set_plan_type(PlanType.STEREOTACTIC)
        elif technique_name == "ELECTRON":
            plan.set_plan_type(PlanType.ELECTRON)
        elif technique_name in ["PROTON", "CARBON"]:
            plan.set_plan_type(PlanType.PARTICLE)
        else:
            plan.set_plan_type(PlanType.EXTERNAL_BEAM)
        
        # Tạo thể hiện kỹ thuật
        technique = self.create_technique_instance(
            technique_name=technique_name,
            plan_name=plan_name,
            plan_id=plan_id,
            **technique_params
        )
        
        # Thiết lập thông tin phân đoạn và máy xạ trị
        technique.set_fractionation(fractionation)
        
        if machine:
            technique.set_treatment_machine(machine)
        
        # Lưu thông tin kỹ thuật vào metadata của kế hoạch
        plan.metadata["technique"] = technique_name
        plan.metadata["technique_params"] = technique_params
        
        return plan, technique
