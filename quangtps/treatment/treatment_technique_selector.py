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
from typing import Dict, List, Any, Optional, Union, Tuple, Type, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from quangtps.planning.plan import Plan, PlanType

from quangtps.treatment.techniques.dcat import DCAT
from quangtps.treatment.techniques.imrt import IMRT, IMRTOptimizationType, IMRTDeliveryType
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.stereotactic import SRS, SBRT
from quangtps.treatment.techniques.conformal import Conformal3DRT
from quangtps.treatment.techniques.electron import ElectronTherapy
from quangtps.treatment.techniques.proton import ProtonTherapy
from quangtps.treatment.techniques.carbon import CarbonIonTherapy
from quangtps.treatment.techniques.flash import FLASHRadiotherapy
from quangtps.treatment.techniques.igrt import IGRT
from quangtps.treatment.techniques.adaptive import AdaptiveRadiotherapy
from quangtps.treatment.techniques.bnct import BNCT
from quangtps.treatment.techniques.tbi import TBI

from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.plan import get_plan_class, get_plan_type_enum
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


class TechniqueSuitabilityCalculator:
    """
    Lớp tính toán độ phù hợp của các kỹ thuật xạ trị dựa trên dữ liệu lâm sàng.
    
    Lớp này phân tích các thông số lâm sàng như vị trí khối u, kích thước, 
    hình dạng, và dữ liệu bệnh nhân để xác định kỹ thuật nào phù hợp nhất.
    """
    
    def __init__(self):
        """Khởi tạo bộ tính toán độ phù hợp kỹ thuật."""
        self.technique_selector = TreatmentTechniqueSelector()
        
        # Trọng số cho các tiêu chí
        self.criteria_weights = {
            "dose_conformity": 0.3,       # Độ phù hợp của liều
            "oar_sparing": 0.25,          # Bảo vệ cơ quan nguy cấp
            "treatment_efficiency": 0.15,  # Hiệu quả điều trị
            "delivery_accuracy": 0.2,      # Độ chính xác phát tia
            "resource_availability": 0.1   # Tính khả dụng của nguồn lực
        }
        
        # Điểm cơ sở cho mỗi kỹ thuật trên từng tiêu chí (thang điểm 0-10)
        self.base_technique_scores = {
            "3D-CRT": {
                "dose_conformity": 6,
                "oar_sparing": 5,
                "treatment_efficiency": 9,
                "delivery_accuracy": 7,
                "resource_availability": 10
            },
            "IMRT": {
                "dose_conformity": 8,
                "oar_sparing": 8,
                "treatment_efficiency": 7,
                "delivery_accuracy": 8,
                "resource_availability": 8
            },
            "VMAT": {
                "dose_conformity": 9,
                "oar_sparing": 9,
                "treatment_efficiency": 8,
                "delivery_accuracy": 8,
                "resource_availability": 7
            },
            "SRS": {
                "dose_conformity": 10,
                "oar_sparing": 7,
                "treatment_efficiency": 6,
                "delivery_accuracy": 10,
                "resource_availability": 6
            },
            "SBRT": {
                "dose_conformity": 9,
                "oar_sparing": 7,
                "treatment_efficiency": 6,
                "delivery_accuracy": 9,
                "resource_availability": 6
            },
            "ELECTRON": {
                "dose_conformity": 5,
                "oar_sparing": 6,
                "treatment_efficiency": 9,
                "delivery_accuracy": 6,
                "resource_availability": 9
            },
            "PROTON": {
                "dose_conformity": 9,
                "oar_sparing": 10,
                "treatment_efficiency": 5,
                "delivery_accuracy": 9,
                "resource_availability": 4
            },
            "CARBON": {
                "dose_conformity": 10,
                "oar_sparing": 10,
                "treatment_efficiency": 4,
                "delivery_accuracy": 9,
                "resource_availability": 2
            }
        }
    
    def calculate_suitability_scores(
        self,
        site: TreatmentSite,
        complexity: TreatmentComplexity,
        target_volume: float = None,
        target_depth: float = None,
        oar_proximity: bool = False,
        patient_condition: str = "Normal",
        **kwargs
    ) -> Dict[str, float]:
        """
        Tính toán điểm phù hợp cho các kỹ thuật xạ trị.
        
        Parameters
        ----------
        site : TreatmentSite
            Vị trí điều trị
        complexity : TreatmentComplexity
            Mức độ phức tạp của điều trị
        target_volume : float, optional
            Thể tích khối u (cc)
        target_depth : float, optional
            Độ sâu của khối u (cm)
        oar_proximity : bool, optional
            Có cơ quan nguy cấp ở gần không
        patient_condition : str, optional
            Tình trạng bệnh nhân (Normal, Critical, etc.)
        **kwargs : dict
            Các tham số bổ sung
            
        Returns
        -------
        Dict[str, float]
            Từ điển với khóa là tên kỹ thuật và giá trị là điểm phù hợp
        """
        # Lấy danh sách các kỹ thuật được khuyến nghị
        recommended_techniques = self.technique_selector.get_recommended_techniques(site, complexity)
        
        # Áp dụng các điều chỉnh dựa trên thông số đầu vào
        adjusted_scores = {}
        
        for technique in recommended_techniques:
            if technique not in self.base_technique_scores:
                continue
                
            # Bắt đầu với điểm cơ sở
            technique_scores = self.base_technique_scores[technique].copy()
            
            # Điều chỉnh dựa trên thể tích khối u
            if target_volume is not None:
                if technique == "SRS" and target_volume > 30:
                    technique_scores["dose_conformity"] -= 2
                elif technique == "SBRT" and target_volume > 100:
                    technique_scores["dose_conformity"] -= 1
                elif technique in ["PROTON", "CARBON"] and target_volume < 10:
                    technique_scores["resource_availability"] -= 1
                    
            # Điều chỉnh dựa trên độ sâu
            if target_depth is not None:
                if technique == "ELECTRON":
                    if target_depth > 5:
                        technique_scores["dose_conformity"] -= 3
                    elif target_depth > 3:
                        technique_scores["dose_conformity"] -= 1
                elif technique in ["3D-CRT", "IMRT", "VMAT"] and target_depth > 15:
                    technique_scores["oar_sparing"] -= 1
                    
            # Điều chỉnh dựa trên độ gần của cơ quan nguy cấp
            if oar_proximity:
                if technique in ["3D-CRT"]:
                    technique_scores["oar_sparing"] -= 2
                elif technique in ["IMRT", "VMAT", "PROTON", "CARBON"]:
                    technique_scores["oar_sparing"] += 1
                    
            # Điều chỉnh dựa trên tình trạng bệnh nhân
            if patient_condition == "Critical":
                if technique in ["PROTON", "CARBON"]:
                    technique_scores["treatment_efficiency"] -= 2
                elif technique in ["IMRT", "VMAT"]:
                    technique_scores["treatment_efficiency"] -= 1
                    
            # Đảm bảo điểm không vượt quá giới hạn
            for criterion in technique_scores:
                technique_scores[criterion] = max(0, min(technique_scores[criterion], 10))
                
            # Tính điểm tổng hợp có trọng số
            weighted_score = sum(
                technique_scores[criterion] * self.criteria_weights[criterion]
                for criterion in self.criteria_weights
            )
            
            adjusted_scores[technique] = round(weighted_score, 2)
            
        return adjusted_scores
    
    def get_recommended_technique(
        self,
        site: TreatmentSite,
        complexity: TreatmentComplexity,
        **kwargs
    ) -> str:
        """
        Lấy kỹ thuật được khuyến nghị nhất dựa trên các tham số lâm sàng.
        
        Parameters
        ----------
        site : TreatmentSite
            Vị trí điều trị
        complexity : TreatmentComplexity
            Mức độ phức tạp của điều trị
        **kwargs : dict
            Các tham số bổ sung
            
        Returns
        -------
        str
            Tên kỹ thuật được khuyến nghị
        """
        scores = self.calculate_suitability_scores(site, complexity, **kwargs)
        
        if not scores:
            # Nếu không có kỹ thuật nào được tính toán, sử dụng kỹ thuật mặc định
            return "3D-CRT"
            
        # Trả về kỹ thuật có điểm cao nhất
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_technique_comparison(
        self,
        site: TreatmentSite,
        complexity: TreatmentComplexity,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tạo báo cáo so sánh các kỹ thuật xạ trị.
        
        Parameters
        ----------
        site : TreatmentSite
            Vị trí điều trị
        complexity : TreatmentComplexity
            Mức độ phức tạp của điều trị
        **kwargs : dict
            Các tham số bổ sung
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Từ điển các kỹ thuật với thông tin so sánh
        """
        scores = self.calculate_suitability_scores(site, complexity, **kwargs)
        recommended_techniques = self.technique_selector.get_recommended_techniques(site, complexity)
        
        comparison = {}
        
        for technique in recommended_techniques:
            if technique not in self.base_technique_scores:
                continue
                
            comparison[technique] = {
                "score": scores.get(technique, 0),
                "advantages": self._get_technique_advantages(technique),
                "disadvantages": self._get_technique_disadvantages(technique),
                "suitable_for": self._get_technique_suitable_sites(technique)
            }
            
        return comparison
    
    def _get_technique_advantages(self, technique: str) -> List[str]:
        """Lấy danh sách ưu điểm của kỹ thuật."""
        advantages = {
            "3D-CRT": [
                "Đơn giản, dễ thực hiện",
                "Thời gian điều trị ngắn",
                "Phù hợp với nhiều cơ sở y tế",
                "Chi phí thấp"
            ],
            "IMRT": [
                "Phân bố liều phù hợp tốt",
                "Bảo vệ cơ quan nguy cấp tốt",
                "Liều thuần nhất trong thể tích điều trị",
                "Tỷ lệ kiểm soát khối u cao"
            ],
            "VMAT": [
                "Thời gian chiếu xạ ngắn hơn IMRT",
                "Phân bố liều phù hợp rất tốt",
                "Bảo vệ cơ quan nguy cấp xuất sắc",
                "Hiệu quả với vùng điều trị phức tạp"
            ],
            "SRS": [
                "Độ chính xác cực cao",
                "Liều đơn hoặc vài phân liều",
                "Xuất sắc cho khối u nhỏ",
                "Giảm thiểu tác động lên mô lành"
            ],
            "SBRT": [
                "Độ chính xác cao",
                "Thời gian điều trị ngắn (vài phân liều)",
                "Kiểm soát khối u tốt",
                "Ít phiên điều trị"
            ],
            "ELECTRON": [
                "Phù hợp với tổn thương nông",
                "Bảo vệ mô sâu tốt",
                "Đơn giản để thực hiện",
                "Chi phí thấp"
            ],
            "PROTON": [
                "Kiểm soát liều rất chính xác",
                "Bảo vệ mô lành xuất sắc",
                "Hiệu quả với khối u gần cơ quan nguy cấp",
                "Giảm liều tích lũy"
            ],
            "CARBON": [
                "Hiệu quả sinh học cao (RBE cao)",
                "Xuất sắc với khối u kháng xạ",
                "Bảo vệ mô lành tối ưu",
                "Độ chính xác vật lý và sinh học cao"
            ]
        }
        
        return advantages.get(technique, ["Không có thông tin"])
    
    def _get_technique_disadvantages(self, technique: str) -> List[str]:
        """Lấy danh sách nhược điểm của kỹ thuật."""
        disadvantages = {
            "3D-CRT": [
                "Độ phù hợp liều thấp với khối u hình dạng phức tạp",
                "Khả năng bảo vệ cơ quan nguy cấp hạn chế",
                "Không tối ưu cho các vị trí phức tạp"
            ],
            "IMRT": [
                "Thời gian điều trị mỗi phiên dài",
                "Yêu cầu công nghệ và kỹ thuật cao hơn",
                "Chi phí cao hơn 3D-CRT",
                "Cần QA phức tạp hơn"
            ],
            "VMAT": [
                "Yêu cầu thiết bị hiện đại",
                "Chi phí cao",
                "Đòi hỏi QA phức tạp",
                "Thể tích liều thấp rộng hơn"
            ],
            "SRS": [
                "Giới hạn kích thước khối u (<3-4cm)",
                "Yêu cầu thiết bị đặc biệt",
                "Đòi hỏi kỹ thuật và nhân lực chuyên sâu",
                "Chi phí cao"
            ],
            "SBRT": [
                "Giới hạn kích thước khối u",
                "Yêu cầu cố định chặt và chính xác",
                "Rủi ro độc tính cao nếu bố trí không chính xác",
                "Chi phí cao"
            ],
            "ELECTRON": [
                "Giới hạn độ sâu điều trị (<5cm)",
                "Không phù hợp với tổn thương sâu",
                "Phân bố liều không đồng đều tại biên",
                "Bị ảnh hưởng bởi độ không đồng nhất của mô"
            ],
            "PROTON": [
                "Chi phí rất cao",
                "Tính khả dụng hạn chế",
                "Yêu cầu công nghệ và kỹ thuật đặc biệt",
                "Nhạy cảm với thay đổi giải phẫu"
            ],
            "CARBON": [
                "Chi phí cực cao",
                "Rất ít trung tâm có thể thực hiện",
                "Đòi hỏi chuyên môn và thiết bị đặc biệt",
                "Mô hình sinh học phức tạp"
            ]
        }
        
        return disadvantages.get(technique, ["Không có thông tin"])
    
    def _get_technique_suitable_sites(self, technique: str) -> List[str]:
        """Lấy danh sách vị trí phù hợp với kỹ thuật."""
        suitable_sites = {
            "3D-CRT": [
                "Ung thư vú",
                "Ung thư phổi giai đoạn sớm",
                "Ung thư tiền liệt tuyến",
                "Ung thư đầu cổ đơn giản"
            ],
            "IMRT": [
                "Ung thư đầu cổ phức tạp",
                "Ung thư tiền liệt tuyến",
                "Ung thư phổi gần cơ quan nguy cấp",
                "Ung thư não",
                "Ung thư vùng chậu"
            ],
            "VMAT": [
                "Ung thư đầu cổ",
                "Ung thư não",
                "Ung thư tiền liệt tuyến",
                "Ung thư vùng chậu phức tạp",
                "Ung thư phổi phức tạp"
            ],
            "SRS": [
                "U não nguyên phát nhỏ",
                "Di căn não",
                "U màng não",
                "U tuyến yên",
                "U dị dạng mạch máu não"
            ],
            "SBRT": [
                "Ung thư phổi không phẫu thuật được",
                "Ung thư gan nguyên phát",
                "Di căn gan, phổi, xương số lượng ít",
                "Ung thư tụy",
                "Ung thư thận"
            ],
            "ELECTRON": [
                "Ung thư da",
                "Ung thư môi",
                "Tổn thương bề mặt tới trung bình",
                "Boost sẹo mổ",
                "Ung thư vú bổ trợ"
            ],
            "PROTON": [
                "U não ở trẻ em",
                "U đáy sọ",
                "U dây thần kinh thị giác",
                "U tủy sống",
                "Ung thư phổi gần tim",
                "Ung thư gan",
                "Ung thư đầu cổ gần thần kinh thị giác"
            ],
            "CARBON": [
                "U kháng xạ",
                "Ung thư tuyến giáp không biệt hóa",
                "Sarcom mô mềm",
                "Ung thư tuyến nước bọt tái phát",
                "U đáy sọ không thể phẫu thuật",
                "Ung thư tụy"
            ]
        }
        
        return suitable_sites.get(technique, ["Không có thông tin"])


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
            "3D-CRT": Conformal3DRT,
            "IMRT": IMRT,
            "VMAT": VMAT,
            "DCAT": DCAT,
            "SRS": SRS,
            "SBRT": SBRT,
            "ELECTRON": ElectronTherapy,
            "PROTON": ProtonTherapy,
            "CARBON": CarbonIonTherapy,
            "FLASH": FLASHRadiotherapy,
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
    ) -> Tuple[Any, Any]:
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
        Tuple
            Kế hoạch điều trị và thể hiện kỹ thuật xạ trị
        """
        # Tạo ID kế hoạch
        plan_id = str(uuid.uuid4())
        
        # Get Plan and PlanType classes dynamically
        Plan = get_plan_class()
        PlanType = get_plan_type_enum()
        
        # Tạo kế hoạch điều trị
        plan = Plan(
            plan_name=plan_name,
            plan_id=plan_id,
            patient_id=patient_id,
            description=f"{technique_name} plan for {site}"
        )
        
        # Thiết lập thông tin cơ bản
        if isinstance(site, str):
            site = TreatmentSite(site)
        
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
        
        # Thiết lập phân đoạn và máy điều trị
        technique.set_fractionation(fractionation)
        
        if machine:
            technique.set_machine(machine)
        
        # Liên kết kế hoạch và kỹ thuật
        plan.treatment_technique = technique
        
        return plan, technique
