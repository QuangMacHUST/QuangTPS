#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Protocol Manager cho QuangTPS.

Module này quản lý các protocol lâm sàng, bao gồm lưu trữ,
tải, và áp dụng các protocol cho đánh giá kế hoạch xạ trị.
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Import từ plan_quality nếu có
try:
    from quangtps.evaluation.plan_quality import (
        ClinicalGoal,
        GoalType,
        GoalPriority,
        ComparisonOperator,
        create_standard_goals,
    )

    HAS_PLAN_QUALITY = True
except ImportError:
    HAS_PLAN_QUALITY = False
    logger.warning("Plan quality module không khả dụng")


@dataclass
class ClinicalProtocol:
    """
    Định nghĩa một protocol lâm sàng.

    Protocol chứa tập hợp các mục tiêu lâm sàng cho một site điều trị cụ thể.
    """

    name: str  # Tên protocol
    site: str  # Vị trí điều trị (prostate, head_neck, etc.)
    description: str = ""  # Mô tả
    version: str = "1.0"  # Phiên bản

    # Goals và constraints
    goals: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_by: str = "QuangTPS"  # Người tạo
    created_date: datetime = field(default_factory=datetime.now)
    modified_date: datetime = field(default_factory=datetime.now)

    # Institution info
    institution: str = ""  # Cơ sở y tế
    department: str = ""  # Khoa

    # Clinical info
    fractionation: Optional[str] = (
        None  # Phân liều (conventional, hypofractionated, etc.)
    )
    prescription_dose: Optional[float] = None  # Liều kê đơn (Gy)

    def __post_init__(self):
        """Xử lý sau khởi tạo."""
        if not self.description and self.site:
            self.description = f"Clinical protocol for {self.site} treatment"

    def add_goal(self, goal_data: Dict[str, Any]):
        """Thêm mục tiêu vào protocol."""
        self.goals.append(goal_data)
        self.modified_date = datetime.now()

    def remove_goal(self, index: int) -> bool:
        """Xóa mục tiêu theo index."""
        try:
            self.goals.pop(index)
            self.modified_date = datetime.now()
            return True
        except IndexError:
            return False

    def get_goals_for_structure(self, structure_name: str) -> List[Dict[str, Any]]:
        """Lấy các mục tiêu cho cấu trúc cụ thể."""
        return [
            goal
            for goal in self.goals
            if goal.get("structure_name", "").lower() == structure_name.lower()
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        data = asdict(self)
        data["created_date"] = self.created_date.isoformat()
        data["modified_date"] = self.modified_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalProtocol":
        """Tạo từ dictionary."""
        # Parse dates
        created_date = datetime.now()
        modified_date = datetime.now()

        if "created_date" in data:
            try:
                created_date = datetime.fromisoformat(data["created_date"])
            except Exception:
                pass

        if "modified_date" in data:
            try:
                modified_date = datetime.fromisoformat(data["modified_date"])
            except Exception:
                pass

        return cls(
            name=data["name"],
            site=data["site"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            goals=data.get("goals", []),
            created_by=data.get("created_by", "QuangTPS"),
            created_date=created_date,
            modified_date=modified_date,
            institution=data.get("institution", ""),
            department=data.get("department", ""),
            fractionation=data.get("fractionation"),
            prescription_dose=data.get("prescription_dose"),
        )

    @classmethod
    def from_json(cls, json_string: str) -> "ClinicalProtocol":
        """Tạo từ JSON string."""
        try:
            data = json.loads(json_string)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            raise

    def to_json(self) -> str:
        """Chuyển đổi thành JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class ClinicalProtocolManager:
    """
    Manager quản lý các clinical protocols.

    Chịu trách nhiệm lưu trữ, tải, và quản lý các protocol lâm sàng.
    """

    def __init__(self, protocols_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Default protocols directory
        if protocols_dir is None:
            protocols_dir = os.path.join(
                os.path.expanduser("~"), ".quangtps", "protocols"
            )

        self.protocols_dir = Path(protocols_dir)
        self.protocols_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded protocols
        self._protocols_cache: Dict[str, ClinicalProtocol] = {}

        # Initialize with standard protocols
        self._initialize_standard_protocols()

        self.logger.info(
            f"Initialized ClinicalProtocolManager with directory: {self.protocols_dir}"
        )

    def _initialize_standard_protocols(self):
        """Khởi tạo các protocols chuẩn."""
        try:
            # Create prostate protocol if not exists
            prostate_file = self.protocols_dir / "prostate_standard.json"
            if not prostate_file.exists():
                prostate_protocol = self._create_prostate_protocol()
                self.save_protocol(prostate_protocol)

            # Create head & neck protocol if not exists
            hn_file = self.protocols_dir / "head_neck_standard.json"
            if not hn_file.exists():
                hn_protocol = self._create_head_neck_protocol()
                self.save_protocol(hn_protocol)

            self.logger.info("Standard protocols initialized")

        except Exception as e:
            self.logger.error(f"Error initializing standard protocols: {e}")

    def _create_prostate_protocol(self) -> ClinicalProtocol:
        """Tạo protocol chuẩn cho prostate."""
        protocol = ClinicalProtocol(
            name="Prostate Standard",
            site="prostate",
            description="Standard clinical protocol for prostate radiotherapy",
            institution="QuangTPS Hospital",
            department="Radiation Oncology",
            fractionation="conventional",
            prescription_dose=78.0,
        )

        # Add standard goals
        goals = [
            {
                "structure_name": "PTV",
                "goal_type": "dose_volume",
                "target_value": 95.0,
                "comparison": "ge",
                "priority": "critical",
                "description": "PTV D95% >= 95% prescription",
                "units": "percent",
            },
            {
                "structure_name": "PTV",
                "goal_type": "dose_volume",
                "target_value": 107.0,
                "comparison": "le",
                "priority": "important",
                "description": "PTV D2% <= 107% prescription",
                "units": "percent",
            },
            {
                "structure_name": "Rectum",
                "goal_type": "volume_dose",
                "target_value": 35.0,
                "comparison": "lt",
                "priority": "important",
                "description": "Rectum V70Gy < 35%",
                "units": "percent",
            },
            {
                "structure_name": "Rectum",
                "goal_type": "volume_dose",
                "target_value": 50.0,
                "comparison": "lt",
                "priority": "important",
                "description": "Rectum V50Gy < 50%",
                "units": "percent",
            },
            {
                "structure_name": "Bladder",
                "goal_type": "volume_dose",
                "target_value": 50.0,
                "comparison": "lt",
                "priority": "important",
                "description": "Bladder V50Gy < 50%",
                "units": "percent",
            },
            {
                "structure_name": "Femoral_Head_L",
                "goal_type": "volume_dose",
                "target_value": 10.0,
                "comparison": "lt",
                "priority": "optional",
                "description": "Femoral Head L V50Gy < 10%",
                "units": "percent",
            },
            {
                "structure_name": "Femoral_Head_R",
                "goal_type": "volume_dose",
                "target_value": 10.0,
                "comparison": "lt",
                "priority": "optional",
                "description": "Femoral Head R V50Gy < 10%",
                "units": "percent",
            },
        ]

        for goal in goals:
            protocol.add_goal(goal)

        return protocol

    def _create_head_neck_protocol(self) -> ClinicalProtocol:
        """Tạo protocol chuẩn cho head & neck."""
        protocol = ClinicalProtocol(
            name="Head & Neck Standard",
            site="head_neck",
            description="Standard clinical protocol for head and neck radiotherapy",
            institution="QuangTPS Hospital",
            department="Radiation Oncology",
            fractionation="conventional",
            prescription_dose=70.0,
        )

        # Add standard goals
        goals = [
            {
                "structure_name": "PTV_70",
                "goal_type": "dose_volume",
                "target_value": 95.0,
                "comparison": "ge",
                "priority": "critical",
                "description": "PTV70 D95% >= 95% prescription",
                "units": "percent",
            },
            {
                "structure_name": "Spinal_Cord",
                "goal_type": "max_dose",
                "target_value": 45.0,
                "comparison": "lt",
                "priority": "critical",
                "description": "Spinal Cord Dmax < 45 Gy",
                "units": "Gy",
            },
            {
                "structure_name": "Brainstem",
                "goal_type": "max_dose",
                "target_value": 54.0,
                "comparison": "lt",
                "priority": "critical",
                "description": "Brainstem Dmax < 54 Gy",
                "units": "Gy",
            },
            {
                "structure_name": "Parotid_L",
                "goal_type": "mean_dose",
                "target_value": 26.0,
                "comparison": "lt",
                "priority": "important",
                "description": "Parotid L Dmean < 26 Gy",
                "units": "Gy",
            },
            {
                "structure_name": "Parotid_R",
                "goal_type": "mean_dose",
                "target_value": 26.0,
                "comparison": "lt",
                "priority": "important",
                "description": "Parotid R Dmean < 26 Gy",
                "units": "Gy",
            },
            {
                "structure_name": "Mandible",
                "goal_type": "max_dose",
                "target_value": 70.0,
                "comparison": "le",
                "priority": "optional",
                "description": "Mandible Dmax <= 70 Gy",
                "units": "Gy",
            },
        ]

        for goal in goals:
            protocol.add_goal(goal)

        return protocol

    def load_protocol(self, protocol_name: str) -> Optional[ClinicalProtocol]:
        """
        Tải protocol từ file.

        Parameters:
            protocol_name: Tên protocol (không bao gồm extension)

        Returns:
            ClinicalProtocol object hoặc None nếu không tìm thấy
        """
        try:
            # Check cache first
            if protocol_name in self._protocols_cache:
                return self._protocols_cache[protocol_name]

            # Look for the file
            protocol_file = self.protocols_dir / f"{protocol_name}.json"
            if not protocol_file.exists():
                self.logger.warning(f"Protocol file not found: {protocol_file}")
                return None

            # Load from file
            with open(protocol_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            protocol = ClinicalProtocol.from_dict(data)

            # Cache it
            self._protocols_cache[protocol_name] = protocol

            self.logger.info(f"Loaded protocol: {protocol_name}")
            return protocol

        except Exception as e:
            self.logger.error(f"Error loading protocol {protocol_name}: {e}")
            return None

    def save_protocol(self, protocol: ClinicalProtocol) -> bool:
        """
        Lưu protocol ra file.

        Parameters:
            protocol: ClinicalProtocol object

        Returns:
            bool: True nếu thành công
        """
        try:
            # Create safe filename
            safe_name = self._create_safe_filename(protocol.name)
            protocol_file = self.protocols_dir / f"{safe_name}.json"

            # Update modified date
            protocol.modified_date = datetime.now()

            # Save to file
            with open(protocol_file, "w", encoding="utf-8") as f:
                json.dump(protocol.to_dict(), f, indent=2, ensure_ascii=False)

            # Update cache
            self._protocols_cache[safe_name] = protocol

            self.logger.info(f"Saved protocol: {protocol.name} -> {protocol_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving protocol {protocol.name}: {e}")
            return False

    def get_protocol(self, protocol_name: str) -> Optional[ClinicalProtocol]:
        """
        Lấy protocol theo tên.

        Alias cho load_protocol để tương thích với code hiện có.

        Parameters:
            protocol_name: Tên protocol

        Returns:
            ClinicalProtocol hoặc None nếu không tìm thấy
        """
        return self.load_protocol(protocol_name)

    def list_protocols(self) -> List[str]:
        """
        Liệt kê tất cả protocols có sẵn.

        Returns:
            List tên các protocols
        """
        try:
            protocols = []

            # Scan directory for JSON files
            for protocol_file in self.protocols_dir.glob("*.json"):
                protocol_name = protocol_file.stem
                protocols.append(protocol_name)

            return sorted(protocols)

        except Exception as e:
            self.logger.error(f"Error listing protocols: {e}")
            return []

    def delete_protocol(self, protocol_name: str) -> bool:
        """
        Xóa protocol.

        Parameters:
            protocol_name: Tên protocol

        Returns:
            bool: True nếu thành công
        """
        try:
            protocol_file = self.protocols_dir / f"{protocol_name}.json"

            if protocol_file.exists():
                protocol_file.unlink()

                # Remove from cache
                if protocol_name in self._protocols_cache:
                    del self._protocols_cache[protocol_name]

                self.logger.info(f"Deleted protocol: {protocol_name}")
                return True
            else:
                self.logger.warning(f"Protocol not found for deletion: {protocol_name}")
                return False

        except Exception as e:
            self.logger.error(f"Error deleting protocol {protocol_name}: {e}")
            return False

    def get_protocols_by_site(self, site: str) -> List[str]:
        """
        Lấy danh sách protocols theo site.

        Parameters:
            site: Tên site (prostate, head_neck, etc.)

        Returns:
            List tên protocols
        """
        matching_protocols = []

        for protocol_name in self.list_protocols():
            protocol = self.load_protocol(protocol_name)
            if protocol and protocol.site.lower() == site.lower():
                matching_protocols.append(protocol_name)

        return matching_protocols

    def create_protocol_from_template(
        self,
        template_site: str,
        new_name: str,
        institution: str = "",
        department: str = "",
    ) -> Optional[ClinicalProtocol]:
        """
        Tạo protocol mới từ template.

        Parameters:
            template_site: Site template (prostate, head_neck)
            new_name: Tên protocol mới
            institution: Tên cơ sở
            department: Tên khoa

        Returns:
            ClinicalProtocol mới hoặc None
        """
        try:
            if template_site.lower() == "prostate":
                protocol = self._create_prostate_protocol()
            elif template_site.lower() == "head_neck":
                protocol = self._create_head_neck_protocol()
            else:
                self.logger.error(f"Unknown template site: {template_site}")
                return None

            # Customize
            protocol.name = new_name
            protocol.institution = institution
            protocol.department = department
            protocol.created_date = datetime.now()
            protocol.modified_date = datetime.now()

            return protocol

        except Exception as e:
            self.logger.error(f"Error creating protocol from template: {e}")
            return None

    def _create_safe_filename(self, name: str) -> str:
        """Tạo tên file an toàn."""
        import re

        # Remove special characters
        safe_name = re.sub(r"[^\w\s-]", "", name)
        # Replace spaces with underscores
        safe_name = re.sub(r"[-\s]+", "_", safe_name)
        return safe_name.lower()

    def export_protocol(self, protocol_name: str, output_path: str) -> bool:
        """
        Xuất protocol ra file.

        Parameters:
            protocol_name: Tên protocol
            output_path: Đường dẫn file xuất

        Returns:
            bool: True nếu thành công
        """
        try:
            protocol = self.load_protocol(protocol_name)
            if not protocol:
                return False

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(protocol.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported protocol {protocol_name} to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting protocol: {e}")
            return False

    def import_protocol(self, input_path: str) -> Optional[str]:
        """
        Nhập protocol từ file.

        Parameters:
            input_path: Đường dẫn file nhập

        Returns:
            str: Tên protocol đã nhập hoặc None
        """
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            protocol = ClinicalProtocol.from_dict(data)

            if self.save_protocol(protocol):
                safe_name = self._create_safe_filename(protocol.name)
                self.logger.info(f"Imported protocol: {protocol.name}")
                return safe_name
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error importing protocol: {e}")
            return None


# Utility functions
def get_available_sites() -> List[str]:
    """Lấy danh sách sites có sẵn."""
    return ["prostate", "head_neck", "breast", "lung", "brain", "abdomen", "pelvis"]


def create_protocol_manager(
    protocols_dir: Optional[str] = None,
) -> ClinicalProtocolManager:
    """Tạo protocol manager."""
    return ClinicalProtocolManager(protocols_dir)


# Global instance
_default_manager: Optional[ClinicalProtocolManager] = None


def get_default_protocol_manager() -> ClinicalProtocolManager:
    """Lấy default protocol manager (singleton)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ClinicalProtocolManager()
    return _default_manager


# Alias cho tương thích
def get_protocol(protocol_name: str) -> Optional[ClinicalProtocol]:
    """
    Lấy protocol theo tên từ default manager.

    Args:
        protocol_name: Tên protocol

    Returns:
        Protocol hoặc None nếu không tìm thấy
    """
    try:
        manager = get_default_protocol_manager()
        return manager.get_protocol(protocol_name)
    except Exception as e:
        logger.error(f"Error getting protocol '{protocol_name}': {e}")
        return None


def select_protocol_dialog(parent=None):
    """
    Hiển thị dialog để chọn protocol lâm sàng.

    Args:
        parent: Widget cha

    Returns:
        Dict chứa thông tin protocol đã chọn hoặc None
    """
    try:
        # Lazy import để tránh circular dependencies
        from PyQt5.QtWidgets import QInputDialog, QMessageBox

        manager = get_default_protocol_manager()
        available_protocols = manager.list_protocols()

        if not available_protocols:
            if parent:
                QMessageBox.information(
                    parent, "Thông báo", "Không có protocol nào khả dụng."
                )
            return None

        # Hiển thị dialog chọn protocol
        protocol_name, ok = QInputDialog.getItem(
            parent,
            "Chọn Protocol Lâm sàng",
            "Chọn protocol để đánh giá:",
            available_protocols,
            0,
            False,
        )

        if ok and protocol_name:
            protocol = manager.get_protocol(protocol_name)
            if protocol:
                # Chuyển đổi thành dict để compatibility
                return {
                    "name": protocol.name,
                    "site": protocol.site,
                    "description": protocol.description,
                    "version": protocol.version,
                    "clinical_goals": [goal.__dict__ for goal in protocol.goals],
                    "created_by": protocol.created_by,
                    "institution": protocol.institution,
                    "department": protocol.department,
                    "fractionation": protocol.fractionation,
                    "prescription_dose": protocol.prescription_dose,
                }

        return None

    except ImportError:
        logger.warning("PyQt5 không khả dụng. Không thể hiển thị dialog.")
        return None
    except Exception as e:
        logger.error(f"Error in select_protocol_dialog: {e}")
        return None


def create_simple_protocol_dialog(parent=None):
    """
    Tạo dialog đơn giản để chọn protocol khi UI phức tạp không khả dụng.

    Args:
        parent: Widget cha

    Returns:
        Dict chứa thông tin protocol đã chọn hoặc None
    """
    try:
        manager = get_default_protocol_manager()
        protocols = manager.list_protocols()

        if not protocols:
            logger.info("Không có protocol nào khả dụng")
            return None

        # Lấy protocol đầu tiên làm mặc định
        protocol_name = protocols[0]
        protocol = manager.get_protocol(protocol_name)

        if protocol:
            logger.info(f"Sử dụng protocol mặc định: {protocol_name}")
            return {
                "name": protocol.name,
                "site": protocol.site,
                "description": protocol.description,
                "clinical_goals": [goal.__dict__ for goal in protocol.goals],
            }

        return None

    except Exception as e:
        logger.error(f"Error in create_simple_protocol_dialog: {e}")
        return None


def get_protocol_by_site(site: str) -> Optional[ClinicalProtocol]:
    """
    Lấy protocol phù hợp với site điều trị.

    Args:
        site: Vị trí điều trị

    Returns:
        Protocol phù hợp hoặc None
    """
    try:
        manager = get_default_protocol_manager()
        protocols_for_site = manager.get_protocols_by_site(site)

        if protocols_for_site:
            # Lấy protocol đầu tiên cho site này
            return manager.get_protocol(protocols_for_site[0])

        return None

    except Exception as e:
        logger.error(f"Error getting protocol for site '{site}': {e}")
        return None


def export_protocol_to_file(protocol: ClinicalProtocol, file_path: str) -> bool:
    """
    Xuất protocol ra file.

    Args:
        protocol: Protocol cần xuất
        file_path: Đường dẫn file

    Returns:
        True nếu thành công
    """
    try:
        manager = get_default_protocol_manager()
        return manager.export_protocol(protocol.name, file_path)
    except Exception as e:
        logger.error(f"Error exporting protocol: {e}")
        return False


def import_protocol_from_file(file_path: str) -> Optional[str]:
    """
    Nhập protocol từ file.

    Args:
        file_path: Đường dẫn file

    Returns:
        Tên protocol đã nhập hoặc None
    """
    try:
        manager = get_default_protocol_manager()
        return manager.import_protocol(file_path)
    except Exception as e:
        logger.error(f"Error importing protocol: {e}")
        return None


def validate_protocol_data(protocol_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate dữ liệu protocol.

    Args:
        protocol_data: Dữ liệu protocol

    Returns:
        Tuple (valid, error_message)
    """
    try:
        if not protocol_data.get("name"):
            return False, "Tên protocol là bắt buộc"

        if not protocol_data.get("site"):
            return False, "Vị trí điều trị là bắt buộc"

        goals = protocol_data.get("clinical_goals", [])
        if not goals:
            return False, "Ít nhất một mục tiêu lâm sàng là bắt buộc"

        return True, ""

    except Exception as e:
        return False, f"Lỗi validation: {e}"


def load_protocol(
    protocol_name: str, protocols_dir: Optional[str] = None
) -> Optional[ClinicalProtocol]:
    """
    Load a protocol from file (function-level interface).

    Args:
        protocol_name: Name of protocol to load
        protocols_dir: Directory containing protocols (optional)

    Returns:
        ClinicalProtocol object or None if not found
    """
    manager = ClinicalProtocolManager(protocols_dir)
    return manager.load_protocol(protocol_name)


def save_default_protocols(protocols_dir: str):
    """
    Save default protocols to specified directory.

    Args:
        protocols_dir: Directory to save protocols
    """
    manager = ClinicalProtocolManager(protocols_dir)
    # Default protocols are automatically created in __init__
    logger.info(f"Default protocols saved to {protocols_dir}")


# Export
__all__ = [
    "ClinicalProtocol",
    "ClinicalProtocolManager",
    "load_protocol",
    "save_default_protocols",
    "get_protocol",
    "get_available_sites",
    "create_protocol_manager",
    "get_default_protocol_manager",
    "select_protocol_dialog",
    "get_protocol_by_site",
    "export_protocol_to_file",
    "import_protocol_from_file",
    "validate_protocol_data",
]
