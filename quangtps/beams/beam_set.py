"""
Lớp BeamSet đại diện cho một tập hợp các chùm tia trong một kế hoạch xạ trị.
"""

import uuid
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from enum import Enum

from .beam import Beam, BeamType
from .isocenter import Isocenter


class BeamSetType(Enum):
    """Loại tập hợp chùm tia."""

    MANUAL = "MANUAL"  # Tạo thủ công
    STATIC = "STATIC"  # Các chùm tĩnh
    CONFORMAL = "CONFORMAL"  # Chùm tia 3D conformal
    IMRT = "IMRT"  # IMRT
    VMAT = "VMAT"  # VMAT
    ARC = "ARC"  # Kỹ thuật arc
    SBRT = "SBRT"  # SBRT
    ELECTRON = "ELECTRON"  # Điện tử
    MIXED = "MIXED"  # Kết hợp nhiều loại
    UNKNOWN = "UNKNOWN"  # Không xác định


class BeamSet:
    """
    Lớp đại diện cho một tập hợp các chùm tia trong một kế hoạch xạ trị.

    Thuộc tính
    ----------
    id : str
        ID của tập hợp chùm tia
    name : str
        Tên của tập hợp chùm tia
    beams : List[Beam]
        Danh sách các chùm tia
    beam_set_type : BeamSetType
        Loại tập hợp chùm tia
    machine : str
        Máy điều trị
    prescription : Optional[Dict[str, Any]]
        Kê đơn liều cho tập hợp chùm tia
    primary_isocenter : Optional[Isocenter]
        Tâm iso chính
    """

    def __init__(
        self,
        name: str,
        beams: Optional[List[Beam]] = None,
        beam_set_type: BeamSetType = BeamSetType.MANUAL,
        machine: str = "TrueBeam",
        prescription: Optional[Dict[str, Any]] = None,
        primary_isocenter: Optional[Isocenter] = None,
    ):
        """
        Khởi tạo một tập hợp chùm tia mới.

        Parameters
        ----------
        name : str
            Tên của tập hợp chùm tia
        beams : Optional[List[Beam]], optional
            Danh sách các chùm tia, mặc định là rỗng
        beam_set_type : BeamSetType, optional
            Loại tập hợp chùm tia, mặc định là MANUAL
        machine : str, optional
            Máy điều trị, mặc định là "TrueBeam"
        prescription : Optional[Dict[str, Any]], optional
            Kê đơn liều, mặc định là None
        primary_isocenter : Optional[Isocenter], optional
            Tâm iso chính, mặc định là None
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.beams = beams if beams is not None else []
        self.beam_set_type = (
            beam_set_type
            if isinstance(beam_set_type, BeamSetType)
            else BeamSetType.MANUAL
        )
        self.machine = machine
        self.prescription = prescription if prescription is not None else {}
        self.primary_isocenter = primary_isocenter

        # Cập nhật loại tập hợp chùm tia dựa trên các chùm tia hiện có
        self._update_beam_set_type()

    def add_beam(self, beam: Beam) -> None:
        """
        Thêm một chùm tia vào tập hợp.

        Parameters
        ----------
        beam : Beam
            Chùm tia cần thêm
        """
        self.beams.append(beam)
        self._update_beam_set_type()

    def remove_beam(self, beam_id: str) -> bool:
        """
        Xóa một chùm tia khỏi tập hợp.

        Parameters
        ----------
        beam_id : str
            ID của chùm tia cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        for i, beam in enumerate(self.beams):
            if beam.id == beam_id:
                self.beams.pop(i)
                self._update_beam_set_type()
                return True
        return False

    def get_beam(self, beam_id: str) -> Optional[Beam]:
        """
        Lấy chùm tia theo ID.

        Parameters
        ----------
        beam_id : str
            ID của chùm tia cần lấy

        Returns
        -------
        Optional[Beam]
            Chùm tia nếu tìm thấy, None nếu không
        """
        for beam in self.beams:
            if beam.id == beam_id:
                return beam
        return None

    def get_beam_by_name(self, name: str) -> Optional[Beam]:
        """
        Lấy chùm tia theo tên.

        Parameters
        ----------
        name : str
            Tên của chùm tia cần lấy

        Returns
        -------
        Optional[Beam]
            Chùm tia đầu tiên có tên khớp, None nếu không tìm thấy
        """
        for beam in self.beams:
            if beam.name == name:
                return beam
        return None

    def set_prescription(
        self,
        dose: float,
        fraction_count: int,
        fraction_dose: Optional[float] = None,
        target: Optional[str] = None,
        prescription_type: str = "TotalDose",
    ) -> None:
        """
        Thiết lập kê đơn liều cho tập hợp chùm tia.

        Parameters
        ----------
        dose : float
            Tổng liều (Gy)
        fraction_count : int
            Số phân liều
        fraction_dose : Optional[float], optional
            Liều mỗi phân liều (Gy), mặc định tính từ dose/fraction_count
        target : Optional[str], optional
            Tên cấu trúc đích, mặc định là None
        prescription_type : str, optional
            Loại kê đơn, mặc định là "TotalDose"
        """
        if fraction_dose is None:
            fraction_dose = dose / fraction_count if fraction_count > 0 else 0

        self.prescription = {
            "dose": dose,
            "fraction_count": fraction_count,
            "fraction_dose": fraction_dose,
            "target": target,
            "type": prescription_type,
        }

    def _update_beam_set_type(self) -> None:
        """Cập nhật loại tập hợp chùm tia dựa trên các chùm tia hiện có."""
        if not self.beams:
            self.beam_set_type = BeamSetType.MANUAL
            return

        # Lấy tất cả các loại chùm tia
        beam_types = set(beam.beam_type for beam in self.beams)

        # Nếu chỉ có một loại chùm tia, phân loại dựa trên loại đó
        if len(beam_types) == 1:
            beam_type = next(iter(beam_types))
            if beam_type == BeamType.STATIC:
                self.beam_set_type = BeamSetType.STATIC
            elif beam_type == BeamType.DYNAMIC:
                self.beam_set_type = BeamSetType.IMRT
            elif beam_type == BeamType.ARC:
                self.beam_set_type = BeamSetType.ARC
            elif beam_type == BeamType.VMAT:
                self.beam_set_type = BeamSetType.VMAT
            elif beam_type == BeamType.IMRT:
                self.beam_set_type = BeamSetType.IMRT
            elif beam_type == BeamType.ELECTRON:
                self.beam_set_type = BeamSetType.ELECTRON
            else:
                self.beam_set_type = BeamSetType.MANUAL
        else:
            # Nếu có nhiều loại chùm tia, đánh dấu là MIXED
            self.beam_set_type = BeamSetType.MIXED

    def normalize_weights(self) -> None:
        """Chuẩn hóa trọng số của các chùm tia để tổng bằng 1."""
        if not self.beams:
            return

        total_weight = sum(beam.weight for beam in self.beams)
        if total_weight <= 0:
            # Nếu tổng trọng số bằng 0, thiết lập trọng số đồng đều
            equal_weight = 1.0 / len(self.beams)
            for beam in self.beams:
                beam.weight = equal_weight
        else:
            # Chuẩn hóa trọng số
            for beam in self.beams:
                beam.weight = beam.weight / total_weight

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi tập hợp chùm tia thành từ điển.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin tập hợp chùm tia
        """
        return {
            "id": self.id,
            "name": self.name,
            "beams": [beam.to_dict() for beam in self.beams],
            "beam_set_type": self.beam_set_type.value,
            "machine": self.machine,
            "prescription": self.prescription,
            "primary_isocenter": (
                self.primary_isocenter.to_dict() if self.primary_isocenter else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamSet":
        """
        Tạo tập hợp chùm tia từ từ điển.

        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin tập hợp chùm tia

        Returns
        -------
        BeamSet
            Đối tượng tập hợp chùm tia mới
        """
        beam_set = cls(
            name=data["name"],
            beam_set_type=BeamSetType(data["beam_set_type"]),
            machine=data.get("machine", "TrueBeam"),
            prescription=data.get("prescription"),
            primary_isocenter=(
                Isocenter.from_dict(data["primary_isocenter"])
                if data.get("primary_isocenter")
                else None
            ),
        )

        # Đặt lại ID nếu có trong dữ liệu
        if "id" in data:
            beam_set.id = data["id"]

        # Tạo các chùm tia
        for beam_data in data.get("beams", []):
            beam_set.beams.append(Beam.from_dict(beam_data))

        return beam_set

    def __str__(self) -> str:
        """
        Biểu diễn chuỗi của tập hợp chùm tia.

        Returns
        -------
        str
            Chuỗi mô tả tập hợp chùm tia
        """
        return f"BeamSet(name={self.name}, type={self.beam_set_type.value}, beams={len(self.beams)})"
