"""
Lớp ControlPoint đại diện cho một điểm điều khiển của chùm tia xạ trị.

Trong xạ trị, control point đại diện cho một trạng thái cụ thể của máy xạ trị
(gantry, MLC, jaw, v.v.) tại một thời điểm trong quá trình điều trị.
"""

from typing import Dict, Any, Optional, List, Union, Tuple

from .mlc_position import MLCPosition


class ControlPoint:
    """
    Lớp đại diện cho một điểm điều khiển của chùm tia xạ trị.

    Thuộc tính
    ----------
    index : int
        Chỉ số của điểm điều khiển
    mu_weight : float
        Trọng số MU (0-1) tại điểm điều khiển
    mlc_positions : Optional[MLCPosition]
        Vị trí của các lá MLC
    gantry_angle : Optional[float]
        Góc gantry (độ)
    collimator_angle : Optional[float]
        Góc collimator (độ)
    couch_angle : Optional[float]
        Góc bàn (độ)
    jaw_positions : Optional[Dict[str, float]]
        Vị trí các jaw (X1, X2, Y1, Y2) tính bằng mm
    """

    def __init__(
        self,
        index: int,
        mu_weight: float,
        mlc_positions: Optional[MLCPosition] = None,
        gantry_angle: Optional[float] = None,
        collimator_angle: Optional[float] = None,
        couch_angle: Optional[float] = None,
        jaw_positions: Optional[Dict[str, float]] = None,
    ):
        """
        Khởi tạo một điểm điều khiển mới.

        Parameters
        ----------
        index : int
            Chỉ số của điểm điều khiển
        mu_weight : float
            Trọng số MU (0-1) tại điểm điều khiển
        mlc_positions : Optional[MLCPosition], optional
            Vị trí của các lá MLC, mặc định là None
        gantry_angle : Optional[float], optional
            Góc gantry (độ), mặc định là None
        collimator_angle : Optional[float], optional
            Góc collimator (độ), mặc định là None
        couch_angle : Optional[float], optional
            Góc bàn (độ), mặc định là None
        jaw_positions : Optional[Dict[str, float]], optional
            Vị trí các jaw (X1, X2, Y1, Y2) tính bằng mm, mặc định là None
        """
        self.index = index
        self.mu_weight = mu_weight
        self.mlc_positions = mlc_positions
        self.gantry_angle = gantry_angle
        self.collimator_angle = collimator_angle
        self.couch_angle = couch_angle
        self.jaw_positions = jaw_positions

    def set_mlc_positions(self, mlc_positions: MLCPosition) -> None:
        """
        Thiết lập vị trí các lá MLC.

        Parameters
        ----------
        mlc_positions : MLCPosition
            Vị trí của các lá MLC
        """
        self.mlc_positions = mlc_positions

    def set_jaw_positions(
        self,
        x1: Optional[float] = None,
        x2: Optional[float] = None,
        y1: Optional[float] = None,
        y2: Optional[float] = None,
    ) -> None:
        """
        Thiết lập vị trí các jaw.

        Parameters
        ----------
        x1 : Optional[float], optional
            Vị trí jaw X1 (mm), mặc định là giữ nguyên
        x2 : Optional[float], optional
            Vị trí jaw X2 (mm), mặc định là giữ nguyên
        y1 : Optional[float], optional
            Vị trí jaw Y1 (mm), mặc định là giữ nguyên
        y2 : Optional[float], optional
            Vị trí jaw Y2 (mm), mặc định là giữ nguyên
        """
        if self.jaw_positions is None:
            self.jaw_positions = {}

        if x1 is not None:
            self.jaw_positions["X1"] = x1
        if x2 is not None:
            self.jaw_positions["X2"] = x2
        if y1 is not None:
            self.jaw_positions["Y1"] = y1
        if y2 is not None:
            self.jaw_positions["Y2"] = y2

    def calculate_field_size(self) -> Tuple[float, float]:
        """
        Tính toán kích thước trường dựa vào vị trí jaw.

        Returns
        -------
        Tuple[float, float]
            Kích thước trường (width, height) tính bằng mm, hoặc (0, 0) nếu không có jaw
        """
        if not self.jaw_positions:
            return (0.0, 0.0)

        try:
            width = abs(
                self.jaw_positions.get("X2", 0) - self.jaw_positions.get("X1", 0)
            )
            height = abs(
                self.jaw_positions.get("Y2", 0) - self.jaw_positions.get("Y1", 0)
            )
            return (width, height)
        except (TypeError, KeyError):
            return (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi điểm điều khiển thành từ điển.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin điểm điều khiển
        """
        return {
            "index": self.index,
            "mu_weight": self.mu_weight,
            "mlc_positions": self.mlc_positions.to_dict()
            if self.mlc_positions
            else None,
            "gantry_angle": self.gantry_angle,
            "collimator_angle": self.collimator_angle,
            "couch_angle": self.couch_angle,
            "jaw_positions": self.jaw_positions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlPoint":
        """
        Tạo điểm điều khiển từ từ điển.

        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin điểm điều khiển

        Returns
        -------
        ControlPoint
            Đối tượng điểm điều khiển mới
        """
        mlc_positions = None
        if data.get("mlc_positions"):
            from .mlc_position import MLCPosition

            mlc_positions = MLCPosition.from_dict(data["mlc_positions"])

        return cls(
            index=data["index"],
            mu_weight=data["mu_weight"],
            mlc_positions=mlc_positions,
            gantry_angle=data.get("gantry_angle"),
            collimator_angle=data.get("collimator_angle"),
            couch_angle=data.get("couch_angle"),
            jaw_positions=data.get("jaw_positions"),
        )

    def __str__(self) -> str:
        """
        Biểu diễn chuỗi của điểm điều khiển.

        Returns
        -------
        str
            Chuỗi mô tả điểm điều khiển
        """
        angle_str = ""
        if self.gantry_angle is not None:
            angle_str += f", gantry={self.gantry_angle:.1f}°"

        return f"ControlPoint(index={self.index}, mu_weight={self.mu_weight:.3f}{angle_str})"
