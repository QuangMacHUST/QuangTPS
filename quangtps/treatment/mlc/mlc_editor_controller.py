#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLC Editor Controller - Lớp điều khiển cho trình soạn thảo MLC.

Module này cung cấp lớp điều khiển trung gian giữa giao diện người dùng
MLC Editor và các thuật toán tối ưu hóa MLC. Nó xử lý logic nghiệp vụ
cho việc chỉnh sửa và tối ưu hóa MLC.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

from quangtps.planning.mlc import MLC
from quangtps.imaging.structures import Structure
from quangtps.treatment.beams.beam import Beam
from quangtps.optimization.mlc_optimization import optimize_mlc_shape

logger = logging.getLogger(__name__)


class MLCEditorController:
    """
    Lớp điều khiển cho MLC Editor giúp kết nối giao diện người dùng
    với các thuật toán tối ưu hóa.
    """

    def __init__(self):
        """Khởi tạo controller với các tham số mặc định."""
        self._current_mlc = None
        self._target_structure = None
        self._oar_structures = []
        self._current_beam = None
        self._optimization_params = {
            "algorithm": "gradient",  # gradient, simulated_annealing, genetic
            "iterations": 100,
            "convergence_threshold": 0.001,
        }

    def set_mlc(self, mlc: MLC):
        """
        Thiết lập MLC hiện tại cho controller.

        Parameters
        ----------
        mlc : MLC
            Đối tượng MLC
        """
        self._current_mlc = mlc

    def get_mlc(self) -> Optional[MLC]:
        """
        Lấy đối tượng MLC hiện tại.

        Returns
        -------
        Optional[MLC]
            Đối tượng MLC hiện tại hoặc None nếu chưa được thiết lập
        """
        return self._current_mlc

    def set_target_structure(self, structure: Structure):
        """
        Thiết lập cấu trúc mục tiêu.

        Parameters
        ----------
        structure : Structure
            Cấu trúc mục tiêu
        """
        self._target_structure = structure

    def add_oar_structure(self, structure: Structure):
        """
        Thêm một cấu trúc OAR vào danh sách.

        Parameters
        ----------
        structure : Structure
            Cấu trúc OAR cần thêm
        """
        if structure not in self._oar_structures:
            self._oar_structures.append(structure)

    def remove_oar_structure(self, structure: Structure) -> bool:
        """
        Xóa một cấu trúc OAR khỏi danh sách.

        Parameters
        ----------
        structure : Structure
            Cấu trúc OAR cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if structure in self._oar_structures:
            self._oar_structures.remove(structure)
            return True
        return False

    def clear_oar_structures(self):
        """Xóa tất cả các cấu trúc OAR."""
        self._oar_structures.clear()

    def set_current_beam(self, beam: Beam):
        """
        Thiết lập chùm tia hiện tại.

        Parameters
        ----------
        beam : Beam
            Chùm tia hiện tại
        """
        self._current_beam = beam

    def set_optimization_parameters(self, **params):
        """
        Thiết lập tham số tối ưu hóa.

        Parameters
        ----------
        **params : dict
            Dictionary chứa các tham số tối ưu hóa
        """
        for key, value in params.items():
            if key in self._optimization_params:
                self._optimization_params[key] = value

    def create_shape_based_mlc(
        self, structure: Structure, mlc_type: str, margin: float = 0.5
    ) -> Optional[MLC]:
        """
        Tạo hình dạng MLC dựa trên cấu trúc.

        Parameters
        ----------
        structure : Structure
            Cấu trúc mục tiêu
        mlc_type : str
            Loại MLC cần tạo
        margin : float, optional
            Lề (margin) xung quanh cấu trúc, mặc định là 0.5 cm

        Returns
        -------
        Optional[MLC]
            Đối tượng MLC đã được thiết kế theo hình dạng cấu trúc hoặc None nếu có lỗi
        """
        try:
            # Tạo MLC mới
            mlc = MLC(mlc_type)

            # Nếu không có chùm tia, sử dụng góc mặc định
            beam = self._current_beam

            # Kiểm tra xem cấu trúc có dữ liệu không
            if not structure.has_contour_data():
                logger.warning(f"Cấu trúc {structure.name} không có dữ liệu contour")
                return None

            # Chuyển cấu trúc sang góc nhìn BEV
            from quangtps.treatment.beams.beam_geometry import get_bev_transform

            if beam:
                transform = get_bev_transform(beam)
            else:
                # Tạo transform mặc định nếu không có beam
                from quangtps.treatment.beams.beam_geometry import BEVTransform

                transform = BEVTransform(0, 0, 0, (0, 0, 0))

            # Lấy contour của cấu trúc từ góc nhìn BEV
            bev_contour = self._get_structure_bev_contour(structure, transform)

            if not bev_contour:
                logger.warning(
                    f"Không thể tạo contour BEV cho cấu trúc {structure.name}"
                )
                return None

            # Tính toán vị trí lá MLC dựa trên contour BEV
            self._set_mlc_positions_from_contour(mlc, bev_contour, margin)

            return mlc

        except Exception as e:
            logger.error(f"Lỗi khi tạo MLC dựa trên cấu trúc: {str(e)}")
            return None

    def _get_structure_bev_contour(
        self, structure: Structure, transform
    ) -> List[Tuple[float, float]]:
        """
        Lấy contour của cấu trúc từ góc nhìn BEV.

        Parameters
        ----------
        structure : Structure
            Cấu trúc cần chuyển đổi
        transform : BEVTransform
            Đối tượng biến đổi BEV

        Returns
        -------
        List[Tuple[float, float]]
            Danh sách các điểm contour trong hệ tọa độ BEV
        """
        try:
            # Lấy điểm bề mặt của cấu trúc
            points = structure.get_surface_points()

            if not points or len(points) == 0:
                return []

            # Chuyển sang tọa độ BEV
            bev_points = transform.transform_points(points)

            # Tạo contour từ các điểm BEV
            # Sử dụng thuật toán tạo contour đơn giản
            import numpy as np

            if len(bev_points) >= 3:  # Cần ít nhất 3 điểm để tạo contour
                try:
                    # Sử dụng phương pháp bao lồi đơn giản thay vì ConvexHull
                    # Tính tâm của các điểm
                    center = np.mean(bev_points, axis=0)

                    # Tính góc từ tâm đến mỗi điểm
                    angles = np.arctan2(
                        bev_points[:, 1] - center[1], bev_points[:, 0] - center[0]
                    )

                    # Sắp xếp các điểm theo góc tăng dần
                    sorted_indices = np.argsort(angles)
                    contour = np.array([bev_points[i] for i in sorted_indices])

                    return contour.tolist()
                except Exception as e:
                    logger.warning(f"Không thể tạo bao lồi: {str(e)}")

            # Fallback nếu không thể tạo bao lồi
            return bev_points.tolist()

        except Exception as e:
            logger.error(f"Lỗi khi lấy contour BEV: {str(e)}")
            return []

    def _set_mlc_positions_from_contour(
        self, mlc: MLC, contour: List[Tuple[float, float]], margin: float
    ):
        """
        Thiết lập vị trí lá MLC dựa trên contour BEV.

        Parameters
        ----------
        mlc : MLC
            Đối tượng MLC cần thiết lập
        contour : List[Tuple[float, float]]
            Contour trong hệ tọa độ BEV
        margin : float
            Lề (margin) xung quanh contour
        """
        if not contour:
            return

        import numpy as np

        # Chuyển contour thành mảng numpy
        contour_np = np.array(contour)

        # Sắp xếp lá MLC theo chỉ số
        leaves_by_index = {}
        for leaf in mlc.leaves:
            if leaf.index not in leaves_by_index:
                leaves_by_index[leaf.index] = {"A": None, "B": None}
            leaves_by_index[leaf.index][leaf.bank] = leaf

        # Lấy vị trí Y của mỗi lá
        leaf_y_positions = []
        for idx in sorted(leaves_by_index.keys()):
            if "A" in leaves_by_index[idx] and leaves_by_index[idx]["A"]:
                leaf_y_positions.append(leaves_by_index[idx]["A"].y_position)

        # Với mỗi cặp lá, tìm các điểm giao với contour
        for idx in sorted(leaves_by_index.keys()):
            leaf_pair = leaves_by_index[idx]

            # Lấy vị trí Y của lá hiện tại
            if "A" not in leaf_pair or leaf_pair["A"] is None:
                continue

            leaf_y = leaf_pair["A"].y_position
            leaf_half_width = leaf_pair["A"].width / 2

            # Tìm các điểm contour trong dải Y của lá
            mask = np.logical_and(
                contour_np[:, 1] >= leaf_y - leaf_half_width,
                contour_np[:, 1] < leaf_y + leaf_half_width,
            )
            points_in_leaf = contour_np[mask]

            if len(points_in_leaf) > 0:
                # Tìm điểm xa nhất bên trái và bên phải
                min_x = np.min(points_in_leaf[:, 0]) - margin
                max_x = np.max(points_in_leaf[:, 0]) + margin

                # Thiết lập vị trí lá
                if "A" in leaf_pair and leaf_pair["A"]:
                    leaf_pair["A"].position = min_x
                if "B" in leaf_pair and leaf_pair["B"]:
                    leaf_pair["B"].position = max_x
            else:
                # Nếu không có điểm nào, đặt lá về vị trí đóng
                if "A" in leaf_pair and leaf_pair["A"]:
                    leaf_pair["A"].position = 0
                if "B" in leaf_pair and leaf_pair["B"]:
                    leaf_pair["B"].position = 0

    def optimize_mlc(self, field_size: float = 40.0) -> Optional[MLC]:
        """
        Tối ưu hóa MLC hiện tại.

        Parameters
        ----------
        field_size : float, optional
            Kích thước trường tối đa (cm)

        Returns
        -------
        Optional[MLC]
            Đối tượng MLC đã được tối ưu hoặc None nếu có lỗi
        """
        if not self._current_mlc or not self._target_structure:
            logger.warning(
                "Không thể tối ưu hóa: MLC hoặc cấu trúc mục tiêu chưa được thiết lập"
            )
            return None

        try:
            # Gọi hàm tối ưu hóa
            optimized_mlc = optimize_mlc_shape(
                original_mlc=self._current_mlc,
                target=self._target_structure,
                oars=self._oar_structures,
                field_size=field_size,
                beam=self._current_beam,
                algorithm=self._optimization_params["algorithm"],
                iterations=self._optimization_params["iterations"],
                convergence_threshold=self._optimization_params[
                    "convergence_threshold"
                ],
            )

            # Cập nhật MLC hiện tại
            self._current_mlc = optimized_mlc

            return optimized_mlc

        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa MLC: {str(e)}")
            return None

    def validate_mlc(self) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính hợp lệ của MLC hiện tại.

        Returns
        -------
        Tuple[bool, List[str]]
            Tuple chứa kết quả kiểm tra (True/False) và danh sách các thông báo lỗi
        """
        if not self._current_mlc:
            return False, ["MLC chưa được thiết lập"]

        errors = []

        # Kiểm tra các ràng buộc vật lý
        for leaf in self._current_mlc.leaves:
            # Kiểm tra giới hạn di chuyển
            if abs(leaf.position) > self._current_mlc.max_leaf_travel:
                errors.append(
                    f"Lá {leaf.index} ({leaf.bank}) vượt quá giới hạn di chuyển"
                )

            # Kiểm tra va chạm giữa các lá đối diện
            opposite_bank = "B" if leaf.bank == "A" else "A"
            for other_leaf in self._current_mlc.leaves:
                if other_leaf.index == leaf.index and other_leaf.bank == opposite_bank:
                    if leaf.bank == "A" and leaf.position > other_leaf.position:
                        errors.append(
                            f"Lá {leaf.index}A va chạm với lá {other_leaf.index}B"
                        )
                    elif leaf.bank == "B" and leaf.position < other_leaf.position:
                        errors.append(
                            f"Lá {leaf.index}B va chạm với lá {other_leaf.index}A"
                        )

            # Kiểm tra chồng chéo giữa các lá liền kề
            for other_leaf in self._current_mlc.leaves:
                if (
                    other_leaf.bank == leaf.bank
                    and abs(other_leaf.index - leaf.index) == 1
                ):
                    # Hiện tại bỏ qua kiểm tra chồng chéo do phức tạp
                    pass

        return len(errors) == 0, errors

    def calculate_coverage_metrics(self) -> Dict[str, float]:
        """
        Tính toán các chỉ số bao phủ của MLC hiện tại.

        Returns
        -------
        Dict[str, float]
            Dictionary chứa các chỉ số bao phủ
        """
        if not self._current_mlc or not self._target_structure:
            return {}

        try:
            # Tạo bản đồ truyền qua từ MLC
            resolution = 100
            transmission_map = self._current_mlc.get_transmission_map(
                resolution=resolution
            )

            # Chuyển cấu trúc sang góc nhìn BEV
            from quangtps.optimization.mlc_optimization import _structure_to_bev_map

            field_size = 40.0  # Giả định kích thước trường
            bev_target_map = _structure_to_bev_map(
                self._target_structure, field_size, resolution, self._current_beam
            )

            # Tính chỉ số bao phủ mục tiêu
            target_coverage = 0
            if np.sum(bev_target_map) > 0:
                target_coverage = np.sum(transmission_map * bev_target_map) / np.sum(
                    bev_target_map
                )

            # Tính chỉ số bao phủ OARs
            oar_metrics = {}
            for i, oar in enumerate(self._oar_structures):
                bev_oar_map = _structure_to_bev_map(
                    oar, field_size, resolution, self._current_beam
                )

                oar_exposure = 0
                if np.sum(bev_oar_map) > 0:
                    oar_exposure = np.sum(transmission_map * bev_oar_map) / np.sum(
                        bev_oar_map
                    )

                oar_metrics[f"oar_{oar.name}_exposure"] = oar_exposure

            # Kết hợp các chỉ số
            metrics = {
                "target_coverage": target_coverage,
                **oar_metrics,
            }

            return metrics

        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ số bao phủ: {str(e)}")
            return {"error": str(e)}
