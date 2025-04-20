#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tính toán liều lượng xạ trị.

Cung cấp các lớp và phương thức để tính toán phân bố liều
từ các chùm tia xạ trị trong kế hoạch điều trị.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from quangtps.core.types import DoseGrid, BeamParameters
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.imaging.dicom_series import DicomSeries
from quangtps.imaging.image import Image
from quangtps.structures.structure_set import StructureSet
from quangtps.structures.structure import Structure
from quangtps.beams.beam import BeamSet

logger = logging.getLogger(__name__)


class DoseAlgorithmBase:
    """Lớp cơ sở cho các thuật toán tính liều."""

    def __init__(self):
        """Khởi tạo thuật toán tính liều."""
        self.name = "Base"
        self.version = "1.0"
        self.description = "Base dose calculation algorithm"
        self.parameters = {}

    def calculate_dose(
        self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid
    ) -> np.ndarray:
        """
        Tính toán phân bố liều cho một chùm tia.

        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán

        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")

    def get_name(self) -> str:
        """Lấy tên của thuật toán."""
        return self.name

    def get_version(self) -> str:
        """Lấy phiên bản của thuật toán."""
        return self.version

    def get_description(self) -> str:
        """Lấy mô tả của thuật toán."""
        return self.description

    def set_parameter(self, key: str, value: Any) -> None:
        """Đặt tham số cho thuật toán."""
        self.parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Lấy giá trị tham số."""
        return self.parameters.get(key, default)


class SimpleRayTracingAlgorithm(DoseAlgorithmBase):
    """Thuật toán tính liều đơn giản dựa trên ray tracing."""

    def __init__(self):
        """Khởi tạo thuật toán ray tracing đơn giản."""
        super().__init__()
        self.name = "SimpleRayTracing"
        self.description = "Simple ray tracing algorithm for demonstration"

        # Tham số mặc định
        self.parameters = {
            "attenuation_factor": 0.002,  # mm^-1
            "use_heterogeneity_correction": True,
            "source_to_isocenter_distance": 1000.0,  # mm
        }

    def calculate_dose(
        self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid
    ) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng ray tracing đơn giản.

        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán

        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        if ct_data is None or ct_data.size == 0:
            logger.error("Dữ liệu CT không hợp lệ")
            return np.zeros_like(dose_grid.data)

        if beam is None:
            logger.error("Chùm tia không hợp lệ")
            return np.zeros_like(dose_grid.data)

        # Khởi tạo mảng liều với giá trị 0
        dose = np.zeros_like(dose_grid.data)

        try:
            # Lấy thông tin chùm tia
            beam_params = self._get_beam_parameters(beam)

            # Lấy thông số tính toán
            attenuation = self.parameters["attenuation_factor"]
            use_heterogeneity = self.parameters["use_heterogeneity_correction"]
            sad = self.parameters["source_to_isocenter_distance"]

            # Tính toán hướng chùm tia
            beam_direction = self._calculate_beam_direction(beam_params)

            # Tính toán vị trí nguồn
            source_pos = self._calculate_source_position(
                beam_params, beam_direction, sad
            )

            # Thực hiện ray tracing
            for i in range(dose.shape[0]):
                for j in range(dose.shape[1]):
                    for k in range(dose.shape[2]):
                        # Vị trí voxel trong không gian 3D
                        pos = np.array(
                            [
                                dose_grid.origin[0] + k * dose_grid.spacing[0],
                                dose_grid.origin[1] + j * dose_grid.spacing[1],
                                dose_grid.origin[2] + i * dose_grid.spacing[2],
                            ]
                        )

                        # Tính khoảng cách từ nguồn đến voxel
                        direction = pos - source_pos
                        distance = np.linalg.norm(direction)

                        # Tính góc so với trục chùm tia
                        angle = np.arccos(np.dot(direction / distance, beam_direction))

                        # Áp dụng nghịch đảo bình phương và suy giảm theo góc
                        inverse_square = (sad / distance) ** 2
                        angular_falloff = np.cos(angle) ** 3

                        # Áp dụng suy giảm theo vật chất nếu cần
                        if (
                            use_heterogeneity
                            and i < ct_data.shape[0]
                            and j < ct_data.shape[1]
                            and k < ct_data.shape[2]
                        ):
                            # Đơn giản hóa: giả sử ct_data là hệ số suy giảm tỷ lệ
                            material_attenuation = 1.0 - attenuation * ct_data[i, j, k]
                        else:
                            material_attenuation = 1.0

                        # Tính liều tại voxel
                        dose[i, j, k] = (
                            beam.monitor_units
                            * inverse_square
                            * angular_falloff
                            * material_attenuation
                        )

            # Chuẩn hóa liều: giả sử 100 MU cho 1 Gy tại isocenter
            dose = dose / 100.0

            logger.info(
                f"Đã tính toán liều cho chùm tia {beam.name} với thuật toán {self.name}"
            )
            return dose

        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều với {self.name}: {str(e)}")
            return np.zeros_like(dose_grid.data)

    def _get_beam_parameters(self, beam: Beam) -> Dict[str, Any]:
        """
        Trích xuất thông số chùm tia từ đối tượng Beam.

        Parameters
        ----------
        beam : Beam
            Chùm tia xạ trị

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa các thông số chùm tia
        """
        # Tạo từ điển thông số
        params = {
            "gantry_angle": getattr(beam, "gantry_angle", 0.0),
            "collimator_angle": getattr(beam, "collimator_angle", 0.0),
            "couch_angle": getattr(beam, "couch_angle", 0.0),
            "isocenter": getattr(beam, "isocenter", [0.0, 0.0, 0.0]),
            "field_size": getattr(beam, "field_size", [100.0, 100.0]),
            "monitor_units": getattr(beam, "monitor_units", 100.0),
        }

        return params

    def _calculate_beam_direction(self, beam_params: Dict[str, Any]) -> np.ndarray:
        """
        Tính toán vector hướng chuẩn hóa của chùm tia.

        Parameters
        ----------
        beam_params : Dict[str, Any]
            Thông số chùm tia

        Returns
        -------
        np.ndarray
            Vector hướng chuẩn hóa
        """
        # Đơn giản hóa: tính hướng chùm tia chỉ dựa trên góc gantry
        gantry_rad = np.radians(beam_params["gantry_angle"])

        # Hướng chùm tia trong hệ tọa độ IEC
        direction = np.array([np.sin(gantry_rad), 0.0, -np.cos(gantry_rad)])

        return direction / np.linalg.norm(direction)

    def _calculate_source_position(
        self, beam_params: Dict[str, Any], beam_direction: np.ndarray, sad: float
    ) -> np.ndarray:
        """
        Tính toán vị trí nguồn chùm tia.

        Parameters
        ----------
        beam_params : Dict[str, Any]
            Thông số chùm tia
        beam_direction : np.ndarray
            Hướng chùm tia
        sad : float
            Khoảng cách từ nguồn đến isocenter

        Returns
        -------
        np.ndarray
            Vị trí nguồn
        """
        isocenter = np.array(beam_params["isocenter"])

        # Tính vị trí nguồn là SAD từ isocenter ngược hướng chùm tia
        source_position = isocenter - beam_direction * sad

        return source_position


class PencilBeamAlgorithm(DoseAlgorithmBase):
    """
    Thuật toán tính liều dựa trên mô hình pencil beam.
    """

    def __init__(self):
        """Khởi tạo thuật toán pencil beam."""
        super().__init__()
        self.name = "PencilBeam"
        self.description = "Pencil beam convolution algorithm"

        # Tham số mặc định
        self.parameters = {
            "kernel_width": 5.0,  # mm
            "kernel_height": 5.0,  # mm
            "use_heterogeneity_correction": True,
            "use_electron_transport": False,
        }

    def calculate_dose(
        self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid
    ) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán pencil beam.

        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán

        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        # [Triển khai thuật toán pencil beam thực tế ở đây]
        # Đây là phiên bản giả lập đơn giản

        logger.info(
            f"Thuật toán Pencil Beam được gọi cho chùm tia {beam.name} - hiện chưa triển khai đầy đủ"
        )

        # Trả về kết quả giả
        return np.ones_like(dose_grid.data) * 0.5


class CollapsedConeAlgorithm(DoseAlgorithmBase):
    """
    Thuật toán tính liều dựa trên mô hình collapsed cone.
    """

    def __init__(self):
        """Khởi tạo thuật toán collapsed cone."""
        super().__init__()
        self.name = "CollapsedCone"
        self.description = "Collapsed cone convolution/superposition algorithm"

        # Tham số mặc định
        self.parameters = {
            "num_cones": 16,
            "use_heterogeneity_correction": True,
            "max_depth": 300.0,  # mm
        }

    def calculate_dose(
        self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid
    ) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán collapsed cone.

        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán

        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        # [Triển khai thuật toán collapsed cone thực tế ở đây]
        # Đây là phiên bản giả lập đơn giản

        logger.info(
            f"Thuật toán Collapsed Cone được gọi cho chùm tia {beam.name} - hiện chưa triển khai đầy đủ"
        )

        # Trả về kết quả giả
        return np.ones_like(dose_grid.data) * 0.7


class MonteCarloAlgorithm(DoseAlgorithmBase):
    """
    Thuật toán tính liều dựa trên mô phỏng Monte Carlo.
    """

    def __init__(self):
        """Khởi tạo thuật toán Monte Carlo."""
        super().__init__()
        self.name = "MonteCarlo"
        self.description = "Monte Carlo simulation algorithm"

        # Tham số mặc định
        self.parameters = {
            "num_histories": 1000000,
            "statistical_uncertainty": 0.02,  # 2%
            "use_variance_reduction": True,
        }

    def calculate_dose(
        self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid
    ) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán Monte Carlo.

        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán

        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        # [Triển khai thuật toán Monte Carlo thực tế ở đây]
        # Đây là phiên bản giả lập đơn giản

        logger.info(
            f"Thuật toán Monte Carlo được gọi cho chùm tia {beam.name} - hiện chưa triển khai đầy đủ"
        )

        # Trả về kết quả giả
        return np.ones_like(dose_grid.data) * 0.9


class DoseCalculator:
    """
    Dose calculation engine for QuangTPS.

    This class provides methods for calculating radiation dose from treatment beams.
    It implements a simplified pencil beam algorithm for dose calculation.
    """

    def __init__(self):
        """Initialize the dose calculator."""
        self.image = None
        self.structure_set = None
        self.beam_set = None
        self.calculation_grid = None
        self.dose_grid = None
        self.is_calculated = False

        # Default parameters
        self.calculation_grid_resolution = (3.0, 3.0, 3.0)  # mm
        self.photon_beam_params = {
            "6MV": {
                "pdd": {  # Percentage depth dose
                    "d_max": 15,  # mm
                    "pdd_values": None,  # Will be initialized with realistic values
                },
                "profile": {  # Beam profile
                    "penumbra": 5.0,  # mm
                    "in_field_factor": 1.0,
                    "out_field_factor": 0.03,
                },
                "scatter": {"factor": 0.02},
                "output_factor": 1.0,
            },
            "10MV": {
                "pdd": {
                    "d_max": 25,  # mm
                    "pdd_values": None,
                },
                "profile": {
                    "penumbra": 5.5,  # mm
                    "in_field_factor": 1.0,
                    "out_field_factor": 0.02,
                },
                "scatter": {"factor": 0.015},
                "output_factor": 1.05,
            },
            "15MV": {
                "pdd": {
                    "d_max": 30,  # mm
                    "pdd_values": None,
                },
                "profile": {
                    "penumbra": 6.0,  # mm
                    "in_field_factor": 1.0,
                    "out_field_factor": 0.015,
                },
                "scatter": {"factor": 0.01},
                "output_factor": 1.08,
            },
        }

        # Initialize PDD values (approximated)
        self._initialize_pdd_values()

        logger.info("Dose calculator initialized")

    def _initialize_pdd_values(self):
        """Initialize percentage depth dose values for each beam energy."""
        # Create a depth range from 0 to 400 mm
        depths = np.arange(0, 400, 1.0)

        for energy, params in self.photon_beam_params.items():
            d_max = params["pdd"]["d_max"]

            # Model PDD curve (approximation of real PDD curves)
            # Build-up region, peak at d_max, then exponential decay
            pdd_values = np.zeros_like(depths)

            # Build-up region (0 to d_max)
            buildup_indices = depths <= d_max
            pdd_values[buildup_indices] = (
                100.0 * (depths[buildup_indices] / d_max) ** 0.5
            )

            # Exponential decay after d_max
            decay_indices = depths > d_max

            # Different decay rates for different energies
            if energy == "6MV":
                decay_rate = 0.004
            elif energy == "10MV":
                decay_rate = 0.003
            else:  # 15MV
                decay_rate = 0.0025

            pdd_values[decay_indices] = 100.0 * np.exp(
                -decay_rate * (depths[decay_indices] - d_max)
            )

            # Assign PDD values to parameters
            self.photon_beam_params[energy]["pdd"]["pdd_values"] = pdd_values

    def set_image(self, image: Image):
        """Set the reference image for dose calculation."""
        self.image = image
        self.is_calculated = False
        logger.info(
            f"Reference image set for dose calculation - shape: {image.data.shape}"
        )

    def set_structure_set(self, structure_set: StructureSet):
        """Set the structure set for dose calculation."""
        self.structure_set = structure_set
        self.is_calculated = False
        logger.info(
            f"Structure set with {len(structure_set.structures)} structures set for dose calculation"
        )

    def set_beam_set(self, beam_set: BeamSet):
        """Set the beam set for dose calculation."""
        self.beam_set = beam_set
        self.is_calculated = False
        logger.info(
        return True

        )

    def set_calculation_grid_resolution(self, resolution: Tuple[float, float, float]):
        """Set the calculation grid resolution in mm."""
        self.calculation_grid_resolution = resolution
        self.is_calculated = False
        logger.info(f"Calculation grid resolution set to {resolution} mm")

    def initialize_calculation_grid(self):
        """Initialize the calculation grid based on the reference image and resolution."""
        if self.image is None:
            logger.error("Cannot initialize calculation grid: No reference image")
            return False

        # Get image dimensions and spacing
        img_shape = self.image.data.shape
        img_spacing = (
            self.image.spacing if hasattr(self.image, "spacing") else (1.0, 1.0, 1.0)
        )

        # Calculate the shape of the dose grid
        grid_shape = (
            int(img_shape[0] * img_spacing[0] / self.calculation_grid_resolution[0]),
            int(img_shape[1] * img_spacing[1] / self.calculation_grid_resolution[1]),
            int(img_shape[2] * img_spacing[2] / self.calculation_grid_resolution[2]),
        )

        # Initialize the calculation grid
        self.calculation_grid = np.zeros(grid_shape, dtype=np.float32)

        # Initialize the dose grid (same as calculation grid)
        self.dose_grid = np.zeros_like(self.calculation_grid)

        logger.info(f"Calculation grid initialized with shape: {grid_shape}")
        return True
        return True

    def calculate_dose(self) -> Optional[np.ndarray]:
        """
        Calculate dose distribution from the beam set.

        Returns:
            Optional[np.ndarray]: The calculated dose grid, or None if calculation failed.
        """
        if self.image is None or self.beam_set is None:
            logger.error("Cannot calculate dose: Missing image or beam set")
            return None

        # Initialize calculation grid if not already done
        if self.calculation_grid is None:
            if not self.initialize_calculation_grid():
                return None

        # Clear dose grid
        self.dose_grid.fill(0.0)

        # Calculate dose for each beam
        for i, beam in enumerate(self.beam_set.beams):
            logger.info(
                f"Calculating dose for beam {i + 1}/{len(self.beam_set.beams)}: {beam.name}"
            )

            # Calculate beam dose
            beam_dose = self._calculate_beam_dose(beam)

            # Add to total dose
            if beam_dose is not None:
                # Apply beam weight
                beam_dose *= beam.weight

                # Add to total dose
                self.dose_grid += beam_dose

        # Normalize dose to prescription
        if hasattr(self.beam_set, "prescription") and self.beam_set.prescription > 0:
            # Find the normalization point (max dose in PTV or isocenter)
            if (
                self.structure_set is not None
                and hasattr(self.beam_set, "target_structure_id")
                and self.beam_set.target_structure_id
            ):
                # Get the target structure
                target_structure = next(
                    (
                        s
                        for s in self.structure_set.structures
                        if s.id == self.beam_set.target_structure_id
                    ),
                    None,
                )

                if target_structure is not None and hasattr(target_structure, "mask"):
                    # Resize the target mask to match the calculation grid
                    target_mask = self._resize_structure_mask(target_structure.mask)

                    # Find the max dose in the target
                    if np.any(target_mask):
                        max_dose = np.max(self.dose_grid[target_mask])
                    else:
                        max_dose = np.max(self.dose_grid)
                else:
                    max_dose = np.max(self.dose_grid)
            else:
                max_dose = np.max(self.dose_grid)

            # Avoid division by zero
            if max_dose > 0:
                # Normalize to prescription dose
                self.dose_grid *= self.beam_set.prescription / max_dose

        self.is_calculated = True
        logger.info("Dose calculation completed")

        return self.dose_grid

    def _calculate_beam_dose(self, beam: Beam) -> Optional[np.ndarray]:
        """
        Calculate dose for a single beam.

        Args:
            beam: The beam to calculate dose for

        Returns:
            Optional[np.ndarray]: The calculated beam dose, or None if calculation failed.
        """
        # Get beam parameters
        energy = beam.energy if hasattr(beam, "energy") else "6MV"
        gantry_angle = beam.gantry_angle if hasattr(beam, "gantry_angle") else 0.0
        couch_angle = beam.couch_angle if hasattr(beam, "couch_angle") else 0.0
        collimator_angle = (
            beam.collimator_angle if hasattr(beam, "collimator_angle") else 0.0
        )

        # Get field size
        if hasattr(beam, "field_size"):
            field_width, field_height = beam.field_size
        else:
            field_width, field_height = 100.0, 100.0  # Default 10x10 cm

        # Get isocenter position
        if hasattr(beam, "isocenter"):
            isocenter = beam.isocenter
        else:
            # Default to center of image
            img_shape = self.image.data.shape
            img_spacing = (
                self.image.spacing
                if hasattr(self.image, "spacing")
                else (1.0, 1.0, 1.0)
            )
            isocenter = (
                img_shape[0] * img_spacing[0] / 2,
                img_shape[1] * img_spacing[1] / 2,
                img_shape[2] * img_spacing[2] / 2,
            )

        # Get beam parameters for the energy
        beam_params = self.photon_beam_params.get(energy)
        if beam_params is None:
            logger.error(f"Unknown beam energy: {energy}")
            return None

        # Create beam dose grid (same size as calculation grid)
        beam_dose = np.zeros_like(self.calculation_grid)

        # Create coordinate grids
        grid_shape = self.calculation_grid.shape
        z, y, x = np.meshgrid(
            np.arange(grid_shape[0]),
            np.arange(grid_shape[1]),
            np.arange(grid_shape[2]),
            indexing="ij",
        )

        # Convert to physical coordinates
        x = x * self.calculation_grid_resolution[0]
        y = y * self.calculation_grid_resolution[1]
        z = z * self.calculation_grid_resolution[2]

        # Shift coordinates to isocenter
        x = x - isocenter[0]
        y = y - isocenter[1]
        z = z - isocenter[2]

        # Rotate coordinates according to beam angles
        # (Simplified rotation - just gantry angle for now)
        gantry_rad = np.radians(gantry_angle)

        # Rotate around y-axis (gantry rotation)
        x_rot = x * np.cos(gantry_rad) + z * np.sin(gantry_rad)
        z_rot = -x * np.sin(gantry_rad) + z * np.cos(gantry_rad)

        # Use rotated coordinates
        x = x_rot
        z = z_rot

        # Calculate depth for each point (distance along beam direction)
        depth = z_rot  # For zero gantry, depth is z

        # Calculate lateral and vertical offsets (perpendicular to beam direction)
        lateral_offset = x  # For zero gantry, lateral is x
        vertical_offset = y  # Vertical is always y

        # Calculate in-field/out-field status based on field size
        half_width = field_width / 2
        half_height = field_height / 2

        in_field = (
            (lateral_offset >= -half_width)
            & (lateral_offset <= half_width)
            & (vertical_offset >= -half_height)
            & (vertical_offset <= half_height)
        )

        # Calculate penumbra region
        penumbra = beam_params["profile"]["penumbra"]

        in_penumbra_x = (
            (lateral_offset >= -half_width - penumbra) & (lateral_offset <= -half_width)
        ) | ((lateral_offset >= half_width) & (lateral_offset <= half_width + penumbra))

        in_penumbra_y = (
            (vertical_offset >= -half_height - penumbra)
            & (vertical_offset <= -half_height)
        ) | (
            (vertical_offset >= half_height)
            & (vertical_offset <= half_height + penumbra)
        )

        in_penumbra = (
            (
                in_penumbra_x
                & (vertical_offset >= -half_height)
                & (vertical_offset <= half_height)
            )
            | (
                in_penumbra_y
                & (lateral_offset >= -half_width)
                & (lateral_offset <= half_width)
            )
            | (in_penumbra_x & in_penumbra_y)
        )

        # Apply PDD (Percentage Depth Dose)
        pdd_values = beam_params["pdd"]["pdd_values"]

        # Convert depth to indices (clip to valid range)
        depth_indices = np.clip(depth.astype(int), 0, len(pdd_values) - 1)

        # Apply PDD to beam_dose
        beam_dose = (
            pdd_values[depth_indices] / 100.0
        )  # Convert from percentage to fraction

        # Apply beam profile
        in_field_factor = beam_params["profile"]["in_field_factor"]
        out_field_factor = beam_params["profile"]["out_field_factor"]

        # Regions outside the field + penumbra
        beam_dose[~(in_field | in_penumbra)] *= out_field_factor
                # In-field regions
        beam_dose[in_field] *= in_field_factor

        # Penumbra regions (linear falloff)
        if np.any(in_penumbra):
            # X penumbra
            x_dist = np.zeros_like(lateral_offset)
            x_dist[lateral_offset < -half_width] = (
                -lateral_offset[lateral_offset < -half_width] - half_width
            )
            x_dist[lateral_offset > half_width] = (
                lateral_offset[lateral_offset > half_width] - half_width
            )

            # Y penumbra
            y_dist = np.zeros_like(vertical_offset)
            y_dist[vertical_offset < -half_height] = (
                -vertical_offset[vertical_offset < -half_height] - half_height
            )
            y_dist[vertical_offset > half_height] = (
                vertical_offset[vertical_offset > half_height] - half_height
            )

            # Penumbra factor (linear falloff from 1.0 to 0.03)
            penumbra_factor = np.ones_like(beam_dose)

            # Points in x penumbra only
            x_only = (
                in_penumbra_x
                & ~in_penumbra_y
                & (vertical_offset >= -half_height)
                & (vertical_offset <= half_height)
            )
            if np.any(x_only):
                penumbra_factor[x_only] = 1.0 - (1.0 - out_field_factor) * (
                    x_dist[x_only] / penumbra
                )

            # Points in y penumbra only
            y_only = (
                ~in_penumbra_x
                & in_penumbra_y
                & (lateral_offset >= -half_width)
                & (lateral_offset <= half_width)
            )
            if np.any(y_only):
                penumbra_factor[y_only] = 1.0 - (1.0 - out_field_factor) * (
                    y_dist[y_only] / penumbra
                )

            # Points in both x and y penumbra - use the larger distance
            both = in_penumbra_x & in_penumbra_y
            if np.any(both):
                max_dist = np.maximum(x_dist[both], y_dist[both])
                penumbra_factor[both] = 1.0 - (1.0 - out_field_factor) * (
                    max_dist / penumbra
                )

            # Apply penumbra factor
            beam_dose[in_penumbra] *= penumbra_factor[in_penumbra]

        # Apply output factor
        beam_dose *= beam_params["output_factor"]

        # Apply MLC modulation if available
        if hasattr(beam, "mlc") and beam.mlc is not None:
            mlc_modulation = self._calculate_mlc_modulation(beam)
            if mlc_modulation is not None:
                beam_dose *= mlc_modulation

        # Return the beam dose
        return beam_dose

    def _calculate_mlc_modulation(self, beam: Beam) -> Optional[np.ndarray]:
        """
        Calculate MLC (Multi-Leaf Collimator) modulation for a beam.

        Args:
            beam: The beam with MLC data

        Returns:
            Optional[np.ndarray]: The MLC modulation factors, or None if not available
        """
        # Simplified implementation - just a placeholder
        # In a real implementation, this would project the MLC leaves onto the calculation grid
        return None

    def _resize_structure_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize a structure mask to match the calculation grid.

        Args:
            mask: The structure mask in the image coordinate system

        Returns:
            np.ndarray: The resized mask in the calculation grid coordinate system
        """
        # Get image spacing and calculation grid spacing
        img_spacing = (
            self.image.spacing if hasattr(self.image, "spacing") else (1.0, 1.0, 1.0)
        )

        # Create resized mask
        mask_shape = mask.shape
        grid_shape = self.calculation_grid.shape

        # Initialize empty mask
        resized_mask = np.zeros(grid_shape, dtype=bool)

        # Simple nearest-neighbor sampling
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    # Calculate corresponding coordinates in original mask
                    i_orig = int(
                        i * self.calculation_grid_resolution[0] / img_spacing[0]
                    )
                    j_orig = int(
                        j * self.calculation_grid_resolution[1] / img_spacing[1]
                    )
                    k_orig = int(
                        k * self.calculation_grid_resolution[2] / img_spacing[2]
                    )

                    # Check bounds
                    if (
                        0 <= i_orig < mask_shape[0]
                        and 0 <= j_orig < mask_shape[1]
                        and 0 <= k_orig < mask_shape[2]
                    ):
                        resized_mask[i, j, k] = mask[i_orig, j_orig, k_orig]

        return resized_mask

    def get_dose_at_point(self, point: Tuple[float, float, float]) -> float:
        """
        Get the dose value at a specific point.

        Args:
            point: The point coordinates in mm

        Returns:
            float: The dose value at the point, or 0 if outside the dose grid
        """
        if not self.is_calculated or self.dose_grid is None:
            logger.error("Dose has not been calculated")
            return 0.0

        # Convert point to dose grid indices
        i = int(point[0] / self.calculation_grid_resolution[0])
        j = int(point[1] / self.calculation_grid_resolution[1])
        k = int(point[2] / self.calculation_grid_resolution[2])

        # Check bounds
        if (
            0 <= i < self.dose_grid.shape[0]
            and 0 <= j < self.dose_grid.shape[1]
            and 0 <= k < self.dose_grid.shape[2]
        ):
            return self.dose_grid[i, j, k]
        else:
            return 0.0

    def get_structure_dose_stats(self, structure: Structure) -> Dict[str, float]:
        """
        Calculate dose statistics for a structure.

        Args:
            structure: The structure to calculate statistics for

        Returns:
            Dict[str, float]: Dictionary containing dose statistics
        """
        if not self.is_calculated or self.dose_grid is None:
            logger.error("Dose has not been calculated")
            return {
                "min_dose": 0.0,
                "max_dose": 0.0,
                "mean_dose": 0.0,
                "median_dose": 0.0,
                "D95": 0.0,
                "D98": 0.0,
                "D99": 0.0,
                "D50": 0.0,
                "D2": 0.0,
                "V95": 0.0,
            }

        if not hasattr(structure, "mask") or structure.mask is None:
            logger.error(f"Structure {structure.name} has no mask")
            return {
                "min_dose": 0.0,
                "max_dose": 0.0,
                "mean_dose": 0.0,
                "median_dose": 0.0,
                "D95": 0.0,
                "D98": 0.0,
                "D99": 0.0,
                "D50": 0.0,
                "D2": 0.0,
                "V95": 0.0,
            }

        # Resize structure mask to match dose grid
        mask = self._resize_structure_mask(structure.mask)

        # Check if mask has any voxels
        if not np.any(mask):
            logger.warning(f"Structure {structure.name} has no voxels in the dose grid")
            return {
                "min_dose": 0.0,
                "max_dose": 0.0,
                "mean_dose": 0.0,
                "median_dose": 0.0,
                "D95": 0.0,
                "D98": 0.0,
                "D99": 0.0,
                "D50": 0.0,
                "D2": 0.0,
                "V95": 0.0,
            }

        # Get dose values in the structure
        structure_dose = self.dose_grid[mask]

        # Calculate basic statistics
        min_dose = np.min(structure_dose)
        max_dose = np.max(structure_dose)
        mean_dose = np.mean(structure_dose)
        median_dose = np.median(structure_dose)

        # Sort dose values for percentile calculations
        sorted_dose = np.sort(structure_dose)

        # Calculate dose-volume metrics
        D95 = sorted_dose[int(len(sorted_dose) * 0.05)]  # Dose to 95% of volume
        D98 = sorted_dose[int(len(sorted_dose) * 0.02)]  # Dose to 98% of volume
        D99 = sorted_dose[int(len(sorted_dose) * 0.01)]  # Dose to 99% of volume
        D50 = sorted_dose[int(len(sorted_dose) * 0.50)]  # Dose to 50% of volume
        D2 = sorted_dose[int(len(sorted_dose) * 0.98)]  # Dose to 2% of volume

        # Calculate volume receiving 95% of prescription dose
        if hasattr(self.beam_set, "prescription") and self.beam_set.prescription > 0:
            prescription = self.beam_set.prescription
            V95 = (
                np.sum(structure_dose >= 0.95 * prescription)
                / len(structure_dose)
                * 100.0
            )
        else:
            V95 = 0.0

        return {
            "min_dose": min_dose,
            "max_dose": max_dose,
            "mean_dose": mean_dose,
            "median_dose": median_dose,
            "D95": D95,
            "D98": D98,
            "D99": D99,
            "D50": D50,
            "D2": D2,
            "V95": V95,
        }


# Example usage
def test_dose_calculator():
    """Test the dose calculator with sample data."""
    from quangtps.imaging.image import Image
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure
    from quangtps.beams.beam import Beam, BeamSet

    # Create sample image
    image_data = np.ones((100, 100, 50), dtype=np.float32)
    image = Image()
    image.data = image_data
    image.spacing = (2.0, 2.0, 3.0)  # mm

    # Create sample structure set
    structure_set = StructureSet()

    # Create PTV
    ptv = Structure()
    ptv.id = "struct_1"
    ptv.name = "PTV"
    ptv.type = "PTV"
    ptv.mask = np.zeros_like(image_data, dtype=bool)
    ptv.mask[40:60, 40:60, 20:30] = True

    # Create OAR
    oar = Structure()
    oar.id = "struct_2"
    oar.name = "OAR"
    oar.type = "OAR"
    oar.mask = np.zeros_like(image_data, dtype=bool)
    oar.mask[55:65, 40:50, 20:30] = True

    # Add structures to structure set
    structure_set.add_structure(ptv)
    structure_set.add_structure(oar)

    # Create sample beam set
    beam_set = BeamSet()
    beam_set.id = "beamset_1"
    beam_set.name = "Sample Plan"
    beam_set.prescription = 70.0  # Gy
    beam_set.target_structure_id = ptv.id

    # Create beams
    beam1 = Beam()
    beam1.id = "beam_1"
    beam1.name = "AP"
    beam1.energy = "6MV"
    beam1.gantry_angle = 0.0
    beam1.couch_angle = 0.0
    beam1.collimator_angle = 0.0
    beam1.field_size = (40.0, 40.0)  # mm
    beam1.isocenter = (100.0, 100.0, 75.0)  # mm
    beam1.weight = 1.0

    beam2 = Beam()
    beam2.id = "beam_2"
    beam2.name = "LPO"
    beam2.energy = "6MV"
    beam2.gantry_angle = 120.0
    beam2.couch_angle = 0.0
    beam2.collimator_angle = 0.0
    beam2.field_size = (40.0, 40.0)  # mm
    beam2.isocenter = (100.0, 100.0, 75.0)  # mm
    beam2.weight = 1.0

    beam3 = Beam()
    beam3.id = "beam_3"
    beam3.name = "RPO"
    beam3.energy = "6MV"
    beam3.gantry_angle = 240.0
    beam3.couch_angle = 0.0
    beam3.collimator_angle = 0.0
    beam3.field_size = (40.0, 40.0)  # mm
    beam3.isocenter = (100.0, 100.0, 75.0)  # mm
    beam3.weight = 1.0

    # Add beams to beam set
    beam_set.add_beam(beam1)
    beam_set.add_beam(beam2)
    beam_set.add_beam(beam3)

    # Create dose calculator
    calculator = DoseCalculator()
    calculator.set_image(image)
    calculator.set_structure_set(structure_set)
    calculator.set_beam_set(beam_set)

    # Set calculation grid resolution (5mm)
    calculator.set_calculation_grid_resolution((5.0, 5.0, 5.0))

    # Calculate dose
    dose_grid = calculator.calculate_dose()

    if dose_grid is not None:
        print(f"Dose calculation successful. Grid shape: {dose_grid.shape}")

        # Calculate dose statistics for PTV
        ptv_stats = calculator.get_structure_dose_stats(ptv)

        print("PTV Dose Statistics:")
        for stat, value in ptv_stats.items():
            print(f"  {stat}: {value:.2f}")

        # Calculate dose statistics for OAR
        oar_stats = calculator.get_structure_dose_stats(oar)

        print("OAR Dose Statistics:")
        for stat, value in oar_stats.items():
            print(f"  {stat}: {value:.2f}")
    else:
        print("Dose calculation failed")


if __name__ == "__main__":
    test_dose_calculator()

# Export
__all__ = [
    "DoseAlgorithmBase",
    "SimpleRayTracingAlgorithm",
    "PencilBeamAlgorithm",
    "CollapsedConeAlgorithm",
    "MonteCarloAlgorithm",
    "DoseCalculator",
]
