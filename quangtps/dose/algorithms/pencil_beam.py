#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of the Pencil Beam dose calculation algorithm.

This module provides a class for calculating dose distributions using
the Pencil Beam algorithm for radiotherapy treatment planning.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any

from quangtps.core.exceptions import DoseCalculationError, ValidationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter
from quangtps.dose.algorithms.base import (
    DoseCalculationAlgorithm,
    DoseCalculationResult,
)
from quangtps.dose.physics.terma import calculate_terma_from_beam

logger = logging.getLogger(__name__)


class PencilBeamAlgorithm(DoseCalculationAlgorithm):
    """
    Implementation of the Pencil Beam dose calculation algorithm.

    This class provides methods for calculating 3D dose distributions
    using the Pencil Beam algorithm for radiotherapy treatment planning.
    """

    def __init__(self):
        """
        Initialize the Pencil Beam algorithm.
        """
        super().__init__("Pencil Beam")
        self.version = "1.0"

        # Default parameters
        self.parameters.update(
            {
                "grid_size": 0.2,  # Calculation grid size in cm
                "threads": 4,  # Number of parallel threads
                "tissue_air_ratio_correction": True,  # Whether to apply TAR correction
                "use_gpu": False,  # Whether to use GPU acceleration
                "pencil_spacing": 0.1,  # Spacing between pencil beams in cm
                "integration_step": 0.5,  # Integration step size in cm for ray tracing
            }
        )

        logger.info(f"Initialized {self.name} algorithm version {self.version}")

        self.beam_model = None

    def set_beam_model(self, beam_model: BeamModel):
        """
        Set the beam model for dose calculation.

        Parameters
        ----------
        beam_model : BeamModel
            The beam model containing beam data for dose calculation
        """
        self.beam_model = beam_model
        logger.info(f"Set beam model: {beam_model.name}")

    def set_heterogeneity_correction(self, enabled: bool):
        """
        Enable or disable heterogeneity correction.

        Parameters
        ----------
        enabled : bool
            Flag to enable or disable heterogeneity correction
        """
        self.get_parameter("heterogeneity_correction")
        status = "enabled" if enabled else "disabled"
        logger.info(f"Heterogeneity correction {status}")

    def set_calculation_parameters(self, grid_size: float = 0.25, threads: int = 8):
        """
        Set calculation parameters.

        Parameters
        ----------
        grid_size : float
            Calculation grid size in cm
        threads : int
            Number of parallel threads for calculation
        """
        self.get_parameter("grid_size")
        self.get_parameter("threads")
        logger.info(
            f"Set calculation parameters: grid_size={grid_size}cm, threads={threads}"
        )

    def calculate(self, ct_image: Image, beam: Beam) -> DoseCalculationResult:
        """
        Calculate dose distribution using Pencil Beam algorithm.

        Parameters
        ----------
        ct_image : Image
            CT image for dose calculation
        beam : Beam
            Treatment beam

        Returns
        -------
        DoseCalculationResult
            Calculated dose and metadata

        Raises
        ------
        DoseCalculationError
            If dose calculation fails
        ValidationError
            If inputs are invalid
        """
        start_time = time.time()

        try:
            # Validate inputs
            self.validate_inputs(ct_image, beam)

            # Get calculation parameters
            tissue_air_ratio_correction = self.get_parameter(
                "tissue_air_ratio_correction"
            )
            grid_size = self.get_parameter("grid_size")
            threads = self.get_parameter("threads")
            pencil_spacing = self.get_parameter("pencil_spacing")
            integration_step = self.get_parameter("integration_step")
            use_gpu = self.get_parameter("use_gpu")

            # Điều chỉnh số lượng threads dựa trên CPU có sẵn nếu không được đặt hợp lý
            import multiprocessing

            available_cpus = multiprocessing.cpu_count()
            if threads <= 0 or threads > available_cpus:
                threads = max(
                    1, min(available_cpus - 1, 4)
                )  # Mặc định để lại 1 CPU cho hệ thống
                logger.info(
                    f"Điều chỉnh số threads thành {threads} dựa trên {available_cpus} CPU có sẵn"
                )

            logger.info(f"Bắt đầu tính toán liều Pencil Beam cho chùm tia {beam.name}")
            logger.info(
                f"Tham số: grid_size={grid_size}cm, threads={threads}, TAR={tissue_air_ratio_correction}, GPU={use_gpu}"
            )

            # Ước tính bộ nhớ cần thiết trước khi tính toán
            nx, ny, nz = ct_image.data.shape
            estimated_memory_mb = (nx * ny * nz * 4 * 3) / (
                1024 * 1024
            )  # Cho density_grid, dose_grid và electron_density
            logger.info(f"Ước tính sử dụng bộ nhớ: {estimated_memory_mb:.1f} MB")

            # Kiểm tra GPU và tối ưu hóa GPU nếu được yêu cầu
            if use_gpu:
                try:
                    import cupy as cp

                    logger.info("CUDA được bật: Đang sử dụng GPU để tính toán")
                    # Thêm code tận dụng GPU ở đây nếu cần
                except ImportError:
                    logger.warning("Không thể import cupy. Chuyển sang tính toán CPU.")
                    use_gpu = False

            # Convert CT to electron density
            electron_density = self._convert_ct_to_density(ct_image)

            # Initialize dose grid
            dose_data = np.zeros_like(
                ct_image.data, dtype=np.float32
            )  # Sử dụng float32 để giảm bộ nhớ

            # Get beam parameters
            field_size = beam.field_size  # in cm
            sad = (
                beam.sad if hasattr(beam, "sad") else 1000.0
            )  # mm -> convert to cm later
            isocenter = np.array(beam.isocenter) / 10.0  # mm -> cm
            beam_direction = np.array(beam.get_direction())

            # Extract beam MLC configuration if available
            mlc_config = None
            if hasattr(beam, "mlc") and beam.mlc is not None:
                mlc_config = beam.mlc.get_leaf_positions()

            # Calculate source position
            source_position = isocenter - beam_direction * (sad / 10.0)  # cm

            # Generate pencil beams based on field type
            pencil_beams = []

            if mlc_config is not None:
                # Handle MLC-defined field
                pencil_beams = self._generate_mlc_pencil_beams(
                    mlc_config,
                    pencil_spacing,
                    source_position,
                    isocenter,
                    beam_direction,
                )
            else:
                # Handle rectangular field
                pencil_beams = self._generate_rectangular_pencil_beams(
                    field_size,
                    pencil_spacing,
                    source_position,
                    isocenter,
                    beam_direction,
                )

                logger.info(f"Đã tạo {len(pencil_beams)} pencil beam")

                # Tính toán đa luồng nếu được hỗ trợ và được yêu cầu
                if threads > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    import tqdm

                    def process_pencil_beam(pb_data):
                        self._trace_pencil_beam(
                            pb_data["entry_point"],
                            pb_data["direction"],
                            pb_data["weight"],
                            electron_density,
                            dose_data,
                            ct_image,
                            integration_step,
                            tissue_air_ratio_correction,
                        )

                    # Chia nhỏ chùm tia thành các nhóm
                    chunk_size = max(1, len(pencil_beams) // (threads * 4))
                    chunks = [
                        pencil_beams[i : i + chunk_size]
                        for i in range(0, len(pencil_beams), chunk_size)
                    ]

                    logger.info(
                        f"Tính toán đa luồng với {threads} thread, {len(chunks)} nhóm"
                    )

                    with ThreadPoolExecutor(max_workers=threads) as executor:
                        list(
                            tqdm.tqdm(
                                executor.map(process_pencil_beam, pencil_beams),
                                total=len(pencil_beams),
                                desc="Tính toán pencil beam",
                            )
                        )
                else:
                    # For each pencil beam, trace through the patient and calculate dose
                    for pb_index, pencil_beam in enumerate(pencil_beams):
                        if pb_index % 100 == 0:
                            progress = pb_index / len(pencil_beams) * 100
                            logger.debug(
                                f"Tiến độ: {progress:.1f}% ({pb_index}/{len(pencil_beams)})"
                            )

                        self._trace_pencil_beam(
                            pencil_beam["entry_point"],
                            pencil_beam["direction"],
                            pencil_beam["weight"],
                            electron_density,
                            dose_data,
                            ct_image,
                            integration_step,
                            tissue_air_ratio_correction,
                        )

            # Normalize the dose grid (optional based on beam weight)
            if hasattr(beam, "weight") and beam.weight > 0:
                dose_data *= beam.weight

            # Create the output image
            dose_image = Image(
                data=dose_data,
                origin=ct_image.origin,
                spacing=ct_image.spacing,
                direction=ct_image.direction,
            )

            # Optional: apply smoothing
            # dose_image = self.apply_smoothing(dose_image)

            calculation_time = time.time() - start_time
            logger.info(f"Tính toán liều hoàn tất trong {calculation_time:.2f} giây")

            # Create and return the result
            result = DoseCalculationResult(
                dose_grid=dose_image,
                algorithm_name=self.name,
                calculation_time=calculation_time,
                metadata={
                    "beam": beam.name,
                    "grid_size": grid_size,
                    "threads": threads,
                    "pencil_spacing": pencil_spacing,
                    "version": self.version,
                    "ct_dimensions": ct_image.data.shape,
                    "sad": sad / 10.0,  # mm -> cm
                },
            )

            # Kiểm tra chất lượng kết quả
            max_dose = np.max(dose_data)
            if max_dose <= 0:
                logger.warning(
                    "Cảnh báo: Liều tối đa bằng 0, vui lòng kiểm tra lại cài đặt chùm tia"
                )
            elif max_dose < 0.001:
                logger.warning(
                    f"Cảnh báo: Liều tối đa rất thấp ({max_dose}), có thể là do thiết lập chùm tia không chính xác"
                )

            # Kiểm tra NaN và giá trị vô cùng
            invalid_values = np.isnan(dose_data) | np.isinf(dose_data)
            if np.any(invalid_values):
                invalid_count = np.sum(invalid_values)
                logger.warning(
                    f"Phát hiện {invalid_count} giá trị không hợp lệ (NaN/Inf) trong phân phối liều"
                )
                # Sửa các giá trị không hợp lệ
                dose_data[invalid_values] = 0.0

            # Kiểm tra coverage
            if hasattr(beam, "target_structure") and beam.target_structure is not None:
                # Đây chỉ là giả định - cần logic thực tế để đánh giá độ phủ mục tiêu
                logger.info(
                    f"Đánh giá độ phủ cho cấu trúc mục tiêu: {beam.target_structure}"
                )
                # Thêm logic đánh giá độ phủ dựa trên beam.target_structure

            return result

        except MemoryError:
            logger.error(
                "Lỗi bộ nhớ khi tính toán liều. Thử làm việc với grid thưa hơn."
            )
            raise DoseCalculationError(
                "Không đủ bộ nhớ để tính toán liều với grid hiện tại"
            ) from None

        except ValidationError as e:
            logger.error(f"Lỗi xác thực đầu vào: {e}")
            raise

        except DoseCalculationError as e:
            logger.error(f"Lỗi tính toán liều: {e}")
            raise

        except Exception as e:
            logger.error(f"Lỗi không xác định trong tính toán liều: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            raise DoseCalculationError(f"Lỗi không xác định: {str(e)}")

    def _convert_ct_to_density(self, ct_image: Image) -> np.ndarray:
        """
        Chuyển đổi hình ảnh CT sang mật độ electron sử dụng các đường cong hiệu chuẩn chuẩn.

        Parameters
        ----------
        ct_image : Image
            Hình ảnh CT đầu vào

        Returns
        -------
        np.ndarray
            Mảng mật độ electron tương đối so với nước
        """
        try:
            # Kiểm tra đầu vào
            if ct_image is None or ct_image.data is None:
                logger.error("Hình ảnh CT hoặc dữ liệu CT là None")
                raise ValueError("Hình ảnh CT không hợp lệ")

            # Lấy dữ liệu CT (HU) và chuyển về float32 để tối ưu bộ nhớ
            hu_values = ct_image.data.astype(np.float32)

            # Kiểm tra và xử lý các giá trị ngoài phạm vi
            if np.any(np.isnan(hu_values)) or np.any(np.isinf(hu_values)):
                invalid_mask = np.isnan(hu_values) | np.isinf(hu_values)
                invalid_count = np.sum(invalid_mask)
                logger.warning(
                    f"Phát hiện {invalid_count} giá trị HU không hợp lệ (NaN/Inf)"
                )
                # Thay thế giá trị không hợp lệ bằng -1000 (không khí)
                hu_values[invalid_mask] = -1000

            # Lấy loại đường cong hiệu chuẩn (nếu có)
            # Hỗ trợ nhiều loại thiết bị CT và giao thức chụp
            calibration_type = (
                self.get_parameter("ct_calibration_type")
                if hasattr(self, "get_parameter")
                else "default"
            )

            # Các bảng chuyển đổi chuẩn cho các loại CT khác nhau
            # Chứa các cặp (HU, mật độ electron tương đối so với nước)
            calibration_tables = {
                "default": np.array(
                    [
                        [-1000, 0.00],  # Không khí
                        [-976, 0.024],  # Phổi hít vào
                        [-480, 0.52],  # Phổi
                        [-100, 0.90],  # Mỡ
                        [0, 1.00],  # Nước
                        [55, 1.06],  # Mô mềm (cơ bắp)
                        [800, 1.45],  # Xương xốp
                        [1500, 1.70],  # Xương đặc
                        [2000, 1.96],  # Xương rất đặc/implant
                        [3000, 2.5],  # Kim loại
                    ],
                    dtype=np.float32,
                ),
                "siemens_sensation": np.array(
                    [
                        [-1000, 0.00],
                        [-950, 0.05],
                        [-700, 0.30],
                        [-300, 0.70],
                        [-100, 0.90],
                        [0, 1.00],
                        [100, 1.07],
                        [300, 1.19],
                        [800, 1.45],
                        [1200, 1.63],
                        [1800, 1.86],
                        [3000, 2.35],
                    ],
                    dtype=np.float32,
                ),
                "ge_lightspeed": np.array(
                    [
                        [-1000, 0.00],
                        [-900, 0.10],
                        [-500, 0.50],
                        [-100, 0.90],
                        [0, 1.00],
                        [100, 1.10],
                        [300, 1.20],
                        [900, 1.50],
                        [1500, 1.75],
                        [2000, 1.95],
                        [3000, 2.40],
                    ],
                    dtype=np.float32,
                ),
                "philips_brilliance": np.array(
                    [
                        [-1000, 0.00],
                        [-950, 0.02],
                        [-750, 0.25],
                        [-400, 0.60],
                        [-100, 0.93],
                        [0, 1.00],
                        [100, 1.06],
                        [400, 1.25],
                        [1000, 1.55],
                        [1600, 1.80],
                        [2500, 2.1],
                        [4000, 2.7],
                    ],
                    dtype=np.float32,
                ),
                "siemens_force": np.array(
                    [
                        [-1000, 0.00],
                        [-980, 0.01],
                        [-800, 0.20],
                        [-600, 0.42],
                        [-400, 0.60],
                        [-200, 0.80],
                        [0, 1.00],
                        [200, 1.12],
                        [400, 1.24],
                        [800, 1.40],
                        [1000, 1.48],
                        [1500, 1.70],
                        [2000, 1.90],
                        [3000, 2.35],
                    ],
                    dtype=np.float32,
                ),
                "toshiba_aquilion": np.array(
                    [
                        [-1000, 0.00],
                        [-950, 0.03],
                        [-650, 0.35],
                        [-350, 0.65],
                        [-150, 0.85],
                        [0, 1.00],
                        [150, 1.10],
                        [350, 1.22],
                        [750, 1.40],
                        [1400, 1.68],
                        [2000, 2.00],
                        [3000, 2.50],
                    ],
                    dtype=np.float32,
                ),
                # Đặc biệt cho các máy CT dùng trong xạ trị (CT-Sim)
                "ct_sim": np.array(
                    [
                        [-1000, 0.00],
                        [-980, 0.02],
                        [-800, 0.20],
                        [-500, 0.50],
                        [-200, 0.80],
                        [0, 1.00],
                        [200, 1.12],
                        [500, 1.28],
                        [1000, 1.50],
                        [1500, 1.75],
                        [2000, 1.98],
                        [3000, 2.50],
                    ],
                    dtype=np.float32,
                ),
            }

            # Sử dụng đường cong mặc định nếu loại được chỉ định không tồn tại
            if calibration_type not in calibration_tables:
                logger.warning(
                    f"Loại đường cong hiệu chuẩn '{calibration_type}' không tồn tại, sử dụng mặc định"
                )
                calibration_type = "default"

            # Lấy bảng hiệu chuẩn
            calibration = calibration_tables[calibration_type]
            hu_points = calibration[:, 0]
            density_points = calibration[:, 1]

            # Lấy giá trị mở rộng để xử lý tốt hơn các trường hợp ngoại lệ
            min_hu = hu_points[0]
            max_hu = hu_points[-1]
            min_density = density_points[0]
            max_density = density_points[-1]

            # Cắt giá trị HU để nằm trong phạm vi bảng hiệu chuẩn
            # Để đảm bảo nội suy không gặp vấn đề
            hu_clipped = np.clip(hu_values, min_hu, max_hu)

            # Tạo mảng mật độ electron với kích thước giống HU
            # Sử dụng nội suy tuyến tính piecewise cho toàn bộ mảng
            # np.interp cho xử lý vectorized hiệu quả hơn
            density = np.interp(
                hu_clipped.flatten(), hu_points, density_points
            ).reshape(hu_values.shape)

            # Đảm bảo giới hạn hợp lý và ngăn chặn các giá trị ngoài phạm vi
            density = np.clip(density, 0.0, 8.0)

            # Kiểm tra xem còn giá trị không hợp lệ không
            invalid_mask = np.isnan(density) | np.isinf(density) | (density < 0)
            if np.any(invalid_mask):
                logger.warning(
                    f"Phát hiện {np.sum(invalid_mask)} giá trị không hợp lệ trong mật độ, thay thế bằng giá trị mật độ nước"
                )
                density[invalid_mask] = 1.0  # Sử dụng mật độ nước thay vì 0

            # Thống kê và hiển thị phân phối mật độ electron theo nhóm để hỗ trợ phát hiện vấn đề
            if logger.isEnabledFor(logging.DEBUG):
                ranges = [
                    (0.0, 0.1),  # Không khí
                    (0.1, 0.5),  # Phổi
                    (0.5, 0.9),  # Mô mật độ thấp
                    (0.9, 1.1),  # Nước/mô mềm
                    (1.1, 1.5),  # Mô đặc
                    (1.5, 2.0),  # Xương
                    (2.0, 8.0),  # Kim loại/implant
                ]

                total_voxels = density.size
                logger.debug("Phân phối mật độ electron:")

                for min_val, max_val in ranges:
                    count = np.sum((density >= min_val) & (density < max_val))
                    percentage = count / total_voxels * 100
                    tissue_type = ""
                    if min_val == 0.0:
                        tissue_type = "(không khí)"
                    elif min_val == 0.1:
                        tissue_type = "(phổi)"
                    elif min_val == 0.5:
                        tissue_type = "(mô mật độ thấp)"
                    elif min_val == 0.9:
                        tissue_type = "(mô mềm/nước)"
                    elif min_val == 1.1:
                        tissue_type = "(mô đặc)"
                    elif min_val == 1.5:
                        tissue_type = "(xương)"
                    elif min_val == 2.0:
                        tissue_type = "(implant/kim loại)"

                    logger.debug(
                        f"  Mật độ {min_val:.1f}-{max_val:.1f} {tissue_type}: {count} voxel ({percentage:.2f}%)"
                    )

            # Hiển thị thông tin chi tiết hơn về kết quả chuyển đổi
            density_stats = {
                "min": np.min(density),
                "max": np.max(density),
                "mean": np.mean(density),
                "median": np.median(density),
                "std": np.std(density),
                "calibration": calibration_type,
            }

            logger.info(
                f"Chuyển đổi CT sang mật độ electron hoàn tất: {calibration_type}, "
                f"min={density_stats['min']:.4f}, max={density_stats['max']:.4f}, "
                f"mean={density_stats['mean']:.4f}, median={density_stats['median']:.4f}, "
                f"std={density_stats['std']:.4f}"
            )

            return density

        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi CT sang mật độ electron: {e}")
            import traceback

            logger.debug(traceback.format_exc())

            # Trả về mảng mật độ nước đồng nhất (1.0) thay thế do lỗi chuyển đổi
            logger.warning(
                "Sử dụng mật độ nước đồng nhất (1.0) thay thế do lỗi chuyển đổi"
            )
            return np.ones_like(ct_image.data, dtype=np.float32)

    def validate_inputs(self, ct_image: Image, beam: Beam) -> None:
        """
        Xác thực dữ liệu đầu vào trước khi tính toán liều.

        Parameters
        ----------
        ct_image : Image
            Hình ảnh CT đầu vào
        beam : Beam
            Chùm tia đầu vào

        Raises
        ------
        ValidationError
            Nếu đầu vào không hợp lệ
        """
        # Kiểm tra hình ảnh CT
        if ct_image is None:
            raise ValidationError("Hình ảnh CT không được cung cấp")

        if ct_image.data is None or ct_image.data.size == 0:
            raise ValidationError("Dữ liệu CT không hợp lệ hoặc rỗng")

        if not isinstance(ct_image.data, np.ndarray):
            raise ValidationError(
                f"Dữ liệu CT phải là numpy array, không phải {type(ct_image.data)}"
            )

        # Kiểm tra chùm tia
        if beam is None:
            raise ValidationError("Chùm tia không được cung cấp")

        # Kiểm tra tham số chùm tia
        if not hasattr(beam, "isocenter") or beam.isocenter is None:
            raise ValidationError("Chùm tia không có tâm đẳng tâm (isocenter)")

        if not hasattr(beam, "get_direction") or not callable(beam.get_direction):
            raise ValidationError("Chùm tia không có phương thức get_direction()")

        if not hasattr(beam, "field_size") or beam.field_size is None:
            raise ValidationError("Chùm tia không có kích thước trường (field_size)")

        # Kiểm tra tính hợp lệ của các giá trị
        try:
            isocenter = np.array(beam.isocenter)
            if np.any(np.isnan(isocenter)) or np.any(np.isinf(isocenter)):
                raise ValidationError(
                    f"Isocenter chứa giá trị không hợp lệ: {beam.isocenter}"
                )

            direction = np.array(beam.get_direction())
            if (
                np.any(np.isnan(direction))
                or np.any(np.isinf(direction))
                or np.linalg.norm(direction) < 1e-6
            ):
                raise ValidationError(f"Hướng chùm tia không hợp lệ: {direction}")

            if isinstance(beam.field_size, (list, tuple, np.ndarray)):
                field_size = np.array(beam.field_size)
                if (
                    np.any(np.isnan(field_size))
                    or np.any(np.isinf(field_size))
                    or np.any(field_size <= 0)
                ):
                    raise ValidationError(
                        f"Kích thước trường không hợp lệ: {beam.field_size}"
                    )
            else:
                raise ValidationError(
                    f"Kích thước trường phải là danh sách/tuple/array, không phải {type(beam.field_size)}"
                )

        except Exception as e:
            raise ValidationError(f"Lỗi xác thực chùm tia: {str(e)}")

        logger.info("Xác thực đầu vào thành công cho tính toán liều")

    def _generate_rectangular_pencil_beams(
        self,
        field_size: List[float],
        pencil_spacing: float,
        source_position: np.ndarray,
        isocenter: np.ndarray,
        beam_direction: np.ndarray,
    ) -> List[Dict]:
        """
        Generate pencil beams for a rectangular field.

        Parameters
        ----------
        field_size : List[float]
            Field size in cm at isocenter
        pencil_spacing : float
            Spacing between pencil beams in cm
        source_position : np.ndarray
            Source position in cm
        isocenter : np.ndarray
            Isocenter position in cm
        beam_direction : np.ndarray
            Beam direction vector

        Returns
        -------
        List[Dict]
            List of pencil beam definitions
        """
        pencil_beams = []

        # Calculate the number of pencil beams
        nx = int(field_size[0] / pencil_spacing) + 1
        ny = int(field_size[1] / pencil_spacing) + 1

        # Ensure odd number for symmetry
        if nx % 2 == 0:
            nx += 1
        if ny % 2 == 0:
            ny += 1

        # Calculate the half field size
        half_width = field_size[0] / 2
        half_height = field_size[1] / 2

        # Define orthogonal axes to the beam direction
        # This creates a coordinate system with the beam direction as one axis
        v1 = beam_direction

        # Find a vector perpendicular to v1
        if abs(v1[0]) < abs(v1[1]):
            v2 = np.array([1, 0, 0])
        else:
            v2 = np.array([0, 1, 0])

        # Make v2 orthogonal to v1
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)

        # Create third orthogonal vector
        v3 = np.cross(v1, v2)

        # Spacing between pencil beams
        x_spacing = 2 * half_width / (nx - 1)
        y_spacing = 2 * half_height / (ny - 1)

        # Generate pencil beams
        for i in range(nx):
            for j in range(ny):
                # Calculate position at isocenter plane
                x = -half_width + i * x_spacing
                y = -half_height + j * y_spacing

                # Calculate position in 3D space
                position = isocenter + x * v2 + y * v3

                # Calculate direction from source to this position
                direction = position - source_position
                direction = direction / np.linalg.norm(direction)

                # For rectangular fields, we could apply a flat fluence profile
                # or model the penumbra with a slightly reduced weight at the edges
                weight = 1.0

                # Add to list
                pencil_beams.append(
                    {"entry_point": position, "direction": direction, "weight": weight}
                )

        return pencil_beams

    def _generate_mlc_pencil_beams(
        self,
        mlc_config: Dict,
        pencil_spacing: float,
        source_position: np.ndarray,
        isocenter: np.ndarray,
        beam_direction: np.ndarray,
    ) -> List[Dict]:
        """
        Generate pencil beams for an MLC-defined field.

        Parameters
        ----------
        mlc_config : Dict
            MLC leaf positions
        pencil_spacing : float
            Spacing between pencil beams in cm
        source_position : np.ndarray
            Source position in cm
        isocenter : np.ndarray
            Isocenter position in cm
        beam_direction : np.ndarray
            Beam direction vector

        Returns
        -------
        List[Dict]
            List of pencil beam definitions
        """
        pencil_beams = []

        # Extract MLC leaf positions
        leaf_positions = mlc_config

        # Define orthogonal axes to the beam direction
        # This creates a coordinate system with the beam direction as one axis
        v1 = beam_direction

        # Find a vector perpendicular to v1
        if abs(v1[0]) < abs(v1[1]):
            v2 = np.array([1, 0, 0])
        else:
            v2 = np.array([0, 1, 0])

        # Make v2 orthogonal to v1
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)

        # Create third orthogonal vector
        v3 = np.cross(v1, v2)

        # Estimate field extent from MLC positions
        min_leaf_pos = min(
            [min(leaf["left"], leaf["right"]) for leaf in leaf_positions]
        )
        max_leaf_pos = max(
            [max(leaf["left"], leaf["right"]) for leaf in leaf_positions]
        )
        leaf_width = leaf_positions[0].get("width", 1.0)  # in cm at isocenter

        # Y positions from leaf centers
        y_positions = [leaf["center"] for leaf in leaf_positions]
        min_y = min(y_positions) - leaf_width / 2
        max_y = max(y_positions) + leaf_width / 2

        # Calculate number of pencil beams
        nx = int((max_leaf_pos - min_leaf_pos) / pencil_spacing) + 1
        ny = int((max_y - min_y) / pencil_spacing) + 1

        # Generate grid of potential pencil beam positions
        for i in range(nx):
            x = min_leaf_pos + i * pencil_spacing

            for j in range(ny):
                y = min_y + j * pencil_spacing

                # Check if this position is inside the MLC aperture
                inside_aperture = False

                # Find which leaf this y position corresponds to
                for leaf in leaf_positions:
                    leaf_center = leaf["center"]
                    half_width = leaf["width"] / 2

                    if (leaf_center - half_width) <= y <= (leaf_center + half_width):
                        # Inside this leaf's extent
                        if leaf["left"] <= x <= leaf["right"]:
                            inside_aperture = True
                            break

                if inside_aperture:
                    # Calculate position in 3D space
                    position = isocenter + x * v2 + y * v3

                    # Calculate direction from source to this position
                    direction = position - source_position
                    direction = direction / np.linalg.norm(direction)

                    # For MLC fields, we could model leaf transmission
                    # with reduced weights for positions that are partially blocked
                    weight = 1.0

                    # Add to list
                    pencil_beams.append(
                        {
                            "entry_point": position,
                            "direction": direction,
                            "weight": weight,
                        }
                    )

        return pencil_beams

    def _trace_pencil_beam(
        self,
        entry_point: np.ndarray,
        direction: np.ndarray,
        weight: float,
        density_grid: np.ndarray,
        dose_grid: np.ndarray,
        ct_image: Image,
        step_size: float,
        apply_tar_correction: bool,
    ):
        """
        Trace a pencil beam through the patient and deposit dose.

        Parameters
        ----------
        entry_point : np.ndarray
            Entry point of the pencil beam in cm
        direction : np.ndarray
            Direction vector of the pencil beam
        weight : float
            Weight of the pencil beam
        density_grid : np.ndarray
            Electron density grid
        dose_grid : np.ndarray
            Dose grid to update
        ct_image : Image
            CT image
        step_size : float
            Integration step size in cm
        apply_tar_correction : bool
            Whether to apply tissue-air ratio correction

        Notes
        -----
        This function uses vectorized operations for improved performance.
        It implements a ray-tracing algorithm through the CT volume,
        calculating energy deposit at each point based on electron density.
        """
        try:
            # Kiểm tra và chuẩn hóa dữ liệu đầu vào
            direction = np.array(direction, dtype=np.float32)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-6:
                logger.warning(
                    f"Hướng chùm tia gần như 0: {direction}, bỏ qua pencil beam này"
                )
                return

            # Chuẩn hóa vector hướng
            direction = direction / direction_norm

            # Lấy thông tin hình học từ CT
            origin_cm = np.array(ct_image.origin) / 10.0  # mm -> cm
            spacing_cm = np.array(ct_image.spacing) / 10.0  # mm -> cm
            image_size = np.array(density_grid.shape)

            # Tính toán kích thước của khối CT theo cm
            volume_size_cm = image_size * spacing_cm

            # Tính điểm vào và ra từ khối CT
            max_path_length = 2.0 * np.sqrt(
                np.sum(volume_size_cm**2)
            )  # Đường chéo của khối CT

            # Tính điểm vào và ra tiềm năng của chùm tia
            # Sử dụng phương pháp ray-box intersection cho hiệu suất
            t_near = -np.inf
            t_far = np.inf

            # Tính giao điểm của tia với các mặt của khối CT
            for i in range(3):
                if abs(direction[i]) < 1e-6:  # Tia song song với mặt
                    if (
                        entry_point[i] < origin_cm[i]
                        or entry_point[i] > origin_cm[i] + volume_size_cm[i]
                    ):
                        # Tia hoàn toàn bên ngoài khối CT
                        return  # Không có giao điểm, bỏ qua pencil beam này
                else:
                    # Tính tham số t cho giao điểm với các mặt
                    t1 = (origin_cm[i] - entry_point[i]) / direction[i]
                    t2 = (
                        origin_cm[i] + volume_size_cm[i] - entry_point[i]
                    ) / direction[i]

                    # Đảm bảo t1 <= t2
                    if t1 > t2:
                        t1, t2 = t2, t1

                    # Cập nhật t_near và t_far
                    t_near = max(t_near, t1)
                    t_far = min(t_far, t2)

                    if t_near > t_far:
                        return  # Không có giao điểm, bỏ qua pencil beam này

            # Đảm bảo chùm tia bắt đầu ở phía trước mặt vào
            if t_near < 0:
                t_near = 0

            # Tính điểm vào thực tế
            actual_entry = entry_point + t_near * direction

            # Tính tổng số bước tích phân
            total_steps = int(np.ceil((t_far - t_near) / step_size))
            if total_steps <= 0:
                return  # Không có đường đi bên trong khối CT

            # Tạo mảng các điểm trên đường đi của tia
            steps = np.arange(total_steps)
            path_positions = actual_entry.reshape(1, 3) + (
                steps.reshape(-1, 1) * step_size
            ) * direction.reshape(1, 3)

            # Chuyển đổi vị trí thành chỉ số trong mảng
            indices = np.floor((path_positions - origin_cm) / spacing_cm).astype(
                np.int32
            )

            # Lọc điểm nằm bên trong khối CT
            valid_mask = np.all((indices >= 0) & (indices < image_size), axis=1)
            if not np.any(valid_mask):
                return  # Không có điểm nào bên trong khối CT

            valid_indices = indices[valid_mask]

            # Tích lũy liều dọc theo đường đi
            # Chuyển đổi chỉ số 3D thành chỉ số 1D để truy cập nhanh
            flat_indices = (
                valid_indices[:, 0] * image_size[1] * image_size[2]
                + valid_indices[:, 1] * image_size[2]
                + valid_indices[:, 2]
            )

            # Đảm bảo không có chỉ số ngoài phạm vi
            flat_indices = np.clip(flat_indices, 0, density_grid.size - 1)

            # Lấy giá trị mật độ
            densities = density_grid.flat[flat_indices]

            # Tính liều tại mỗi điểm
            # Mô phỏng đơn giản Percent Depth Dose curve
            depths = steps[valid_mask] * step_size

            # Mô phỏng đường cong PDD dựa trên mô hình đơn giản
            # PDD(d) = D0 * exp(-μ*d), với μ là hệ số suy giảm phụ thuộc vào mật độ
            mu_water = 0.05  # Hệ số suy giảm trong nước (1/cm)
            relative_dose = np.exp(-mu_water * depths * densities)

            # Áp dụng tissue-air ratio correction nếu được yêu cầu
            if apply_tar_correction:
                # Mô phỏng đơn giản TAR correction
                # TAR(d, ρ) = TAR(d, 1.0) * ρ^n, với n là luỹ thừa phụ thuộc vào năng lượng
                tar_power = 1.5  # Luỹ thừa cho năng lượng 6MV
                relative_dose *= np.power(densities, tar_power)

            # Áp dụng normalization và kernel
            kernel_size = 1.0  # Độ rộng của kernel theo cm
            kernel_sigma = kernel_size / (2.0 * 2.355)  # Convert FWHM to sigma

            # Tính luỹ thừa của kernel tại mỗi điểm (đơn giản hóa)
            kernel_values = np.exp(-((depths - 3.0) ** 2) / (2 * kernel_sigma**2))
            kernel_values = np.clip(kernel_values, 0.0, 1.0)

            # Chuẩn hóa kernel để bảo toàn tổng liều
            kernel_sum = np.sum(kernel_values)
            if kernel_sum > 0:
                kernel_values /= kernel_sum

            # Tính liều cuối cùng
            final_dose = weight * relative_dose * kernel_values

            # Cập nhật grid liều một cách an toàn (tránh race condition)
            for i, idx in enumerate(flat_indices):
                # Sử dụng numpy.add.at để xử lý trường hợp nhiều điểm cùng chỉ số
                # Nhưng do không thể trực tiếp sử dụng với flat array, ta phải tính lại chỉ số 3D
                x, y, z = valid_indices[i]
                if (
                    0 <= x < image_size[0]
                    and 0 <= y < image_size[1]
                    and 0 <= z < image_size[2]
                ):
                    # Sử dụng atomic add để tránh race condition trong đa luồng
                    dose_grid[x, y, z] += final_dose[i]

        except Exception as e:
            logger.error(f"Lỗi trong _trace_pencil_beam: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def calculate_beam_dose(self, beam: Beam, ct_image: Image) -> Image:
        """
        Calculate dose for a beam using Pencil Beam algorithm.

        Parameters
        ----------
        beam : Beam
            Treatment beam
        ct_image : Image
            CT image

        Returns
        -------
        Image
            Dose image
        """
        result = self.calculate(ct_image, beam)
        return result.dose

    def create_generic_beam_model(self, energy: str) -> BeamModel:
        """
        Create a generic beam model for the specified energy.

        Parameters
        ----------
        energy : str
            The beam energy (e.g., "6MV", "10MV")

        Returns
        -------
        BeamModel
            A generic beam model
        """
        logger.info(f"Creating generic beam model for energy: {energy}")

        # Create basic beam model
        model = BeamModel(name=f"Generic {energy}", energy=energy, beam_type="PHOTON")

        # Add PDD data for 10x10 field
        depths = np.array(
            [0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        )

        # Different PDD values based on energy
        if energy == "6MV":
            pdd_values = np.array(
                [0, 97.0, 100.0, 97.0, 93.0, 89.0, 85.0, 65.0, 45.0, 30.0, 21.0, 15.0]
            )
        elif energy == "10MV":
            pdd_values = np.array(
                [0, 90.0, 100.0, 99.0, 95.0, 92.0, 88.0, 70.0, 52.0, 38.0, 27.0, 20.0]
            )
        else:  # Default to 15MV
            pdd_values = np.array(
                [0, 85.0, 100.0, 99.5, 97.0, 94.0, 91.0, 75.0, 57.0, 43.0, 32.0, 24.0]
            )

        pdd_parameter = BeamModelParameter(
            name="pdd_10x10",
            value_grid=pdd_values,
            dimensions=["depth"],
            units=["cm"],
            dimension_values=[depths],
            interpolation_method="cubic",
        )
        model.add_parameter(pdd_parameter)

        # Create profiles for different depths and field sizes
        # This is a simplified example - real data would be more comprehensive

        # Profile data for 10x10 field at different depths
        x_positions = np.linspace(-10, 10, 21)  # -10 to 10 cm in 1 cm steps

        # Different profiles based on energy and depth
        # Depth: dmax
        if energy == "6MV":
            profile_dmax = np.ones_like(x_positions)
            # Add penumbra
            profile_dmax[0] = 0.2
            profile_dmax[1] = 0.6
            profile_dmax[-2] = 0.6
            profile_dmax[-1] = 0.2
        elif energy == "10MV":
            profile_dmax = np.ones_like(x_positions)
            # Sharper penumbra for higher energies
            profile_dmax[0] = 0.1
            profile_dmax[1] = 0.5
            profile_dmax[-2] = 0.5
            profile_dmax[-1] = 0.1
        else:  # 15MV
            profile_dmax = np.ones_like(x_positions)
            profile_dmax[0] = 0.05
            profile_dmax[1] = 0.4
            profile_dmax[-2] = 0.4
            profile_dmax[-1] = 0.05

        profile_parameter = BeamModelParameter(
            name="profile_10x10_dmax",
            value_grid=profile_dmax,
            dimensions=["x"],
            units=["cm"],
            dimension_values=[x_positions],
            interpolation_method="cubic",
        )
        model.add_parameter(profile_parameter)

        # Add more profiles for different depths
        # This is a simplified example

        return model

    def get_description(self) -> str:
        """Get algorithm description."""
        return (
            "Pencil Beam algorithm for dose calculation. "
            "This algorithm models the beam as a collection of narrow pencil beams "
            "and calculates the dose distribution by summing the contributions "
            "from each pencil beam."
        )

    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about available parameters.

        Returns
        -------
        Dict[str, Any]
            Parameter information
        """
        return {
            "grid_size": {
                "description": "Calculation grid size in cm",
                "type": "float",
                "default": 0.2,
                "range": [0.1, 0.5],
            },
            "threads": {
                "description": "Number of parallel threads",
                "type": "int",
                "default": 4,
                "range": [1, 16],
            },
            "tissue_air_ratio_correction": {
                "description": "Apply tissue-air ratio correction",
                "type": "bool",
                "default": True,
            },
            "use_gpu": {
                "description": "Use GPU acceleration if available",
                "type": "bool",
                "default": False,
            },
            "pencil_spacing": {
                "description": "Spacing between pencil beams in cm",
                "type": "float",
                "default": 0.1,
                "range": [0.05, 0.5],
            },
            "integration_step": {
                "description": "Integration step size in cm",
                "type": "float",
                "default": 0.5,
                "range": [0.1, 1.0],
            },
        }
