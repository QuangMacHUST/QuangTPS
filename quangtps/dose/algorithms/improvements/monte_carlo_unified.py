#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API thống nhất cho thuật toán tính toán liều Monte Carlo.

Module này cung cấp một API thống nhất để sử dụng các thuật toán Monte Carlo
trên nhiều loại phần cứng và công nghệ khác nhau (CPU, CuPy, PyCUDA, OpenCL).
"""

import logging
import os
import time
import warnings
import multiprocessing
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

# Thử import DoseCalculationAlgorithm và DoseCalculationResult từ module base
try:
    from quangtps.dose.algorithms.base import (
        DoseCalculationAlgorithm,
        DoseCalculationResult,
    )
except ImportError:
    # Tạo lớp mock nếu không thể import
    class DoseCalculationAlgorithm:
        """Mock class cho DoseCalculationAlgorithm."""

        def __init__(self):
            self.name = "MockDoseCalculationAlgorithm"
            self.version = "0.0.0"

    class DoseCalculationResult:
        """Mock class cho DoseCalculationResult."""

        def __init__(
            self, dose_grid=None, algorithm_name=None, calculation_time=0, metadata=None
        ):
            self.dose_grid = dose_grid
            self.algorithm_name = algorithm_name
            self.calculation_time = calculation_time
            self.metadata = metadata or {}


# Thử import các exception cần thiết hoặc tạo mock nếu không tìm thấy
try:
    from quangtps.core.exceptions import DoseCalculationError
except ImportError:

    class DoseCalculationError(Exception):
        """Mock exception cho DoseCalculationError."""

        pass


from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam

logger = logging.getLogger(__name__)


class AccelerationType(Enum):
    CPU = auto()
    CUPY = auto()
    PYCUDA = auto()
    OPENCL = auto()


# Initialize acceleration libraries
HAS_CUPY = False
HAS_PYCUDA = False
HAS_OPENCL = False

try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy đã được tải thành công để tăng tốc GPU")
except ImportError:
    logger.debug("CuPy không khả dụng, thử phương pháp tăng tốc khác")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray

    HAS_PYCUDA = True
    logger.info("PyCUDA đã được tải thành công để tăng tốc GPU")
except ImportError:
    logger.debug("PyCUDA không khả dụng, thử phương pháp tăng tốc khác")

try:
    import pyopencl as cl

    HAS_OPENCL = True
    logger.info("PyOpenCL đã được tải thành công để tăng tốc GPU")
except ImportError:
    logger.debug("PyOpenCL không khả dụng")

# Kiểm tra xem GPU tăng tốc có khả dụng không
HAS_GPU = HAS_CUPY or HAS_PYCUDA or HAS_OPENCL


class UnifiedMonteCarloAPI:
    """
    API thống nhất cho thuật toán Monte Carlo hỗ trợ nhiều loại phần cứng.

    Lớp này cung cấp một giao diện đơn giản để chạy thuật toán Monte Carlo
    trên nhiều loại phần cứng khác nhau, bao gồm CPU, GPU (CuPy, PyCUDA) và OpenCL.
    """

    def __init__(
        self,
        preferred_backend: AccelerationType = None,
        device_ids: List[int] = None,
        **kwargs,
    ):
        """
        Khởi tạo UnifiedMonteCarloAPI.

        Parameters
        ----------
        preferred_backend : AccelerationType, optional
            Backend ưu tiên sử dụng
        device_ids : List[int], optional
            Danh sách ID thiết bị cụ thể để sử dụng
        **kwargs
            Các tham số tùy chọn khác:
            - verbose: bool - Hiện thông tin chi tiết
            - fallback_order: List[AccelerationType] - Thứ tự backend dự phòng
            - num_histories: int - Số lịch sử Monte Carlo mặc định
        """
        self.verbose = kwargs.get("verbose", False)
        self.fallback_order = kwargs.get(
            "fallback_order",
            [
                AccelerationType.CUPY,  # Ưu tiên CUPY (hiệu quả nhất)
                AccelerationType.PYCUDA,  # Sau đó đến PYCUDA
                AccelerationType.OPENCL,  # Sau đó đến OpenCL
                AccelerationType.CPU,  # Cuối cùng là CPU (chậm nhất)
            ],
        )
        self.num_histories = kwargs.get("num_histories", 1_000_000)

        # Lưu các thiết bị được chỉ định
        self.device_ids = device_ids

        # Các thuộc tính sẽ được thiết lập trong _select_backend và _initialize_backend
        self.backend_type = None  # Loại backend đang sử dụng
        self.backend_module = None  # Module thực tế được sử dụng
        self.backend_devices = []  # Danh sách thiết bị có sẵn
        self.backend_functions = {}  # Dict các hàm được cung cấp bởi backend

        # Quét các backend có sẵn
        self.available_backends = self._scan_available_backends()

        if self.verbose:
            logger.info(
                f"Các backend có sẵn: {[b.name for b in self.available_backends]}"
            )

        # Chọn và khởi tạo backend
        self._select_backend(preferred_backend, device_ids)
        self._initialize_backend()

    def _scan_available_backends(self):
        """Xác định tất cả các backend khả dụng trong hệ thống."""
        available_backends = []

        # Luôn thêm CPU - luôn khả dụng
        available_backends.append(AccelerationType.CPU)

        # Kiểm tra và thêm các backend GPU
        if HAS_CUPY:
            try:
                num_gpus = cp.cuda.runtime.getDeviceCount()
                if num_gpus > 0:
                    available_backends.append(AccelerationType.CUPY)
                    logger.info(f"Tìm thấy {num_gpus} GPU hỗ trợ với CuPy")
            except Exception as e:
                logger.warning(f"Lỗi khi kiểm tra GPU CuPy: {e}")

        if HAS_PYCUDA:
            try:
                num_gpus = cuda.Device.count()
                if num_gpus > 0:
                    available_backends.append(AccelerationType.PYCUDA)
                    logger.info(f"Tìm thấy {num_gpus} GPU hỗ trợ với PyCUDA")
            except Exception as e:
                logger.warning(f"Lỗi khi kiểm tra GPU PyCUDA: {e}")

        if HAS_OPENCL:
            try:
                platforms = cl.get_platforms()
                if platforms:
                    devices = []
                    for platform in platforms:
                        devices.extend(
                            platform.get_devices(device_type=cl.device_type.GPU)
                        )

                    if devices:
                        available_backends.append(AccelerationType.OPENCL)
                        logger.info(f"Tìm thấy {len(devices)} GPU hỗ trợ với OpenCL")
            except Exception as e:
                logger.warning(f"Lỗi khi kiểm tra GPU OpenCL: {e}")

        return available_backends

    def _select_backend(self, preferred_backend, device_ids):
        """Chọn backend tính toán phù hợp nhất."""
        # Nếu người dùng yêu cầu backend cụ thể
        if preferred_backend is not None:
            if preferred_backend in self.available_backends:
                self.backend_type = preferred_backend
                logger.info(f"Sử dụng backend được yêu cầu: {preferred_backend.name}")
            else:
                logger.warning(
                    f"Backend được yêu cầu {preferred_backend.name} không khả dụng. "
                    f"Sử dụng tự động chọn."
                )
                preferred_backend = None

        # Tự động chọn backend tốt nhất (nếu không có yêu cầu hoặc yêu cầu không khả dụng)
        if preferred_backend is None:
            # Ưu tiên theo thứ tự: CuPy > PyCUDA > OpenCL > CPU
            if AccelerationType.CUPY in self.available_backends:
                self.backend_type = AccelerationType.CUPY
            elif AccelerationType.PYCUDA in self.available_backends:
                self.backend_type = AccelerationType.PYCUDA
            elif AccelerationType.OPENCL in self.available_backends:
                self.backend_type = AccelerationType.OPENCL
            else:
                self.backend_type = AccelerationType.CPU

            logger.info(f"Tự động chọn backend: {self.backend_type.name}")

    def _initialize_backend(self):
        """Khởi tạo backend đã chọn."""
        if self.backend_type == AccelerationType.CPU:
            self._initialize_cpu_backend()
        elif self.backend_type == AccelerationType.CUPY:
            self._initialize_cupy_backend()
        elif self.backend_type == AccelerationType.PYCUDA:
            self._initialize_pycuda_backend()
        elif self.backend_type == AccelerationType.OPENCL:
            self._initialize_opencl_backend()

    def _initialize_cpu_backend(self):
        """Khởi tạo backend CPU."""
        num_cores = multiprocessing.cpu_count()
        num_threads = max(
            1, min(num_cores - 1, self.config.get("max_threads", num_cores - 1))
        )

        self.device_info = {
            "device_type": "CPU",
            "num_cores": num_cores,
            "num_threads": num_threads,
            "memory_total": None,  # Không có cách đáng tin cậy để lấy tổng RAM
            "device_name": f"{num_cores} CPU cores",
        }

        logger.info(f"Đã khởi tạo backend CPU với {num_threads} luồng")

    def _initialize_cupy_backend(self):
        """Khởi tạo backend CuPy GPU."""
        devices = []
        device_ids = self.config.get("device_ids", None)

        # Lấy tổng số GPU khả dụng
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()

            # Chọn các GPU cụ thể nếu được chỉ định
            if device_ids:
                gpu_indices = [i for i in device_ids if i < num_gpus]
            else:
                gpu_indices = list(range(num_gpus))

            # Khởi tạo từng GPU
            for idx in gpu_indices:
                device = cp.cuda.Device(idx)
                mem_info = device.mem_info
                props = cp.cuda.runtime.getDeviceProperties(idx)

                device_info = {
                    "device_id": idx,
                    "device_name": props["name"].decode("utf-8"),
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "memory_total": mem_info[1],
                    "memory_free": mem_info[0],
                    "multiprocessor_count": props["multiProcessorCount"],
                }

                devices.append({"device": device, "info": device_info})

                logger.info(
                    f"GPU {idx}: {device_info['device_name']} - "
                    f"{device_info['memory_free'] / 1e9:.2f}/{device_info['memory_total'] / 1e9:.2f} GB"
                )

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo CuPy: {e}")
            # Chuyển về backend CPU
            self.backend_type = AccelerationType.CPU
            self._initialize_cpu_backend()
            return

        self.backend_devices = devices
        if devices:
            # Sử dụng thông tin GPU đầu tiên cho device_info chung
            self.device_info = devices[0]["info"]
            self.device_info["num_devices"] = len(devices)
        else:
            logger.error("Không tìm thấy GPU khả dụng cho CuPy")
            # Chuyển về backend CPU
            self.backend_type = AccelerationType.CPU
            self._initialize_cpu_backend()

    def _initialize_pycuda_backend(self):
        """Khởi tạo backend PyCUDA GPU."""
        devices = []
        device_ids = self.config.get("device_ids", None)

        try:
            num_gpus = cuda.Device.count()

            # Chọn các GPU cụ thể nếu được chỉ định
            if device_ids:
                gpu_indices = [i for i in device_ids if i < num_gpus]
            else:
                gpu_indices = list(range(num_gpus))

            # Khởi tạo từng GPU
            for idx in gpu_indices:
                device = cuda.Device(idx)
                device_attrs = device.get_attributes()

                device_info = {
                    "device_id": idx,
                    "device_name": device.name(),
                    "compute_capability": f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
                    "memory_total": device.total_memory(),
                    "memory_free": device.total_memory() - device.used_memory(),
                }

                devices.append({"device": device, "info": device_info})

                logger.info(
                    f"GPU {idx}: {device_info['device_name']} - "
                    f"{device_info['memory_free'] / 1e9:.2f}/{device_info['memory_total'] / 1e9:.2f} GB"
                )

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo PyCUDA: {e}")
            # Chuyển về backend CPU
            self.backend_type = AccelerationType.CPU
            self._initialize_cpu_backend()
            return

        self.backend_devices = devices
        if devices:
            # Sử dụng thông tin GPU đầu tiên cho device_info chung
            self.device_info = devices[0]["info"]
            self.device_info["num_devices"] = len(devices)
        else:
            logger.error("Không tìm thấy GPU khả dụng cho PyCUDA")
            # Chuyển về backend CPU
            self.backend_type = AccelerationType.CPU
            self._initialize_cpu_backend()

    def _initialize_opencl_backend(self):
        """Khởi tạo backend OpenCL GPU."""
        devices = []

        try:
            platforms = cl.get_platforms()

            if not platforms:
                logger.error("Không tìm thấy nền tảng OpenCL")
                # Chuyển về backend CPU
                self.backend_type = AccelerationType.CPU
                self._initialize_cpu_backend()
                return

            # Tìm tất cả GPU OpenCL
            for platform_idx, platform in enumerate(platforms):
                platform_devices = platform.get_devices(device_type=cl.device_type.GPU)

                for device_idx, device in enumerate(platform_devices):
                    # Lấy thông tin thiết bị
                    device_info = {
                        "platform_id": platform_idx,
                        "platform_name": platform.name,
                        "device_id": device_idx,
                        "device_name": device.name,
                        "memory_total": device.global_mem_size,
                        "compute_units": device.max_compute_units,
                        "max_work_group_size": device.max_work_group_size,
                        "local_mem_size": device.local_mem_size,
                    }

                    # Tạo context và queue
                    context = cl.Context([device])
                    queue = cl.CommandQueue(context)

                    devices.append(
                        {
                            "device": device,
                            "context": context,
                            "queue": queue,
                            "info": device_info,
                        }
                    )

                    logger.info(
                        f"OpenCL GPU: {device_info['platform_name']} - {device_info['device_name']} - "
                        f"{device_info['memory_total'] / 1e9:.2f} GB"
                    )

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo OpenCL: {e}")
            # Chuyển về backend CPU
            self.backend_type = AccelerationType.CPU
            self._initialize_cpu_backend()
            return

        self.backend_devices = devices
        if devices:
            # Sử dụng thông tin GPU đầu tiên cho device_info chung
            self.device_info = devices[0]["info"]
            self.device_info["num_devices"] = len(devices)
        else:
            logger.error("Không tìm thấy GPU khả dụng cho OpenCL")
            # Chuyển về backend CPU
            self.backend_type = AccelerationType.CPU
            self._initialize_cpu_backend()

    def get_backend_info(self):
        """
        Lấy thông tin backend hiện tại đang sử dụng.

        Returns
        -------
        Dict[str, Any]
            Thông tin về backend và thiết bị
        """
        return {
            "backend_type": self.backend_type.name,
            "device_info": self.device_info,
            "num_devices": len(self.backend_devices) if self.backend_devices else 1,
        }

    def calculate(
        self,
        ct_data: np.ndarray,
        structures: Dict[str, np.ndarray],
        beam_config: Dict[str, Any],
        settings: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Tính toán phân phối liều sử dụng Monte Carlo.

        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        structures : Dict[str, np.ndarray]
            Cấu trúc ROI (vùng quan tâm) - mặt nạ boolean 3D đại diện cho các cấu trúc đích và nguy cấp
        beam_config : Dict[str, Any]
            Cấu hình chùm tia (năng lượng, góc, v.v.)
        settings : Dict[str, Any], tùy chọn
            Cài đặt bổ sung

        Returns
        -------
        Dict[str, Any]
            Kết quả tính toán liều
        """
        settings = settings or {}

        # Gọi phương thức tính toán dựa trên backend đã chọn
        if self.backend_type == AccelerationType.CPU:
            return self._calculate_cpu(ct_data, structures, beam_config, settings)
        elif self.backend_type == AccelerationType.CUPY:
            return self._calculate_cupy(ct_data, structures, beam_config, settings)
        elif self.backend_type == AccelerationType.PYCUDA:
            return self._calculate_pycuda(ct_data, structures, beam_config, settings)
        elif self.backend_type == AccelerationType.OPENCL:
            return self._calculate_opencl(ct_data, structures, beam_config, settings)
        else:
            raise ValueError(f"Backend không được hỗ trợ: {self.backend_type}")

    def _calculate_cpu(self, ct_data, structures, beam_config, settings):
        """Triển khai tính toán trên CPU."""
        # Triển khai chi tiết sẽ được thêm vào đây...
        # Đây chỉ là phương thức giữ chỗ (placeholder)
        logger.info("Tính toán trên CPU...")
        # Đoạn code tính toán thực tế sẽ đi ở đây
        return {
            "status": "not_implemented",
            "message": "Tính toán CPU chưa được triển khai đầy đủ",
        }

    def _calculate_cupy(self, ct_data, structures, beam_config, settings):
        """Triển khai tính toán với CuPy."""
        # Triển khai chi tiết sẽ được thêm vào đây...
        # Đây chỉ là phương thức giữ chỗ (placeholder)
        logger.info("Tính toán với CuPy...")
        # Đoạn code tính toán thực tế sẽ đi ở đây
        return {
            "status": "not_implemented",
            "message": "Tính toán CuPy chưa được triển khai đầy đủ",
        }

    def _calculate_pycuda(self, ct_data, structures, beam_config, settings):
        """Triển khai tính toán với PyCUDA."""
        # Triển khai chi tiết sẽ được thêm vào đây...
        # Đây chỉ là phương thức giữ chỗ (placeholder)
        logger.info("Tính toán với PyCUDA...")
        # Đoạn code tính toán thực tế sẽ đi ở đây
        return {
            "status": "not_implemented",
            "message": "Tính toán PyCUDA chưa được triển khai đầy đủ",
        }

    def _calculate_opencl(self, ct_data, structures, beam_config, settings):
        """Triển khai tính toán với OpenCL."""
        # Triển khai chi tiết sẽ được thêm vào đây...
        # Đây chỉ là phương thức giữ chỗ (placeholder)
        logger.info("Tính toán với OpenCL...")
        # Đoạn code tính toán thực tế sẽ đi ở đây
        return {
            "status": "not_implemented",
            "message": "Tính toán OpenCL chưa được triển khai đầy đủ",
        }


class UnifiedMonteCarloAlgorithm(DoseCalculationAlgorithm):
    """
    Thuật toán tính liều Monte Carlo thống nhất hỗ trợ nhiều backend tính toán.

    Lớp này kế thừa từ DoseCalculationAlgorithm và sử dụng UnifiedMonteCarloAPI
    để tính toán liều với nhiều backend khác nhau (CPU, GPU, OpenCL).
    """

    def __init__(self):
        """Khởi tạo thuật toán Monte Carlo thống nhất."""
        # Thử gọi constructor của lớp cha nếu phương thức tồn tại
        try:
            super().__init__()
        except (AttributeError, TypeError):
            # Nếu lớp cha không có constructor hoặc có lỗi, tự thiết lập các thuộc tính cơ bản
            self.name = "UnifiedMonteCarloAlgorithm"
            self.version = "1.0.0"
            self.description = "Thuật toán tính liều Monte Carlo thống nhất đa backend"

        # Các tham số mặc định
        self.parameters = {
            "num_histories": 1_000_000,  # Số lịch sử Monte Carlo
            "statistical_uncertainty": 0.02,  # Độ không đảm bảo thống kê mục tiêu (2%)
            "voxel_size": 0.2,  # Kích thước voxel mặc định (cm)
            "use_gpu": True,  # Sử dụng GPU nếu có
            "preferred_backend": None,  # Backend ưa thích (tự động chọn)
            "device_ids": None,  # ID thiết bị (tự động chọn)
            "verbose": False,  # Hiển thị thông tin chi tiết
        }

        # Khởi tạo beam model và API
        self.beam_model = None
        self.monte_carlo_api = None

        # Khởi tạo API
        self._initialize_api()

    def _initialize_api(self):
        """Khởi tạo API Monte Carlo thống nhất với các tham số hiện tại."""
        try:
            # Lấy các tham số từ thuật toán
            preferred_backend_str = self.parameters.get("preferred_backend", None)
            preferred_backend = None

            if preferred_backend_str:
                # Chuyển đổi string thành enum
                try:
                    preferred_backend = AccelerationType[preferred_backend_str]
                except KeyError:
                    logger.warning(
                        f"Backend không hợp lệ: {preferred_backend_str}. Sử dụng tự động."
                    )

            device_ids = self.parameters.get("device_ids", None)
            verbose = self.parameters.get("verbose", False)
            num_histories = self.parameters.get("num_histories", 1_000_000)

            # Khởi tạo API Monte Carlo
            self.monte_carlo_api = UnifiedMonteCarloAPI(
                preferred_backend=preferred_backend,
                device_ids=device_ids,
                verbose=verbose,
                num_histories=num_histories,
            )

            logger.info(
                f"Đã khởi tạo API Monte Carlo với backend: {self.monte_carlo_api.backend_type.name}"
            )

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo API Monte Carlo: {e}")
            # Đặt monte_carlo_api thành None để chỉ ra lỗi
            self.monte_carlo_api = None
            raise

    def set_beam_model(self, beam_model):
        """Cài đặt mô hình chùm tia cho thuật toán."""
        self.beam_model = beam_model

    def calculate(self, ct_image, beam):
        """
        Tính toán liều bằng thuật toán Monte Carlo thống nhất.

        Parameters
        ----------
        ct_image : Image
            Hình ảnh CT
        beam : Beam
            Thông tin chùm tia

        Returns
        -------
        DoseCalculationResult
            Kết quả tính toán liều
        """
        start_time = time.time()

        try:
            # Kiểm tra API đã được khởi tạo
            if self.monte_carlo_api is None:
                self._initialize_api()
                if self.monte_carlo_api is None:
                    raise DoseCalculationError("Không thể khởi tạo API Monte Carlo")

            # Xác thực đầu vào
            self.validate_inputs(ct_image, beam)

            # Chuẩn bị dữ liệu đầu vào
            ct_data = ct_image.data

            # Chuẩn bị cấu trúc (nếu có)
            structures = {}
            if hasattr(beam, "target_structure") and beam.target_structure is not None:
                structures["target"] = beam.target_structure

            # Tạo cấu hình chùm tia
            beam_config = {
                "energy": beam.energy if hasattr(beam, "energy") else 6.0,  # MeV
                "sad": beam.sad if hasattr(beam, "sad") else 1000.0,  # mm
                "field_size": beam.field_size
                if hasattr(beam, "field_size")
                else [10, 10],  # cm
                "isocenter": beam.isocenter
                if hasattr(beam, "isocenter")
                else [0, 0, 0],  # mm
                "gantry_angle": beam.gantry_angle
                if hasattr(beam, "gantry_angle")
                else 0.0,  # degrees
                "collimator_angle": beam.collimator_angle
                if hasattr(beam, "collimator_angle")
                else 0.0,  # degrees
                "couch_angle": beam.couch_angle
                if hasattr(beam, "couch_angle")
                else 0.0,  # degrees
                "mlc": beam.mlc.get_leaf_positions()
                if hasattr(beam, "mlc") and beam.mlc is not None
                else None,
            }

            # Cài đặt cho tính toán
            settings = {
                "num_histories": self.parameters.get("num_histories", 10000000),
                "statistical_uncertainty": self.parameters.get(
                    "statistical_uncertainty", 1.0
                ),
                "grid_size": self.parameters.get("grid_size", 0.3),
                "electron_cutoff": self.parameters.get("electron_cutoff", 0.1),
                "photon_cutoff": self.parameters.get("photon_cutoff", 0.01),
                "beam_model_type": self.parameters.get("beam_model_type", "analytical"),
                "beam_model": self.beam_model,
            }

            # Thực hiện tính toán
            logger.info(
                f"Bắt đầu tính toán Monte Carlo với {settings['num_histories']} hạt"
            )
            result = self.monte_carlo_api.calculate(
                ct_data=ct_data,
                structures=structures,
                beam_config=beam_config,
                settings=settings,
            )

            # Kiểm tra kết quả
            if result.get("status") == "not_implemented":
                raise NotImplementedError(
                    f"Phương thức tính toán chưa được triển khai đầy đủ: {result.get('message')}"
                )

            # Tạo đối tượng kết quả
            calculation_time = time.time() - start_time

            # Tạo ảnh liều từ kết quả
            dose_data = result.get("dose_grid", np.zeros_like(ct_data))
            dose_image = Image(
                data=dose_data,
                origin=ct_image.origin,
                spacing=ct_image.spacing,
                direction=ct_image.direction,
            )

            # Tạo đối tượng kết quả
            calculation_result = DoseCalculationResult(
                dose_grid=dose_image,
                algorithm_name=self.name,
                calculation_time=calculation_time,
                metadata={
                    "version": self.version,
                    "backend": self.monte_carlo_api.backend_type.name,
                    "histories": settings["num_histories"],
                    "uncertainty": result.get("uncertainty", 0.0),
                    "beam": beam.name if hasattr(beam, "name") else "unknown",
                    "performance": result.get("performance", {}),
                },
            )

            logger.info(
                f"Tính toán Monte Carlo hoàn tất trong {calculation_time:.2f} giây"
            )
            return calculation_result

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tính toán Monte Carlo: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            raise DoseCalculationError(f"Lỗi tính toán Monte Carlo: {str(e)}")

    def validate_inputs(self, ct_image, beam):
        """Xác thực dữ liệu đầu vào trước khi tính toán."""
        if ct_image is None:
            raise ValueError("Hình ảnh CT không được cung cấp")

        if ct_image.data is None or ct_image.data.size == 0:
            raise ValueError("Dữ liệu CT không hợp lệ hoặc rỗng")

        if not isinstance(ct_image.data, np.ndarray):
            raise ValueError(
                f"Dữ liệu CT phải là mảng numpy, không phải {type(ct_image.data)}"
            )

        # Kiểm tra chùm tia
        if beam is None:
            raise ValueError("Chùm tia không được cung cấp")

        # Kiểm tra các thông số cần thiết của chùm tia
        required_attrs = ["isocenter", "field_size"]
        for attr in required_attrs:
            if not hasattr(beam, attr):
                raise ValueError(f"Chùm tia thiếu thuộc tính {attr}")

    def get_description(self):
        """Lấy mô tả về thuật toán."""
        return (
            "Thuật toán Monte Carlo thống nhất cho tính toán liều trong lập kế hoạch xạ trị. "
            "Tự động chọn phương pháp tính toán tối ưu dựa trên phần cứng khả dụng (CPU, CUDA, OpenCL). "
            "Cung cấp độ chính xác cao nhất cho tính toán liều trong các trường hợp phức tạp, "
            "đặc biệt là khi có sự không đồng nhất mô và các bề mặt giao diện."
        )
