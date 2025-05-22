#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module Factory cho thuật toán tính liều Monte Carlo.

Module này cung cấp các lớp và hàm để tự động chọn và khởi tạo
thuật toán Monte Carlo phù hợp dựa trên phần cứng có sẵn và yêu cầu độ chính xác.
"""

import logging
import os
import platform
import sys
import multiprocessing
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Union, Type

logger = logging.getLogger(__name__)

# Nhập các module tùy chọn với xử lý ngoại lệ
try:
    from quangtps.dose.algorithms.base import DoseCalculationAlgorithm
except ImportError:
    # Tạo lớp giả nếu không thể import
    class DoseCalculationAlgorithm:
        """Mock class cho DoseCalculationAlgorithm."""

        def __init__(self):
            logger.debug("Sử dụng DoseCalculationAlgorithm giả")

        def calculate(self, *args, **kwargs):
            logger.error("Phương thức calculate không được triển khai")
            return None


class MonteCarloImplementation(Enum):
    """Các loại cài đặt Monte Carlo có sẵn."""

    STANDARD = auto()  # Thuật toán Monte Carlo tiêu chuẩn dựa trên CPU
    GPU_BASIC = auto()  # Thuật toán Monte Carlo đơn giản tăng tốc GPU
    GPU_ADVANCED = auto()  # Thuật toán Monte Carlo tối ưu tăng tốc GPU
    UNIFIED = auto()  # API thống nhất đa platform (CPU, CuPy, PyCUDA, OpenCL)
    VMCPRO = auto()  # Biến thể độc quyền của VMC Pro
    CUSTOM = auto()  # Cài đặt tùy chỉnh do người dùng xác định


class AccuracyLevel(Enum):
    """Mức độ chính xác mong muốn cho tính liều."""

    LOW = auto()  # Độ chính xác thấp, tốc độ cao
    MEDIUM = auto()  # Cân bằng độ chính xác và tốc độ
    HIGH = auto()  # Độ chính xác cao, tốc độ chậm hơn
    ULTRA = auto()  # Độ chính xác cực cao, tốc độ chậm


class HardwareType(Enum):
    """Loại phần cứng có sẵn để tính toán."""

    CPU = auto()  # Chỉ CPU
    CPU_MULTI = auto()  # Nhiều CPU/đa luồng
    GPU_BASIC = auto()  # GPU cấp thấp (OpenCL)
    GPU_NVIDIA = auto()  # NVIDIA GPU (CUDA)
    GPU_AMD = auto()  # AMD GPU
    GPU_MULTI = auto()  # Nhiều GPU
    CLOUD = auto()  # Cloud computing
    CUSTOM = auto()  # Cấu hình phần cứng tùy chỉnh


class MonteCarloFactory:
    """
    Factory class để tạo và cấu hình các thuật toán Monte Carlo.

    Lớp này cung cấp giao diện để tạo các thuật toán Monte Carlo phù hợp
    dựa trên phần cứng có sẵn và mức độ chính xác mong muốn.
    """

    def __init__(self):
        """Khởi tạo MonteCarloFactory."""
        # Phát hiện phần cứng và thuật toán có sẵn
        self._available_implementations = self._detect_available_implementations()
        self._hardware_type = self._detect_hardware_capabilities()

        logger.info(
            f"Các thuật toán Monte Carlo có sẵn: {self._available_implementations}"
        )
        logger.info(f"Loại phần cứng phát hiện: {self._hardware_type}")

        # Lưu trữ triển khai được đề xuất
        self._recommended_implementation = self._select_implementation(
            AccuracyLevel.MEDIUM, self._hardware_type
        )
        logger.info(
            f"Triển khai Monte Carlo được đề xuất: {self._recommended_implementation}"
        )

    def _detect_available_implementations(self) -> List[MonteCarloImplementation]:
        """
        Phát hiện các thuật toán Monte Carlo có sẵn.

        Returns
        -------
        List[MonteCarloImplementation]
            Danh sách các thuật toán Monte Carlo có sẵn
        """
        available = []

        # Luôn có thuật toán chuẩn dựa trên CPU
        available.append(MonteCarloImplementation.STANDARD)

        # Kiểm tra thuật toán unified
        try:
            # Thử import thuật toán unified, nếu không có sẽ bỏ qua
            from quangtps.dose.algorithms.improvements import monte_carlo_unified

            available.append(MonteCarloImplementation.UNIFIED)
            logger.info("Phát hiện UnifiedMonteCarloAPI")
        except ImportError:
            logger.debug("Không tìm thấy UnifiedMonteCarloAPI")

        # Kiểm tra thuật toán GPU cơ bản
        try:
            # Thử import GPU module
            import numpy as np  # Cần thiết cho tất cả các thuật toán

            # Kiểm tra GPU cơ bản (không phụ thuộc vào CUDA)
            try:
                # Chỉ cần thử import, không cần thực sự sử dụng
                from quangtps.dose.algorithms.improvements import monte_carlo_gpu

                available.append(MonteCarloImplementation.GPU_BASIC)
                logger.info("Phát hiện BasicMonteCarloGPUAlgorithm")
            except ImportError:
                logger.debug("Không tìm thấy BasicMonteCarloGPUAlgorithm")

            # Kiểm tra GPU nâng cao
            try:
                import cupy
                from quangtps.dose.algorithms.improvements import (
                    monte_carlo_gpu_algorithm,
                )

                available.append(MonteCarloImplementation.GPU_ADVANCED)
                logger.info("Phát hiện AdvancedMonteCarloGPUAlgorithm")
            except ImportError:
                logger.debug("Không tìm thấy AdvancedMonteCarloGPUAlgorithm hoặc CuPy")

        except ImportError:
            logger.warning(
                "Không thể import numpy, các thuật toán GPU sẽ không khả dụng"
            )

        # Kiểm tra VMC Pro nếu có trong plugins
        try:
            from quangtps.plugins.montecarlo_dose import vmcpro

            available.append(MonteCarloImplementation.VMCPRO)
            logger.info("Phát hiện VMC Pro plugin")
        except ImportError:
            logger.debug("Không tìm thấy VMC Pro plugin")

        return available

    def _detect_hardware_capabilities(self) -> HardwareType:
        """
        Phát hiện loại phần cứng có sẵn cho tính toán.

        Returns
        -------
        HardwareType
            Loại phần cứng phát hiện được
        """
        # Mặc định là CPU đơn
        hardware_type = HardwareType.CPU

        # Kiểm tra CPU đa lõi
        cpu_count = multiprocessing.cpu_count()
        if cpu_count > 1:
            hardware_type = HardwareType.CPU_MULTI
            logger.info(f"Phát hiện CPU đa lõi: {cpu_count} lõi")
        else:
            logger.info("Phát hiện CPU đơn lõi")

        # Kiểm tra GPU - ưu tiên theo thứ tự: CUDA, OpenCL, CPU
        try:
            # Kiểm tra CuPy (CUDA)
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() > 0:
                    # Kiểm tra thiết bị NVIDIA
                    device_name = cp.cuda.runtime.getDeviceProperties(0)[
                        "name"
                    ].decode()
                    if "NVIDIA" in device_name.upper():
                        hardware_type = HardwareType.GPU_NVIDIA
                        logger.info(f"Phát hiện GPU NVIDIA: {device_name}")
                    else:
                        # GPU khác hỗ trợ CUDA
                        hardware_type = HardwareType.GPU_BASIC
                        logger.info(f"Phát hiện GPU hỗ trợ CUDA: {device_name}")

                    # Kiểm tra nhiều GPU
                    if cp.cuda.runtime.getDeviceCount() > 1:
                        hardware_type = HardwareType.GPU_MULTI
                        logger.info(
                            f"Phát hiện nhiều GPU: {cp.cuda.runtime.getDeviceCount()} GPU"
                        )
            except ImportError:
                logger.debug("CuPy không khả dụng")

                # Thử với PyCUDA
                try:
                    import pycuda.driver as drv

                    drv.init()
                    if drv.Device.count() > 0:
                        device = drv.Device(0)
                        device_name = device.name()

                        if "NVIDIA" in device_name.upper():
                            hardware_type = HardwareType.GPU_NVIDIA
                            logger.info(
                                f"Phát hiện GPU NVIDIA (qua PyCUDA): {device_name}"
                            )
                        else:
                            hardware_type = HardwareType.GPU_BASIC
                            logger.info(
                                f"Phát hiện GPU hỗ trợ CUDA (qua PyCUDA): {device_name}"
                            )

                        # Kiểm tra nhiều GPU
                        if drv.Device.count() > 1:
                            hardware_type = HardwareType.GPU_MULTI
                            logger.info(
                                f"Phát hiện nhiều GPU: {drv.Device.count()} GPU"
                            )
                except ImportError:
                    logger.debug("PyCUDA không khả dụng")

            # Nếu không có CUDA, kiểm tra OpenCL
            if hardware_type in [HardwareType.CPU, HardwareType.CPU_MULTI]:
                try:
                    import pyopencl as cl

                    platforms = cl.get_platforms()
                    if platforms:
                        for platform in platforms:
                            devices = platform.get_devices(
                                device_type=cl.device_type.GPU
                            )
                            if devices:
                                device_name = devices[0].name

                                if "AMD" in device_name.upper():
                                    hardware_type = HardwareType.GPU_AMD
                                    logger.info(
                                        f"Phát hiện GPU AMD (qua OpenCL): {device_name}"
                                    )
                                else:
                                    hardware_type = HardwareType.GPU_BASIC
                                    logger.info(
                                        f"Phát hiện GPU hỗ trợ OpenCL: {device_name}"
                                    )

                                # Kiểm tra nhiều GPU
                                if len(devices) > 1:
                                    hardware_type = HardwareType.GPU_MULTI
                                    logger.info(
                                        f"Phát hiện nhiều GPU OpenCL: {len(devices)} GPU"
                                    )
                                break
                except ImportError:
                    logger.debug("PyOpenCL không khả dụng")

        except Exception as e:
            # Nếu có bất kỳ lỗi nào khi phát hiện GPU, sử dụng CPU
            logger.warning(f"Lỗi khi phát hiện GPU: {str(e)}")
            logger.warning("Quay về sử dụng CPU")

        return hardware_type

    def create(
        self,
        implementation: Optional[MonteCarloImplementation] = None,
        accuracy_level: Optional[AccuracyLevel] = None,
        hardware_type: Optional[HardwareType] = None,
        **kwargs,
    ) -> DoseCalculationAlgorithm:
        """
        Tạo một thuật toán Monte Carlo với cấu hình phù hợp.

        Parameters
        ----------
        implementation : MonteCarloImplementation, optional
            Loại cài đặt Monte Carlo muốn sử dụng
        accuracy_level : AccuracyLevel, optional
            Mức độ chính xác mong muốn
        hardware_type : HardwareType, optional
            Loại phần cứng muốn sử dụng
        **kwargs : dict
            Các tham số tùy chọn khác

        Returns
        -------
        DoseCalculationAlgorithm
            Thuật toán Monte Carlo được cấu hình
        """
        # Sử dụng phần cứng đã phát hiện nếu không chỉ định
        if hardware_type is None:
            hardware_type = self._hardware_type

        # Mức độ chính xác mặc định là MEDIUM
        if accuracy_level is None:
            accuracy_level = AccuracyLevel.MEDIUM

        # Nếu không chỉ định cụ thể, chọn thuật toán tốt nhất dựa trên phần cứng và độ chính xác
        if implementation is None:
            implementation = self._select_implementation(accuracy_level, hardware_type)

        # Khởi tạo tham số
        params = kwargs.copy()

        # Cấu hình tham số theo mức độ chính xác
        params = self._configure_accuracy(accuracy_level, params)

        # Cấu hình tham số theo phần cứng
        params = self._configure_hardware(hardware_type, params)

        # Tạo thuật toán với tham số đã cấu hình
        algorithm = self._create_algorithm(implementation, params)

        return algorithm

    def _select_implementation(
        self, accuracy_level: AccuracyLevel, hardware_type: HardwareType
    ) -> MonteCarloImplementation:
        """
        Chọn thuật toán Monte Carlo tốt nhất dựa trên độ chính xác và phần cứng.

        Parameters
        ----------
        accuracy_level : AccuracyLevel
            Mức độ chính xác mong muốn
        hardware_type : HardwareType
            Loại phần cứng có sẵn

        Returns
        -------
        MonteCarloImplementation
            Loại thuật toán Monte Carlo được chọn
        """
        # Ưu tiên thuật toán UNIFIED nếu có sẵn
        if MonteCarloImplementation.UNIFIED in self._available_implementations:
            return MonteCarloImplementation.UNIFIED

        # Nếu cần độ chính xác cực cao, ưu tiên VMC Pro
        if (
            accuracy_level == AccuracyLevel.ULTRA
            and MonteCarloImplementation.VMCPRO in self._available_implementations
        ):
            return MonteCarloImplementation.VMCPRO

        # Với GPU NVIDIA, ưu tiên GPU_ADVANCED
        if hardware_type in [HardwareType.GPU_NVIDIA, HardwareType.GPU_MULTI]:
            if MonteCarloImplementation.GPU_ADVANCED in self._available_implementations:
                return MonteCarloImplementation.GPU_ADVANCED
            elif MonteCarloImplementation.GPU_BASIC in self._available_implementations:
                return MonteCarloImplementation.GPU_BASIC

        # Với GPU AMD hoặc cơ bản, sử dụng GPU_BASIC
        if hardware_type in [HardwareType.GPU_AMD, HardwareType.GPU_BASIC]:
            if MonteCarloImplementation.GPU_BASIC in self._available_implementations:
                return MonteCarloImplementation.GPU_BASIC

        # Mặc định sử dụng thuật toán chuẩn
        return MonteCarloImplementation.STANDARD

    def _configure_accuracy(
        self, accuracy_level: AccuracyLevel, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cấu hình các tham số dựa trên mức độ chính xác.

        Parameters
        ----------
        accuracy_level : AccuracyLevel
            Mức độ chính xác mong muốn
        params : Dict[str, Any]
            Tham số hiện tại

        Returns
        -------
        Dict[str, Any]
            Tham số được cập nhật
        """
        # Số lịch sử mô phỏng mặc định
        if "num_histories" not in params:
            if accuracy_level == AccuracyLevel.LOW:
                params["num_histories"] = 100_000  # 100K
            elif accuracy_level == AccuracyLevel.MEDIUM:
                params["num_histories"] = 1_000_000  # 1M
            elif accuracy_level == AccuracyLevel.HIGH:
                params["num_histories"] = 10_000_000  # 10M
            elif accuracy_level == AccuracyLevel.ULTRA:
                params["num_histories"] = 100_000_000  # 100M

        # Độ phân giải lưới liều
        if "dose_grid_resolution" not in params:
            if accuracy_level == AccuracyLevel.LOW:
                params["dose_grid_resolution"] = [5, 5, 5]  # mm
            elif accuracy_level == AccuracyLevel.MEDIUM:
                params["dose_grid_resolution"] = [3, 3, 3]  # mm
            elif accuracy_level == AccuracyLevel.HIGH:
                params["dose_grid_resolution"] = [2, 2, 2]  # mm
            elif accuracy_level == AccuracyLevel.ULTRA:
                params["dose_grid_resolution"] = [1, 1, 1]  # mm

        # Ngưỡng năng lượng cắt electron
        if "electron_cutoff" not in params:
            if accuracy_level == AccuracyLevel.LOW:
                params["electron_cutoff"] = 0.1  # MeV
            elif accuracy_level == AccuracyLevel.MEDIUM:
                params["electron_cutoff"] = 0.05  # MeV
            elif accuracy_level == AccuracyLevel.HIGH:
                params["electron_cutoff"] = 0.025  # MeV
            elif accuracy_level == AccuracyLevel.ULTRA:
                params["electron_cutoff"] = 0.01  # MeV

        if "photon_cutoff" not in params:
            if accuracy_level == AccuracyLevel.LOW:
                params["photon_cutoff"] = 0.05  # MeV
            elif accuracy_level == AccuracyLevel.MEDIUM:
                params["photon_cutoff"] = 0.01  # MeV
            elif accuracy_level == AccuracyLevel.HIGH:
                params["photon_cutoff"] = 0.005  # MeV
            elif accuracy_level == AccuracyLevel.ULTRA:
                params["photon_cutoff"] = 0.001  # MeV

        return params

    def _configure_hardware(
        self, hardware_type: HardwareType, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cấu hình các tham số dựa trên phần cứng có sẵn.

        Parameters
        ----------
        hardware_type : HardwareType
            Loại phần cứng có sẵn
        params : Dict[str, Any]
            Tham số hiện tại

        Returns
        -------
        Dict[str, Any]
            Tham số được cập nhật
        """
        # Cấu hình sử dụng GPU
        if "use_gpu" not in params:
            params["use_gpu"] = hardware_type in [
                HardwareType.GPU_BASIC,
                HardwareType.GPU_NVIDIA,
                HardwareType.GPU_AMD,
                HardwareType.GPU_MULTI,
            ]

        # Cấu hình số lượng threads/workers
        if "num_workers" not in params:
            if hardware_type == HardwareType.CPU:
                params["num_workers"] = 1
            elif hardware_type == HardwareType.CPU_MULTI:
                params["num_workers"] = max(1, multiprocessing.cpu_count() - 1)
            elif hardware_type in [
                HardwareType.GPU_BASIC,
                HardwareType.GPU_NVIDIA,
                HardwareType.GPU_AMD,
            ]:
                params["num_workers"] = 1  # Mặc định 1 GPU
            elif hardware_type == HardwareType.GPU_MULTI:
                try:
                    # Thử xác định số lượng GPU
                    num_gpus = 1  # Mặc định

                    try:
                        import cupy as cp

                        num_gpus = cp.cuda.runtime.getDeviceCount()
                    except ImportError:
                        try:
                            import pycuda.driver as drv

                            drv.init()
                            num_gpus = drv.Device.count()
                        except ImportError:
                            try:
                                import pyopencl as cl

                                platforms = cl.get_platforms()
                                if platforms:
                                    for platform in platforms:
                                        devices = platform.get_devices(
                                            device_type=cl.device_type.GPU
                                        )
                                        num_gpus = max(num_gpus, len(devices))
                            except ImportError:
                                # Không tìm thấy thư viện GPU nào
                                num_gpus = 1

                    params["num_workers"] = num_gpus
                except Exception as e:
                    logger.warning(f"Lỗi khi xác định số lượng GPU: {str(e)}")
                    params["num_workers"] = 1

        # Cấu hình danh sách thiết bị
        if "device_ids" not in params and hardware_type == HardwareType.GPU_MULTI:
            try:
                # Cố gắng lấy danh sách thiết bị
                device_ids = list(range(params["num_workers"]))
                params["device_ids"] = device_ids
            except Exception as e:
                logger.warning(f"Lỗi khi cấu hình danh sách thiết bị: {str(e)}")

        return params

    def _create_algorithm(
        self, implementation: MonteCarloImplementation, params: Dict[str, Any]
    ) -> DoseCalculationAlgorithm:
        """
        Tạo và khởi tạo thuật toán Monte Carlo.

        Parameters
        ----------
        implementation : MonteCarloImplementation
            Loại thuật toán Monte Carlo
        params : Dict[str, Any]
            Các tham số cấu hình

        Returns
        -------
        DoseCalculationAlgorithm
            Thuật toán Monte Carlo được khởi tạo
        """
        algorithm = None

        try:
            if implementation == MonteCarloImplementation.STANDARD:
                # Tạo thuật toán Monte Carlo tiêu chuẩn
                try:
                    # Thử import từ cả hai vị trí có thể có
                    try:
                        from quangtps.dose.algorithms.monte_carlo import (
                            MonteCarloAlgorithm,
                        )
                    except ImportError:
                        from quangtps.dose.algorithms.monte_carlo_algorithm import (
                            MonteCarloAlgorithm,
                        )

                    algorithm = MonteCarloAlgorithm()
                    logger.info("Đã tạo thuật toán Monte Carlo tiêu chuẩn")
                except ImportError as e:
                    logger.error(
                        f"Không thể tạo thuật toán Monte Carlo tiêu chuẩn: {str(e)}"
                    )
                    # Tạo thuật toán giả
                    algorithm = DoseCalculationAlgorithm()

            elif implementation == MonteCarloImplementation.GPU_BASIC:
                # Tạo thuật toán Monte Carlo GPU cơ bản
                try:
                    from quangtps.dose.algorithms.improvements.monte_carlo_gpu import (
                        BasicMonteCarloGPUAlgorithm,
                    )

                    algorithm = BasicMonteCarloGPUAlgorithm()
                    logger.info("Đã tạo thuật toán Monte Carlo GPU cơ bản")
                except ImportError as e:
                    logger.error(
                        f"Không thể tạo thuật toán Monte Carlo GPU cơ bản: {str(e)}"
                    )
                    # Thử thuật toán thay thế
                    return self._create_algorithm(
                        MonteCarloImplementation.STANDARD, params
                    )

            elif implementation == MonteCarloImplementation.GPU_ADVANCED:
                # Tạo thuật toán Monte Carlo GPU nâng cao
                try:
                    # Thử import từ cả hai vị trí có thể có
                    try:
                        from quangtps.dose.algorithms.improvements.monte_carlo_gpu_algorithm import (
                            AdvancedMonteCarloGPUAlgorithm,
                        )
                    except ImportError:
                        from quangtps.dose.algorithms.improvements.monte_carlo_gpu import (
                            AdvancedMonteCarloGPUAlgorithm,
                        )

                    algorithm = AdvancedMonteCarloGPUAlgorithm()
                    logger.info("Đã tạo thuật toán Monte Carlo GPU nâng cao")
                except ImportError as e:
                    logger.error(
                        f"Không thể tạo thuật toán Monte Carlo GPU nâng cao: {str(e)}"
                    )
                    # Thử thuật toán thay thế
                    return self._create_algorithm(
                        MonteCarloImplementation.GPU_BASIC, params
                    )

            elif implementation == MonteCarloImplementation.UNIFIED:
                # Tạo thuật toán Monte Carlo thống nhất
                try:
                    from quangtps.dose.algorithms.improvements.monte_carlo_unified import (
                        UnifiedMonteCarloAlgorithm,
                    )

                    algorithm = UnifiedMonteCarloAlgorithm()
                    logger.info("Đã tạo thuật toán Monte Carlo thống nhất")
                except ImportError as e:
                    logger.error(
                        f"Không thể tạo thuật toán Monte Carlo thống nhất: {str(e)}"
                    )
                    # Thử thuật toán thay thế
                    return self._create_algorithm(
                        MonteCarloImplementation.GPU_ADVANCED, params
                    )

            elif implementation == MonteCarloImplementation.VMCPRO:
                # Tạo thuật toán VMC Pro
                try:
                    from quangtps.plugins.montecarlo_dose.vmcpro import VMCProAlgorithm

                    algorithm = VMCProAlgorithm()
                    logger.info("Đã tạo thuật toán VMC Pro")
                except ImportError as e:
                    logger.error(f"Không thể tạo thuật toán VMC Pro: {str(e)}")
                    # Thử thuật toán thay thế
                    return self._create_algorithm(
                        MonteCarloImplementation.UNIFIED, params
                    )

            else:
                # Không hỗ trợ hoặc không nhận ra
                logger.warning(
                    f"Không nhận ra loại thuật toán {implementation}, sử dụng thuật toán tiêu chuẩn"
                )
                return self._create_algorithm(MonteCarloImplementation.STANDARD, params)

            # Cấu hình thuật toán với các tham số
            if algorithm is not None:
                # Áp dụng các tham số
                for key, value in params.items():
                    # Kiểm tra xem thuật toán có thuộc tính này không
                    if hasattr(algorithm, key):
                        setattr(algorithm, key, value)
                    # Nếu có phương thức setter tương ứng
                    elif hasattr(algorithm, f"set_{key}"):
                        getattr(algorithm, f"set_{key}")(value)

                logger.info(
                    f"Đã cấu hình thuật toán {implementation} với các tham số: {params}"
                )

            return algorithm

        except Exception as e:
            logger.error(
                f"Lỗi khi khởi tạo thuật toán Monte Carlo {implementation}: {str(e)}"
            )
            # Ghi log traceback để dễ dàng gỡ lỗi
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")

            # Sử dụng DoseCalculationAlgorithm giả trong trường hợp lỗi
            return DoseCalculationAlgorithm()

    def get_available_implementations(self) -> List[MonteCarloImplementation]:
        """
        Lấy danh sách các thuật toán Monte Carlo có sẵn.

        Returns
        -------
        List[MonteCarloImplementation]
            Danh sách các thuật toán Monte Carlo có sẵn
        """
        return self._available_implementations

    def get_recommended_implementation(self) -> MonteCarloImplementation:
        """
        Lấy thuật toán Monte Carlo được đề xuất cho hệ thống hiện tại.

        Returns
        -------
        MonteCarloImplementation
            Thuật toán Monte Carlo được đề xuất
        """
        return self._recommended_implementation

    def get_hardware_capabilities(self) -> HardwareType:
        """
        Lấy thông tin về phần cứng được phát hiện.

        Returns
        -------
        HardwareType
            Loại phần cứng đã phát hiện
        """
        return self._hardware_type


def create_monte_carlo_algorithm(**kwargs) -> DoseCalculationAlgorithm:
    """
    Hàm tiện ích để tạo thuật toán Monte Carlo với các tham số mặc định.

    Parameters
    ----------
    **kwargs : dict
        Các tham số tùy chọn

    Returns
    -------
    DoseCalculationAlgorithm
        Thuật toán Monte Carlo được khởi tạo
    """
    factory = MonteCarloFactory()
    return factory.create(**kwargs)
