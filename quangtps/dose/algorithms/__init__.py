#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module chứa các thuật toán tính liều.

Module này cung cấp các thuật toán tính phân bố liều cho xạ trị,
bao gồm nhiều phương pháp tính như Collapsed Cone, Pencil Beam,
và Monte Carlo.
"""

import logging
import importlib
from typing import Dict, List, Any, Optional, Type, Set, Union

logger = logging.getLogger(__name__)

# Danh sách các thuật toán tính liều cơ bản đã biết
ALGORITHMS = [
    "collapsed_cone",
    "pencil_beam",
    "monte_carlo",
    "fast_superposition",
    "monte_carlo_gpu",  # Thêm Monte Carlo GPU
]

# Dictionary ánh xạ tên hiển thị tới tên module
ALGORITHM_NAME_MAPPING = {
    "Collapsed Cone": "collapsed_cone",
    "Pencil Beam": "pencil_beam",
    "Monte Carlo": "monte_carlo",
    "Fast Superposition": "fast_superposition",
    "Monte Carlo (GPU)": "monte_carlo_gpu",  # Thêm Monte Carlo GPU
}


# Classes for algorithm and result management
class DoseCalculationAlgorithm:
    """Lớp cơ sở cho các thuật toán tính liều."""

    def __init__(self) -> None:
        """
        Khởi tạo thuật toán tính liều.
        """
        pass

    def initialize(self, patient_data: Any) -> bool:
        """
        Khởi tạo thuật toán với dữ liệu bệnh nhân.

        Parameters
        ----------
        patient_data : Any
            Dữ liệu bệnh nhân, bao gồm hình ảnh CT và các cấu trúc

        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu thất bại
        """
        raise NotImplementedError("Phương thức initialize phải được ghi đè")

    def calculate_dose(self, beam_arrangement: Any) -> Any:
        """
        Tính phân bố liều dựa trên bố trí chùm tia cho bệnh nhân.

        Parameters
        ----------
        beam_arrangement : Any
            Bố trí chùm tia và thông số kỹ thuật

        Returns
        -------
        Any
            Kết quả tính toán liều
        """
        raise NotImplementedError("Phương thức calculate_dose phải được ghi đè")

    def get_algorithm_type(self) -> str:
        """
        Trả về loại thuật toán.

        Returns
        -------
        str
            Định danh của thuật toán
        """
        raise NotImplementedError("Phương thức get_algorithm_type phải được ghi đè")

    def get_display_name(self) -> str:
        """
        Trả về tên hiển thị của thuật toán.

        Returns
        -------
        str
            Tên thuật toán để hiển thị trong giao diện người dùng
        """
        raise NotImplementedError("Phương thức get_display_name phải được ghi đè")

    def get_description(self) -> str:
        """
        Trả về mô tả của thuật toán.

        Returns
        -------
        str
            Mô tả chi tiết của thuật toán
        """
        raise NotImplementedError("Phương thức get_description phải được ghi đè")


class DoseCalculationResult:
    """Lớp cơ sở cho kết quả tính toán liều."""

    def __init__(self) -> None:
        """
        Khởi tạo đối tượng kết quả tính toán liều.
        """
        self.success = False
        self.error_message = ""
        self.execution_time = 0.0  # seconds

    def is_valid(self) -> bool:
        """
        Kiểm tra xem kết quả tính toán liều có hợp lệ không.

        Returns
        -------
        bool
            True nếu kết quả hợp lệ, False nếu không
        """
        return self.success

    def get_dose_grid(self) -> Any:
        """
        Trả về lưới phân bố liều.

        Returns
        -------
        Any
            Lưới phân bố liều 3D
        """
        raise NotImplementedError("Phương thức get_dose_grid phải được ghi đè")


# Import các class thuật toán
_algorithm_classes: Dict[str, Type[DoseCalculationAlgorithm]] = {}
_available_algorithms: Set[str] = set()


def _try_import_algorithm(algorithm_name: str) -> bool:
    """
    Thử import module thuật toán tính liều.

    Parameters
    ----------
    algorithm_name : str
        Tên của thuật toán để import

    Returns
    -------
    bool
        True nếu import thành công, False nếu thất bại
    """
    try:
        # Thử import module
        module_name = f"quangtps.dose.algorithms.{algorithm_name}"
        module = importlib.import_module(module_name)

        # Lấy tên class từ tên module
        # Ví dụ: collapsed_cone -> CollapsedConeAlgorithm
        class_name_parts = [part.capitalize() for part in algorithm_name.split("_")]
        algorithm_class_name = "".join(class_name_parts) + "Algorithm"

        if hasattr(module, algorithm_class_name):
            _algorithm_classes[algorithm_name] = getattr(module, algorithm_class_name)
            _available_algorithms.add(algorithm_name)
            logger.info(f"Đã tải thuật toán {algorithm_name} thành công.")
            return True
        else:
            logger.warning(
                f"Module {module_name} không chứa class {algorithm_class_name}."
            )
            return False
    except ImportError as e:
        # Thử import từ thư mục improvements
        try:
            module_name = f"quangtps.dose.algorithms.improvements.{algorithm_name}"
            module = importlib.import_module(module_name)

            # Tìm tên lớp theo quy tắc camelCase với hậu tố Algorithm
            class_name_parts = [part.capitalize() for part in algorithm_name.split("_")]
            algorithm_class_name = "".join(class_name_parts) + "Algorithm"

            if hasattr(module, algorithm_class_name):
                _algorithm_classes[algorithm_name] = getattr(
                    module, algorithm_class_name
                )
                _available_algorithms.add(algorithm_name)
                logger.info(f"Đã tải thuật toán cải tiến {algorithm_name} thành công.")
                return True
            else:
                logger.warning(
                    f"Module cải tiến {module_name} không chứa class {algorithm_class_name}."
                )
                return False
        except ImportError as inner_e:
            logger.warning(
                f"Không thể import thuật toán {algorithm_name}: {str(e)}. "
                f"Vị trí cải tiến cũng không khả dụng: {str(inner_e)}"
            )
            return False
        except Exception as inner_e:
            logger.error(
                f"Lỗi khi import thuật toán cải tiến {algorithm_name}: {str(inner_e)}"
            )
            return False
    except Exception as e:
        logger.error(f"Lỗi khi import thuật toán {algorithm_name}: {str(e)}")
        return False


# Import tất cả các thuật toán đã biết
for algorithm in ALGORITHMS:
    _try_import_algorithm(algorithm)


def get_available_algorithms() -> List[str]:
    """
    Trả về danh sách các thuật toán tính liều khả dụng.

    Returns
    -------
    List[str]
        Danh sách các thuật toán khả dụng
    """
    return list(_available_algorithms)


def get_algorithm_display_names() -> List[str]:
    """
    Trả về danh sách tên hiển thị của các thuật toán khả dụng.

    Returns
    -------
    List[str]
        Danh sách tên hiển thị
    """
    display_names = []
    reverse_mapping = {v: k for k, v in ALGORITHM_NAME_MAPPING.items()}

    for algo in _available_algorithms:
        if algo in reverse_mapping:
            display_names.append(reverse_mapping[algo])
        else:
            # Tạo tên hiển thị từ tên module
            display_name = " ".join(word.capitalize() for word in algo.split("_"))
            display_names.append(display_name)

    return display_names


def create_algorithm(algorithm_name: str) -> Optional[DoseCalculationAlgorithm]:
    """
    Tạo đối tượng thuật toán tính liều.

    Parameters
    ----------
    algorithm_name : str
        Tên thuật toán cần tạo đối tượng

    Returns
    -------
    Optional[DoseCalculationAlgorithm]
        Đối tượng thuật toán tính liều hoặc None nếu không tìm thấy
    """
    # Kiểm tra xem tên được cung cấp có phải là tên hiển thị không
    if algorithm_name in ALGORITHM_NAME_MAPPING:
        algorithm_name = ALGORITHM_NAME_MAPPING[algorithm_name]

    # Đảm bảo rằng thuật toán đã được import
    if algorithm_name not in _algorithm_classes:
        success = _try_import_algorithm(algorithm_name)
        if not success:
            logger.error(
                f"Không thể tạo thuật toán {algorithm_name}: thuật toán không khả dụng"
            )
            return None

    # Tạo đối tượng thuật toán
    try:
        algorithm_class = _algorithm_classes[algorithm_name]
        return algorithm_class()
    except Exception as e:
        logger.error(f"Lỗi khi tạo thuật toán {algorithm_name}: {str(e)}")
        return None


def register_algorithm(
    algorithm_name: str, algorithm_class: Type[DoseCalculationAlgorithm]
) -> bool:
    """
    Đăng ký thuật toán tính liều mới.

    Parameters
    ----------
    algorithm_name : str
        Tên của thuật toán
    algorithm_class : Type[DoseCalculationAlgorithm]
        Lớp thuật toán kế thừa từ DoseCalculationAlgorithm

    Returns
    -------
    bool
        True nếu đăng ký thành công, False nếu thất bại
    """
    if algorithm_name in _algorithm_classes:
        logger.warning(f"Thuật toán {algorithm_name} đã được đăng ký trước đó")
        return False

    _algorithm_classes[algorithm_name] = algorithm_class
    _available_algorithms.add(algorithm_name)

    # Thêm vào danh sách thuật toán đã biết nếu chưa có
    if algorithm_name not in ALGORITHMS:
        ALGORITHMS.append(algorithm_name)

    # Thêm tên hiển thị nếu được cung cấp
    instance = None
    try:
        instance = algorithm_class()
        display_name = instance.get_display_name()
        if display_name and display_name not in ALGORITHM_NAME_MAPPING:
            ALGORITHM_NAME_MAPPING[display_name] = algorithm_name
    except Exception as e:
        logger.warning(
            f"Không thể lấy tên hiển thị của thuật toán {algorithm_name}: {str(e)}"
        )

    logger.info(f"Đã đăng ký thuật toán {algorithm_name} thành công")
    return True


def get_best_available_algorithm() -> Optional[DoseCalculationAlgorithm]:
    """
    Trả về thuật toán tính liều tốt nhất hiện có dựa trên phần cứng có sẵn.

    Thứ tự ưu tiên:
    1. Monte Carlo GPU nếu có GPU với CUDA
    2. Collapsed Cone nếu có nhiều lõi CPU
    3. Pencil Beam cho các trường hợp khác

    Returns
    -------
    Optional[DoseCalculationAlgorithm]
        Thuật toán tốt nhất hiện có, hoặc None nếu không có thuật toán nào
    """
    # Kiểm tra GPU với CUDA
    has_gpu = False
    gpu_info = None

    try:
        # Thử phát hiện GPU qua CuPy
        import cupy as cp

        num_gpus = cp.cuda.runtime.getDeviceCount()
        if num_gpus > 0:
            device = cp.cuda.Device(0)
            mem_info = device.mem_info
            gpu_name = device.attributes.get("name", b"Unknown").decode("utf-8")
            free_memory = mem_info[0] / (1024**3)  # GB
            total_memory = mem_info[1] / (1024**3)  # GB

            has_gpu = True
            gpu_info = {
                "name": gpu_name,
                "count": num_gpus,
                "free_memory": free_memory,
                "total_memory": total_memory,
                "detected_by": "cupy",
            }
    except:
        try:
            # Thử phát hiện GPU qua PyCUDA
            import pycuda.driver as cuda
            import pycuda.autoinit

            num_gpus = cuda.Device.count()
            if num_gpus > 0:
                device = cuda.Device(0)
                gpu_name = device.name()
                total_memory = device.total_memory() / (1024**3)  # GB
                free_memory = (device.total_memory() - device.used_memory()) / (
                    1024**3
                )  # GB

                has_gpu = True
                gpu_info = {
                    "name": gpu_name,
                    "count": num_gpus,
                    "free_memory": free_memory,
                    "total_memory": total_memory,
                    "detected_by": "pycuda",
                }
        except:
            # Không tìm thấy GPU với CUDA
            pass

    # Kiểm tra số lõi CPU
    import multiprocessing

    num_cores = multiprocessing.cpu_count()

    # Log thông tin phần cứng phát hiện được
    logger = logging.getLogger(__name__)

    if has_gpu:
        logger.info(
            f"Phát hiện {gpu_info['count']} GPU: {gpu_info['name']} với {gpu_info['free_memory']:.2f}GB/{gpu_info['total_memory']:.2f}GB bộ nhớ (qua {gpu_info['detected_by']})"
        )
    else:
        logger.info(f"Không phát hiện GPU. Sẽ sử dụng tính toán CPU ({num_cores} lõi).")

    # Chọn thuật toán tốt nhất dựa trên phần cứng
    available_algorithms = get_available_algorithms()

    # Ưu tiên Monte Carlo GPU nếu có GPU
    if has_gpu and "monte_carlo_gpu" in available_algorithms:
        logger.info("Chọn thuật toán Monte Carlo GPU (nhanh nhất)")
        return create_algorithm("monte_carlo_gpu")

    # Ưu tiên Collapsed Cone nếu có nhiều lõi CPU
    if num_cores >= 4 and "collapsed_cone" in available_algorithms:
        logger.info(f"Chọn thuật toán Collapsed Cone (tối ưu cho {num_cores} lõi CPU)")
        return create_algorithm("collapsed_cone")

    # Sử dụng Pencil Beam cho các trường hợp khác
    if "pencil_beam" in available_algorithms:
        logger.info("Chọn thuật toán Pencil Beam (thuật toán cơ bản)")
        return create_algorithm("pencil_beam")

    # Sử dụng bất kỳ thuật toán nào có sẵn
    if available_algorithms:
        algo_name = available_algorithms[0]
        logger.info(f"Chọn thuật toán {algo_name} (thuật toán mặc định)")
        return create_algorithm(algo_name)

    logger.error("Không tìm thấy thuật toán tính liều nào. Kiểm tra cài đặt.")
    return None


# Thử import thuật toán Monte Carlo GPU từ thư mục improvements
def _import_monte_carlo_gpu():
    """
    Thử import thuật toán Monte Carlo GPU từ thư mục improvements.
    """
    try:
        # Import Monte Carlo GPU từ improvements
        from quangtps.dose.algorithms.improvements.monte_carlo_gpu import (
            MonteCarloGPUAlgorithm,
        )

        # Đăng ký thuật toán
        register_algorithm("monte_carlo_gpu", MonteCarloGPUAlgorithm)
        logger.info("Đã đăng ký thuật toán Monte Carlo GPU thành công")

        # Thêm tên hiển thị
        ALGORITHM_NAME_MAPPING["Monte Carlo (GPU)"] = "monte_carlo_gpu"

    except ImportError as e:
        logger.warning(f"Thuật toán Monte Carlo GPU không khả dụng: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi khi tải thuật toán Monte Carlo GPU: {str(e)}")


# Import Monte Carlo GPU khi khởi tạo module
_import_monte_carlo_gpu()


__all__ = [
    "DoseCalculationAlgorithm",
    "DoseCalculationResult",
    "get_available_algorithms",
    "get_algorithm_display_names",
    "create_algorithm",
    "register_algorithm",
    "get_best_available_algorithm",
]

# Version of the algorithms module
__version__ = "0.7.8"
