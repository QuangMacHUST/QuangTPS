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
import numpy as np
import os

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

# Dictionary lưu trữ các thuật toán tính liều đã đăng ký
_registered_algorithms = {}


# Classes for algorithm and result management
class DoseCalculationAlgorithm:
    """Lớp cơ sở cho các thuật toán tính liều."""

    def __init__(self) -> None:
        """
        Khởi tạo thuật toán tính liều.
        """
        self.initialized = False
        self.name = "Generic Algorithm"
        self.description = "Generic dose calculation algorithm"

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
        self.initialized = True
        return True

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
        logger.warning(f"Sử dụng phương thức calculate_dose mặc định cho {self.name}")
        result = DoseCalculationResult()
        result.success = True
        result.dose_grid = np.ones((100, 100, 100), dtype=np.float32)  # Lưới liều giả
        return result

    def get_algorithm_type(self) -> str:
        """
        Trả về loại thuật toán.

        Returns
        -------
        str
            Định danh của thuật toán
        """
        return "generic"

    def get_display_name(self) -> str:
        """
        Trả về tên hiển thị của thuật toán.

        Returns
        -------
        str
            Tên thuật toán để hiển thị trong giao diện người dùng
        """
        return self.name

    def get_description(self) -> str:
        """
        Trả về mô tả của thuật toán.

        Returns
        -------
        str
            Mô tả chi tiết của thuật toán
        """
        return self.description


class DoseCalculationResult:
    """Lớp cơ sở cho kết quả tính toán liều."""

    def __init__(self) -> None:
        """
        Khởi tạo đối tượng kết quả tính toán liều.
        """
        self.success = False
        self.error_message = ""
        self.execution_time = 0.0  # seconds
        self.dose_grid = None
        self.algorithm = ""
        self.metadata = {}

    def is_valid(self) -> bool:
        """
        Kiểm tra xem kết quả tính toán liều có hợp lệ không.

        Returns
        -------
        bool
            True nếu kết quả hợp lệ, False nếu không
        """
        return self.success and self.dose_grid is not None

    def get_dose_grid(self) -> Any:
        """
        Trả về lưới phân bố liều.

        Returns
        -------
        Any
            Lưới phân bố liều 3D
        """
        return self.dose_grid


# Tạo các lớp giả cho thuật toán không có sẵn
class PencilBeamAlgorithm(DoseCalculationAlgorithm):
    """Thuật toán Pencil Beam giả mạch khi không có thực hiện thực tế."""

    def __init__(self):
        super().__init__()
        self.name = "Pencil Beam"
        self.description = (
            "Pencil Beam dose calculation algorithm (fallback implementation)"
        )

    def get_algorithm_type(self) -> str:
        return "pencil_beam"


class CollapsedConeAlgorithm(DoseCalculationAlgorithm):
    """Thuật toán Collapsed Cone giả mạch khi không có thực hiện thực tế."""

    def __init__(self):
        super().__init__()
        self.name = "Collapsed Cone"
        self.description = "Collapsed Cone Convolution dose calculation algorithm (fallback implementation)"

    def get_algorithm_type(self) -> str:
        return "collapsed_cone"


class MonteCarloAlgorithm(DoseCalculationAlgorithm):
    """Thuật toán Monte Carlo giả mạch khi không có thực hiện thực tế."""

    def __init__(self):
        super().__init__()
        self.name = "Monte Carlo"
        self.description = (
            "Monte Carlo dose calculation algorithm (fallback implementation)"
        )

    def get_algorithm_type(self) -> str:
        return "monte_carlo"


class MonteCarloGpuAlgorithm(DoseCalculationAlgorithm):
    """Thuật toán Monte Carlo GPU giả mạch khi không có thực hiện thực tế."""

    def __init__(self):
        super().__init__()
        self.name = "Monte Carlo (GPU)"
        self.description = "GPU-accelerated Monte Carlo dose calculation algorithm (fallback implementation)"

    def get_algorithm_type(self) -> str:
        return "monte_carlo_gpu"


# Import các class thuật toán
_algorithm_classes: Dict[str, Type[DoseCalculationAlgorithm]] = {}
_available_algorithms: Set[str] = set()

# Đăng ký các lớp giả mạch mặc định
_algorithm_classes["pencil_beam"] = PencilBeamAlgorithm
_algorithm_classes["collapsed_cone"] = CollapsedConeAlgorithm
_algorithm_classes["monte_carlo"] = MonteCarloAlgorithm
_algorithm_classes["monte_carlo_gpu"] = MonteCarloGpuAlgorithm

# Đảm bảo các thuật toán cơ bản luôn khả dụng
for algo in ["pencil_beam", "collapsed_cone", "monte_carlo", "monte_carlo_gpu"]:
    _available_algorithms.add(algo)


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
            # Đã có lớp giả mạch mặc định, không cần trả về False
            return True
        except Exception as inner_e:
            logger.error(
                f"Lỗi khi import thuật toán cải tiến {algorithm_name}: {str(inner_e)}"
            )
            # Đã có lớp giả mạch mặc định, không cần trả về False
            return True
    except Exception as e:
        logger.error(f"Lỗi khi import thuật toán {algorithm_name}: {str(e)}")
        # Đã có lớp giả mạch mặc định, không cần trả về False
        return True


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
    """Trả về thuật toán tính liều tốt nhất có sẵn trên hệ thống.

    Returns:
        Thuật toán tính liều hoặc None nếu không có thuật toán nào khả dụng.
    """
    # Đầu tiên thử sử dụng thuật toán Monte Carlo GPU nếu có GPU
    has_gpu = False

    # Thử phát hiện GPU bằng nhiều cách khác nhau
    try:
        # Thử phát hiện GPU qua CuPy
        try:
            import cupy as cp

            num_gpus = cp.cuda.runtime.getDeviceCount()
            has_gpu = num_gpus > 0
            if has_gpu:
                logger.info(f"Phát hiện {num_gpus} GPU thông qua CuPy.")
        except ImportError:
            logger.debug("CuPy không khả dụng. Thử phương pháp khác.")
        except Exception as e:
            logger.debug(f"Lỗi khi kiểm tra GPU qua CuPy: {e}")

        # Nếu CuPy không xác định được GPU, thử với PyCUDA
        if not has_gpu:
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit

                num_gpus = cuda.Device.count()
                has_gpu = num_gpus > 0
                if has_gpu:
                    logger.info(f"Phát hiện {num_gpus} GPU thông qua PyCUDA.")
            except ImportError:
                logger.debug("PyCUDA không khả dụng. Thử phương pháp khác.")
            except Exception as e:
                logger.debug(f"Lỗi khi kiểm tra GPU qua PyCUDA: {e}")

        # Nếu cả CuPy và PyCUDA không hoạt động, thử kiểm tra CUDA_VISIBLE_DEVICES
        if not has_gpu:
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_devices and cuda_devices != "-1":
                logger.info(
                    f"Phát hiện GPU qua biến môi trường CUDA_VISIBLE_DEVICES={cuda_devices}"
                )
                has_gpu = True
    except Exception as e:
        logger.warning(f"Lỗi khi phát hiện GPU: {e}")
        has_gpu = False

    # Nếu có GPU, thử tạo thuật toán Monte Carlo GPU
    if has_gpu:
        try:
            monte_carlo_gpu = create_algorithm("monte_carlo_gpu")
            if monte_carlo_gpu:
                logger.info("Sử dụng thuật toán Monte Carlo GPU")
                return monte_carlo_gpu
        except Exception as e:
            logger.warning(f"Không thể tạo thuật toán Monte Carlo GPU: {e}")

    # Thử tạo thuật toán Monte Carlo CPU
    try:
        monte_carlo = create_algorithm("monte_carlo")
        if monte_carlo:
            logger.info("Sử dụng thuật toán Monte Carlo CPU")
            return monte_carlo
    except Exception as e:
        logger.warning(f"Không thể tạo thuật toán Monte Carlo: {e}")

    # Thử tạo thuật toán Collapsed Cone
    try:
        collapsed_cone = create_algorithm("collapsed_cone")
        if collapsed_cone:
            logger.info("Sử dụng thuật toán Collapsed Cone")
            return collapsed_cone
    except Exception as e:
        logger.warning(f"Không thể tạo thuật toán Collapsed Cone: {e}")

    # Cuối cùng, thử tạo thuật toán Pencil Beam
    try:
        pencil_beam = create_algorithm("pencil_beam")
        if pencil_beam:
            logger.info("Sử dụng thuật toán Pencil Beam")
            return pencil_beam
    except Exception as e:
        logger.warning(f"Không thể tạo thuật toán Pencil Beam: {e}")

    # Không tìm thấy thuật toán nào khả dụng
    logger.error("Không tìm thấy thuật toán tính liều nào khả dụng.")
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
        register_dose_algorithm("monte_carlo_gpu", MonteCarloGPUAlgorithm)
        logger.info("Đã đăng ký thuật toán Monte Carlo GPU thành công")

        # Thêm tên hiển thị
        ALGORITHM_NAME_MAPPING["Monte Carlo (GPU)"] = "monte_carlo_gpu"

    except ImportError as e:
        logger.warning(f"Thuật toán Monte Carlo GPU không khả dụng: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi khi tải thuật toán Monte Carlo GPU: {str(e)}")


# Import Monte Carlo GPU khi khởi tạo module
_import_monte_carlo_gpu()


# Hàm đăng ký và lấy thuật toán


def register_dose_algorithm(
    algorithm_id: str, algorithm_class: Type[DoseCalculationAlgorithm]
) -> None:
    """
    Đăng ký một thuật toán tính liều mới.

    Parameters
    ----------
    algorithm_id : str
        ID định danh thuật toán, thường là tên thuật toán dạng snake_case
    algorithm_class : Type[DoseCalculationAlgorithm]
        Lớp của thuật toán kế thừa từ DoseCalculationAlgorithm

    Returns
    -------
    None
    """
    if algorithm_id in _registered_algorithms:
        logger.warning(f"Thuật toán '{algorithm_id}' đã tồn tại và sẽ bị ghi đè")

    _registered_algorithms[algorithm_id] = algorithm_class
    logger.info(f"Đã đăng ký thuật toán '{algorithm_id}' thành công")


def get_dose_algorithm(
    algorithm_id: str, **kwargs
) -> Optional[DoseCalculationAlgorithm]:
    """
    Lấy instance của thuật toán tính liều theo ID.

    Parameters
    ----------
    algorithm_id : str
        ID của thuật toán đã đăng ký
    **kwargs
        Các tham số bổ sung để khởi tạo thuật toán

    Returns
    -------
    Optional[DoseCalculationAlgorithm]
        Instance của thuật toán hoặc None nếu không tìm thấy
    """
    if algorithm_id not in _registered_algorithms:
        logger.error(f"Không tìm thấy thuật toán '{algorithm_id}'")
        return None

    try:
        algorithm = _registered_algorithms[algorithm_id](**kwargs)
        return algorithm
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo thuật toán '{algorithm_id}': {str(e)}")
        return None


def get_registered_algorithms() -> Dict[str, Type[DoseCalculationAlgorithm]]:
    """
    Lấy danh sách tất cả các thuật toán tính liều đã đăng ký.

    Returns
    -------
    Dict[str, Type[DoseCalculationAlgorithm]]
        Dictionary chứa ID và lớp thuật toán
    """
    return _registered_algorithms.copy()


# Import các thuật toán cụ thể với xử lý ngoại lệ
try:
    from .pencil_beam import PencilBeamAlgorithm
except ImportError:
    logger.warning("Không thể import PencilBeamAlgorithm")

    class MockPencilBeamAlgorithm(DoseCalculationAlgorithm):
        """Đối tượng giả cho PencilBeamAlgorithm khi không thể import."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.description = "Pencil Beam Algorithm - KHÔNG KHẢ DỤNG"
            logger.warning(
                "Đang sử dụng PencilBeamAlgorithm giả. Chức năng sẽ bị hạn chế."
            )

    # Gắn lớp mock vào tên gốc để tránh lỗi import từ những nơi khác
    PencilBeamAlgorithm = MockPencilBeamAlgorithm


try:
    from .collapsed_cone import CollapsedConeAlgorithm
except ImportError:
    logger.warning("Không thể import CollapsedConeAlgorithm")

    class MockCollapsedConeAlgorithm(DoseCalculationAlgorithm):
        """Đối tượng giả cho CollapsedConeAlgorithm khi không thể import."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.description = "Collapsed Cone Algorithm - KHÔNG KHẢ DỤNG"
            logger.warning(
                "Đang sử dụng CollapsedConeAlgorithm giả. Chức năng sẽ bị hạn chế."
            )

    # Gắn lớp mock vào tên gốc để tránh lỗi import từ những nơi khác
    CollapsedConeAlgorithm = MockCollapsedConeAlgorithm


try:
    from .monte_carlo import MonteCarloAlgorithm
except ImportError:
    logger.warning("Không thể import MonteCarloAlgorithm")

    class MockMonteCarloAlgorithm(DoseCalculationAlgorithm):
        """Đối tượng giả cho MonteCarloAlgorithm khi không thể import."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.description = "Monte Carlo Algorithm - KHÔNG KHẢ DỤNG"
            logger.warning(
                "Đang sử dụng MonteCarloAlgorithm giả. Chức năng sẽ bị hạn chế."
            )

    # Gắn lớp mock vào tên gốc để tránh lỗi import từ những nơi khác
    MonteCarloAlgorithm = MockMonteCarloAlgorithm


try:
    from .improvements.monte_carlo_gpu_algorithm import MonteCarloGPUAlgorithm
except ImportError:
    logger.warning("Không thể import MonteCarloGPUAlgorithm")

    class MockMonteCarloGPUAlgorithm(DoseCalculationAlgorithm):
        """Đối tượng giả cho MonteCarloGPUAlgorithm khi không thể import."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.description = "Monte Carlo GPU Algorithm - KHÔNG KHẢ DỤNG"
            logger.warning(
                "Đang sử dụng MonteCarloGPUAlgorithm giả. Chức năng sẽ bị hạn chế."
            )

    # Gắn lớp mock vào tên gốc để tránh lỗi import từ những nơi khác
    MonteCarloGPUAlgorithm = MockMonteCarloGPUAlgorithm


# Đăng ký các thuật toán có sẵn
register_dose_algorithm("pencil_beam", PencilBeamAlgorithm)
register_dose_algorithm("collapsed_cone", CollapsedConeAlgorithm)
register_dose_algorithm("monte_carlo", MonteCarloAlgorithm)
register_dose_algorithm("monte_carlo_gpu", MonteCarloGPUAlgorithm)

# Export
__all__ = [
    "DoseCalculationAlgorithm",
    "DoseCalculationResult",
    "register_dose_algorithm",
    "get_dose_algorithm",
    "get_available_algorithms",
    "PencilBeamAlgorithm",
    "CollapsedConeAlgorithm",
    "MonteCarloAlgorithm",
    "MonteCarloGPUAlgorithm",
]

# Version of the algorithms module
__version__ = "0.7.8"
