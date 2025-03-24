"""
Định nghĩa các hằng số dùng chung trong hệ thống QuangTPS.
"""

from enum import Enum, auto

# Epsilon cho phép toán số
EPSILON = 1e-6

class DoseCalculationAlgorithm(str, Enum):
    """Các thuật toán tính toán liều trong QuangTPS."""
    COLLAPSED_CONE = "Collapsed Cone Convolution"
    PENCIL_BEAM = "Pencil Beam"
    AAA = "Analytical Anisotropic Algorithm"
    MONTE_CARLO = "Monte Carlo"
    GBBS = "Grid-Based Boltzmann Solver"

class DICOMTypes(Enum):
    """Các loại dữ liệu DICOM được hỗ trợ trong QuangTPS."""
    CT = auto()
    RTDOSE = auto() 
    RTPLAN = auto()
    RTSTRUCT = auto()
    RTIMAGE = auto()
    PT = auto()  # PET
    MR = auto()  # MRI 
    US = auto()  # Ultrasound
    RF = auto()  # X-Ray Radiofluoroscopic
    XA = auto()  # X-Ray Angiographic
    DX = auto()  # Digital Radiography
    SR = auto()  # Structured Report

class ContourOperations(Enum):
    """Các phép toán trên contour."""
    UNION = auto()  # Phép hợp
    INTERSECTION = auto()  # Phép giao
    DIFFERENCE = auto()  # Phép trừ
    SYMMETRIC_DIFFERENCE = auto()  # Phép XOR
    DILATION = auto()  # Phép giãn
    EROSION = auto()  # Phép co
    MARGIN = auto()  # Thêm biên
    CROP = auto()  # Cắt

class ReportFormat(Enum):
    """Các định dạng báo cáo được hỗ trợ."""
    PDF = auto()
    HTML = auto()
    DOCX = auto()
    TXT = auto()
    DICOM_SR = auto()  # DICOM Structured Report

class OptimizationAlgorithm(Enum):
    """Các thuật toán tối ưu hóa kế hoạch xạ trị."""
    GRADIENT_DESCENT = auto()  # Gradient Descent
    NEWTON = auto()  # Newton Method
    L_BFGS = auto()  # Limited memory BFGS
    MCO = auto()  # Multi-Criteria Optimization
    PSO = auto()  # Particle Swarm Optimization
    GA = auto()  # Genetic Algorithm
    IPOPT = auto()  # Interior Point Optimizer

class Constants:
    """Chứa các hằng số dùng chung trong hệ thống"""
    
    # Đơn vị đo
    MM_PER_CM = 10.0
    CM_PER_M = 100.0
    
    # Các hằng số vật lý
    ELECTRON_REST_MASS = 9.10938356e-31  # kg
    ELECTRON_CHARGE = 1.60217662e-19  # Coulomb
    SPEED_OF_LIGHT = 299792458.0  # m/s
    PLANCK_CONSTANT = 6.62607004e-34  # J*s
    
    # Hằng số độ phân giải
    DEFAULT_CT_RESOLUTION = 512
    DEFAULT_DOSE_GRID_SIZE = 3.0  # mm
    DEFAULT_SLICE_SPACING = 3.0  # mm
    
    # Hằng số giá trị
    DEFAULT_HU_AIR = -1000
    DEFAULT_HU_WATER = 0
    CT_LEVEL_DEFAULT = 40
    CT_WINDOW_DEFAULT = 400
    
    # Màu sắc mặc định
    DEFAULT_ISODOSE_COLORS = {
        110: (204, 0, 0),     # Đỏ đậm 
        105: (255, 0, 0),     # Đỏ
        100: (255, 102, 102), # Hồng
        95: (255, 153, 51),   # Cam
        90: (255, 255, 0),    # Vàng
        80: (0, 204, 0),      # Xanh lá
        70: (0, 255, 255),    # Xanh lam
        50: (51, 51, 255),    # Xanh dương
        30: (0, 0, 204),      # Xanh đậm
        10: (127, 0, 255)     # Tím
    }
    
    # Thông tin khác
    DEFAULT_INSTITUTION = "QuangTPS"
    VERSION = "0.1.0"
