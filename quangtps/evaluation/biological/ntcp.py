"""
Module tính toán NTCP (Normal Tissue Complication Probability) cho đánh giá kế hoạch xạ trị.

Module này cung cấp các hàm để tính toán xác suất biến chứng mô lành (NTCP) dựa trên 
các mô hình sinh học khác nhau như Lyman-Kutcher-Burman (LKB), mô hình Niemierko,
và mô hình Relative Seriality.
"""

import numpy as np
import logging
from scipy import stats, integrate
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

def calculate_ntcp_lkb(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    num_fractions: int,
    td50: float = None,
    n: float = None,
    m: float = None,
    alpha_beta: float = 3.0,
    dose_threshold: Optional[float] = None,
    organ_parameters: Optional[Dict[str, Dict[str, float]]] = None,
    organ_type: Optional[str] = None
) -> float:
    """
    Tính toán NTCP dựa trên mô hình Lyman-Kutcher-Burman (LKB).
    
    NTCP = 1/√2π ∫_{-∞}^t exp(-x²/2)dx, với t = (EUD - TD50)/(m*TD50)
    
    Parameters:
        dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
        structure_mask (np.ndarray): Mảng mask 3D của cơ quan nguy cấp
        num_fractions (int): Số phân liều
        td50 (float, optional): Liều đồng nhất gây ra biến chứng với xác suất 50% (Gy)
        n (float, optional): Tham số thể hiện tính nối tiếp/song song của cơ quan (0 < n < 1)
        m (float, optional): Tham số thể hiện độ dốc của đường cong liều-đáp ứng
        alpha_beta (float, optional): Tỉ lệ alpha/beta cho cơ quan (Gy)
        dose_threshold (float, optional): Ngưỡng liều để tính NTCP, voxel có liều < ngưỡng sẽ bị bỏ qua
        organ_parameters (dict, optional): Dict tham số cho các cơ quan khác nhau
        organ_type (str, optional): Loại cơ quan, dùng để lấy tham số từ organ_parameters
    
    Returns:
        float: Giá trị NTCP (0-1)
    
    Raises:
        ValueError: Nếu không có đủ tham số và không xác định được loại cơ quan
    """
    # Kiểm tra mask và dose có cùng kích thước
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
    # Nếu không cung cấp trực tiếp tham số td50, n, m, thử lấy từ organ_parameters
    if (td50 is None or n is None or m is None) and organ_type is not None and organ_parameters is not None:
        if organ_type in organ_parameters:
            params = organ_parameters[organ_type]
            td50 = params.get('td50', td50)
            n = params.get('n', n)
            m = params.get('m', m)
        else:
            logger.warning(f"Organ type '{organ_type}' not found in organ_parameters")
    
    # Kiểm tra nếu vẫn thiếu tham số
    if td50 is None or n is None or m is None:
        # Cung cấp tham số mặc định cho một số cơ quan phổ biến
        default_params = {
            'lung': {'td50': 24.5, 'n': 0.87, 'm': 0.18, 'endpoint': 'Pneumonitis'},
            'heart': {'td50': 48.0, 'n': 0.35, 'm': 0.10, 'endpoint': 'Pericarditis'},
            'parotid': {'td50': 46.0, 'n': 0.7, 'm': 0.18, 'endpoint': 'Xerostomia'},
            'liver': {'td50': 40.0, 'n': 0.97, 'm': 0.12, 'endpoint': 'Liver failure'},
            'kidney': {'td50': 28.0, 'n': 0.7, 'm': 0.1, 'endpoint': 'Nephropathy'},
            'brain': {'td50': 60.0, 'n': 0.25, 'm': 0.15, 'endpoint': 'Necrosis'},
            'spinal_cord': {'td50': 66.5, 'n': 0.05, 'm': 0.175, 'endpoint': 'Myelopathy'},
            'rectum': {'td50': 80.0, 'n': 0.12, 'm': 0.15, 'endpoint': 'Proctitis'},
            'bladder': {'td50': 80.0, 'n': 0.5, 'm': 0.11, 'endpoint': 'Contracture'},
            'esophagus': {'td50': 68.0, 'n': 0.06, 'm': 0.11, 'endpoint': 'Stricture'},
            'small_bowel': {'td50': 55.0, 'n': 0.15, 'm': 0.16, 'endpoint': 'Obstruction'},
            'larynx': {'td50': 70.0, 'n': 0.08, 'm': 0.17, 'endpoint': 'Edema'},
            'optic_chiasm': {'td50': 65.0, 'n': 0.25, 'm': 0.14, 'endpoint': 'Blindness'},
            'brainstem': {'td50': 65.0, 'n': 0.16, 'm': 0.14, 'endpoint': 'Necrosis'}
        }
        
        # Nếu có organ_type và nó tồn tại trong default_params
        if organ_type is not None and organ_type.lower() in default_params:
            params = default_params[organ_type.lower()]
            td50 = params.get('td50', 50.0)
            n = params.get('n', 0.5)
            m = params.get('m', 0.1)
            logger.info(f"Using default parameters for {organ_type}: TD50={td50}, n={n}, m={m}")
        else:
            # Sử dụng giá trị mặc định chung
            td50 = 50.0
            n = 0.5
            m = 0.1
            logger.warning(f"Using generic parameters: TD50={td50}, n={n}, m={m}")
    
    # Chỉ xét các voxel trong cơ quan
    mask = structure_mask > 0
    if dose_threshold is not None:
        mask = mask & (dose_array >= dose_threshold)
    
    # Nếu không có voxel nào thỏa điều kiện
    if not np.any(mask):
        logger.warning("No valid voxels found for NTCP calculation")
        return 0.0
    
    # Lấy liều tại các voxel trong cơ quan
    organ_doses = dose_array[mask]
    
    # Liều mỗi phân liều (Gy)
    dose_per_fraction = organ_doses / num_fractions
    
    # Tính EQD2 cho mỗi voxel
    eqd2 = organ_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
    # Tính tỉ lệ thể tích của mỗi voxel (tất cả voxel có cùng kích thước)
    num_voxels = np.sum(mask)
    volume_fraction = 1.0 / num_voxels
    
    # Tính gEUD theo công thức Niemierko với tham số a = 1/n
    a = 1.0 / n
    geud = np.power(np.sum(volume_fraction * np.power(eqd2, a)), 1.0/a)
    
    # Tính giá trị t trong mô hình Lyman
    t = (geud - td50) / (m * td50)
    
    # Tính NTCP theo mô hình Lyman sử dụng hàm lỗi
    ntcp = 0.5 * (1 + stats.norm.cdf(t))
    
    return ntcp

def calculate_ntcp_relative_seriality(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    num_fractions: int,
    d50: float,
    gamma50: float,
    seriality: float,
    alpha_beta: float = 3.0,
    dose_threshold: Optional[float] = None
) -> float:
    """
    Tính toán NTCP dựa trên mô hình Relative Seriality.
    
    Parameters:
        dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
        structure_mask (np.ndarray): Mảng mask 3D của cơ quan nguy cấp
        num_fractions (int): Số phân liều
        d50 (float): Liều đồng nhất gây ra biến chứng với xác suất 50% (Gy)
        gamma50 (float): Độ dốc của đường cong liều-đáp ứng tại 50%
        seriality (float): Tham số thể hiện tính nối tiếp của cơ quan (0-1)
        alpha_beta (float, optional): Tỉ lệ alpha/beta cho cơ quan (Gy)
        dose_threshold (float, optional): Ngưỡng liều để tính NTCP, voxel có liều < ngưỡng sẽ bị bỏ qua
    
    Returns:
        float: Giá trị NTCP (0-1)
    """
    # Kiểm tra mask và dose có cùng kích thước
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
    # Chỉ xét các voxel trong cơ quan
    mask = structure_mask > 0
    if dose_threshold is not None:
        mask = mask & (dose_array >= dose_threshold)
    
    # Nếu không có voxel nào thỏa điều kiện
    if not np.any(mask):
        logger.warning("No valid voxels found for NTCP calculation")
        return 0.0
    
    # Lấy liều tại các voxel trong cơ quan
    organ_doses = dose_array[mask]
    
    # Liều mỗi phân liều (Gy)
    dose_per_fraction = organ_doses / num_fractions
    
    # Tính EQD2 cho mỗi voxel
    eqd2 = organ_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
    # Tính tỉ lệ thể tích của mỗi voxel (tất cả voxel có cùng kích thước)
    num_voxels = np.sum(mask)
    dv = 1.0 / num_voxels
    
    # Tính xác suất biến chứng cho mỗi voxel
    e = np.exp(gamma50 * (1 - eqd2/d50))
    p = 1.0 / (1.0 + e)
    
    # Tính NTCP theo mô hình Relative Seriality
    ntcp = np.power(np.prod(np.power(1.0 - p, dv * seriality)), 1.0/seriality)
    ntcp = 1.0 - ntcp
    
    return ntcp

def calculate_ntcp_logit(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    num_fractions: int,
    d50: float,
    k: float,
    alpha_beta: float = 3.0,
    dose_threshold: Optional[float] = None
) -> float:
    """
    Tính toán NTCP dựa trên mô hình Logit.
    
    NTCP = 1 / (1 + (D50/D)^k)
    
    Parameters:
        dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
        structure_mask (np.ndarray): Mảng mask 3D của cơ quan nguy cấp
        num_fractions (int): Số phân liều
        d50 (float): Liều đồng nhất gây ra biến chứng với xác suất 50% (Gy)
        k (float): Tham số k trong mô hình Logit
        alpha_beta (float, optional): Tỉ lệ alpha/beta cho cơ quan (Gy)
        dose_threshold (float, optional): Ngưỡng liều để tính NTCP, voxel có liều < ngưỡng sẽ bị bỏ qua
    
    Returns:
        float: Giá trị NTCP (0-1)
    """
    # Kiểm tra mask và dose có cùng kích thước
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
    # Chỉ xét các voxel trong cơ quan
    mask = structure_mask > 0
    if dose_threshold is not None:
        mask = mask & (dose_array >= dose_threshold)
    
    # Nếu không có voxel nào thỏa điều kiện
    if not np.any(mask):
        logger.warning("No valid voxels found for NTCP calculation")
        return 0.0
    
    # Lấy liều tại các voxel trong cơ quan
    organ_doses = dose_array[mask]
    
    # Liều mỗi phân liều (Gy)
    dose_per_fraction = organ_doses / num_fractions
    
    # Tính EQD2 cho mỗi voxel
    eqd2 = organ_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
    # Tính liều trung bình
    mean_dose = np.mean(eqd2)
    
    # Tính NTCP theo mô hình Logit
    ntcp = 1.0 / (1.0 + np.power(d50/mean_dose, k))
    
    return ntcp

def calculate_ntcp_poisson(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    num_fractions: int,
    d50: float,
    gamma50: float,
    alpha_beta: float = 3.0,
    dose_threshold: Optional[float] = None
) -> float:
    """
    Tính toán NTCP dựa trên mô hình Poisson.
    
    NTCP = 1 - exp(-exp(e0 + gamma * (D - D50)/D50))
    
    Parameters:
        dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
        structure_mask (np.ndarray): Mảng mask 3D của cơ quan nguy cấp
        num_fractions (int): Số phân liều
        d50 (float): Liều đồng nhất gây ra biến chứng với xác suất 50% (Gy)
        gamma50 (float): Độ dốc của đường cong liều-đáp ứng tại 50%
        alpha_beta (float, optional): Tỉ lệ alpha/beta cho cơ quan (Gy)
        dose_threshold (float, optional): Ngưỡng liều để tính NTCP, voxel có liều < ngưỡng sẽ bị bỏ qua
    
    Returns:
        float: Giá trị NTCP (0-1)
    """
    # Kiểm tra mask và dose có cùng kích thước
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
    # Chỉ xét các voxel trong cơ quan
    mask = structure_mask > 0
    if dose_threshold is not None:
        mask = mask & (dose_array >= dose_threshold)
    
    # Nếu không có voxel nào thỏa điều kiện
    if not np.any(mask):
        logger.warning("No valid voxels found for NTCP calculation")
        return 0.0
    
    # Lấy liều tại các voxel trong cơ quan
    organ_doses = dose_array[mask]
    
    # Liều mỗi phân liều (Gy)
    dose_per_fraction = organ_doses / num_fractions
    
    # Tính EQD2 cho mỗi voxel
    eqd2 = organ_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
    # Tính liều trung bình
    mean_dose = np.mean(eqd2)
    
    # Hằng số e0
    e0 = -np.log(np.log(2))
    
    # Tính NTCP theo mô hình Poisson
    ntcp = 1.0 - np.exp(-np.exp(e0 + gamma50 * (mean_dose - d50)/d50))
    
    return ntcp

def calculate_cutoff_ntcp(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    tolerance_dose: float,
    critical_volume: float,
    alpha_beta: float = 3.0,
    num_fractions: int = 1
) -> float:
    """
    Tính toán NTCP dựa trên mô hình ngưỡng đơn giản.
    
    Nếu thể tích nhận liều > tolerance_dose vượt quá critical_volume, NTCP = 1, ngược lại NTCP = 0.
    
    Parameters:
        dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
        structure_mask (np.ndarray): Mảng mask 3D của cơ quan nguy cấp
        tolerance_dose (float): Ngưỡng liều tới hạn (Gy)
        critical_volume (float): Thể tích tới hạn (%)
        alpha_beta (float, optional): Tỉ lệ alpha/beta cho cơ quan (Gy)
        num_fractions (int, optional): Số phân liều
    
    Returns:
        float: Giá trị NTCP (0 hoặc 1)
    """
    # Kiểm tra mask và dose có cùng kích thước
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
    # Chỉ xét các voxel trong cơ quan
    mask = structure_mask > 0
    
    # Nếu không có voxel nào trong cơ quan
    if not np.any(mask):
        logger.warning("No voxels found in the structure")
        return 0.0
    
    # Lấy liều tại các voxel trong cơ quan
    organ_doses = dose_array[mask]
    
    # Liều mỗi phân liều (Gy)
    if num_fractions > 1:
        dose_per_fraction = organ_doses / num_fractions
        # Tính EQD2 cho mỗi voxel
        organ_doses = organ_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
    # Tính thể tích nhận liều > tolerance_dose
    volume_over_tolerance = np.sum(organ_doses > tolerance_dose) / len(organ_doses) * 100.0
    
    # Nếu thể tích nhận liều > tolerance_dose vượt quá critical_volume, NTCP = 1
    if volume_over_tolerance > critical_volume:
        return 1.0
    else:
        return 0.0

def get_ntcp_constraints(
    organ_type: str,
    endpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lấy các ràng buộc NTCP cho một cơ quan cụ thể.
    
    Parameters:
        organ_type (str): Loại cơ quan
        endpoint (str, optional): Điểm cuối cụ thể cần xem xét
    
    Returns:
        dict: Dict chứa các ràng buộc và tham số NTCP
    """
    # Dict các ràng buộc NTCP cho các cơ quan khác nhau
    # Dựa trên các nghiên cứu và hướng dẫn lâm sàng
    constraints = {
        'lung': {
            'pneumonitis': {
                'model': 'lkb',
                'parameters': {'td50': 24.5, 'n': 0.87, 'm': 0.18, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'V20', 'value': 30, 'unit': '%', 'priority': 'high'},
                    {'type': 'V5', 'value': 60, 'unit': '%', 'priority': 'medium'},
                    {'type': 'MLD', 'value': 20, 'unit': 'Gy', 'priority': 'high'}
                ],
                'ntcp_threshold': 0.2
            }
        },
        'heart': {
            'pericarditis': {
                'model': 'lkb',
                'parameters': {'td50': 48.0, 'n': 0.35, 'm': 0.10, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'V25', 'value': 10, 'unit': '%', 'priority': 'high'},
                    {'type': 'Mean', 'value': 26, 'unit': 'Gy', 'priority': 'high'}
                ],
                'ntcp_threshold': 0.15
            }
        },
        'parotid': {
            'xerostomia': {
                'model': 'lkb',
                'parameters': {'td50': 46.0, 'n': 0.7, 'm': 0.18, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'Mean', 'value': 26, 'unit': 'Gy', 'priority': 'high'},
                    {'type': 'V30', 'value': 45, 'unit': '%', 'priority': 'medium'}
                ],
                'ntcp_threshold': 0.25
            }
        },
        'liver': {
            'liver_failure': {
                'model': 'lkb',
                'parameters': {'td50': 40.0, 'n': 0.97, 'm': 0.12, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'V30', 'value': 30, 'unit': '%', 'priority': 'high'},
                    {'type': 'Mean', 'value': 30, 'unit': 'Gy', 'priority': 'high'}
                ],
                'ntcp_threshold': 0.1
            }
        },
        'kidney': {
            'nephropathy': {
                'model': 'lkb',
                'parameters': {'td50': 28.0, 'n': 0.7, 'm': 0.1, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'V18', 'value': 33, 'unit': '%', 'priority': 'high'},
                    {'type': 'Mean', 'value': 18, 'unit': 'Gy', 'priority': 'high'}
                ],
                'ntcp_threshold': 0.05
            }
        },
        'brain': {
            'necrosis': {
                'model': 'lkb',
                'parameters': {'td50': 60.0, 'n': 0.25, 'm': 0.15, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'Max', 'value': 60, 'unit': 'Gy', 'priority': 'high'}
                ],
                'ntcp_threshold': 0.05
            }
        },
        'spinal_cord': {
            'myelopathy': {
                'model': 'lkb',
                'parameters': {'td50': 66.5, 'n': 0.05, 'm': 0.175, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'Max', 'value': 45, 'unit': 'Gy', 'priority': 'high'},
                    {'type': 'V40', 'value': 0.1, 'unit': 'cc', 'priority': 'high'}
                ],
                'ntcp_threshold': 0.01
            }
        },
        'rectum': {
            'proctitis': {
                'model': 'lkb',
                'parameters': {'td50': 80.0, 'n': 0.12, 'm': 0.15, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'V70', 'value': 15, 'unit': '%', 'priority': 'high'},
                    {'type': 'V50', 'value': 50, 'unit': '%', 'priority': 'medium'}
                ],
                'ntcp_threshold': 0.15
            }
        },
        'bladder': {
            'contracture': {
                'model': 'lkb',
                'parameters': {'td50': 80.0, 'n': 0.5, 'm': 0.11, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'V70', 'value': 35, 'unit': '%', 'priority': 'high'},
                    {'type': 'V65', 'value': 50, 'unit': '%', 'priority': 'medium'}
                ],
                'ntcp_threshold': 0.15
            }
        },
        'esophagus': {
            'stricture': {
                'model': 'lkb',
                'parameters': {'td50': 68.0, 'n': 0.06, 'm': 0.11, 'alpha_beta': 3.0},
                'constraints': [
                    {'type': 'V55', 'value': 30, 'unit': '%', 'priority': 'high'},
                    {'type': 'Mean', 'value': 34, 'unit': 'Gy', 'priority': 'medium'}
                ],
                'ntcp_threshold': 0.2
            }
        }
    }
    
    # Lấy thông tin cho cơ quan cụ thể
    organ_info = constraints.get(organ_type.lower())
    if organ_info is None:
        logger.warning(f"No NTCP constraints found for organ '{organ_type}'")
        return {}
    
    # Nếu không chỉ định endpoint, lấy endpoint đầu tiên
    if endpoint is None:
        endpoint = list(organ_info.keys())[0]
    elif endpoint.lower() not in organ_info:
        logger.warning(f"Endpoint '{endpoint}' not found for organ '{organ_type}', using default")
        endpoint = list(organ_info.keys())[0]
    
    return organ_info[endpoint.lower()]

def calculate_ntcp_for_dvh(
    dvh_data: Dict[str, np.ndarray],
    model: str = 'lkb',
    parameters: Dict[str, float] = None,
    organ_type: Optional[str] = None,
    num_fractions: int = 30
) -> float:
    """
    Tính toán NTCP từ dữ liệu DVH sử dụng các mô hình khác nhau.
    
    Parameters:
        dvh_data (dict): Dict chứa dữ liệu DVH với các key:
            - 'dose': Mảng giá trị liều (Gy)
            - 'volume': Mảng giá trị thể tích (% hoặc cc)
            - 'type': Loại DVH ('cumulative' hoặc 'differential')
        model (str, optional): Mô hình NTCP ('lkb', 'relative_seriality', 'logit', 'poisson')
        parameters (dict, optional): Tham số cho mô hình
        organ_type (str, optional): Loại cơ quan, dùng để lấy tham số mặc định
        num_fractions (int, optional): Số phân liều
    
    Returns:
        float: Giá trị NTCP (0-1)
    
    Raises:
        ValueError: Nếu không có đủ tham số và không xác định được loại cơ quan
    """
    # Kiểm tra loại DVH
    is_cumulative = dvh_data.get('type', 'cumulative') == 'cumulative'
    
    # Lấy dữ liệu từ DVH
    doses = dvh_data['dose']
    volumes = dvh_data['volume']
    
    # Nếu là DVH tích lũy, chuyển thành DVH vi phân
    if is_cumulative:
        # Tính delta volume
        diff_volumes = np.abs(np.diff(volumes, append=0))
    else:
        diff_volumes = volumes
    
    # Nếu không cung cấp tham số, thử lấy từ organ_type
    if parameters is None:
        if organ_type is None:
            raise ValueError("Either parameters or organ_type must be provided")
        
        # Lấy tham số cho cơ quan cụ thể
        organ_info = get_ntcp_constraints(organ_type)
        if not organ_info:
            raise ValueError(f"No parameters found for organ '{organ_type}'")
        
        parameters = organ_info.get('parameters', {})
        model = organ_info.get('model', model)
    
    # Tính NTCP theo mô hình được chọn
    if model.lower() == 'lkb':
        # Cần có td50, n, m
        td50 = parameters.get('td50', 50.0)
        n = parameters.get('n', 0.5)
        m = parameters.get('m', 0.1)
        alpha_beta = parameters.get('alpha_beta', 3.0)
        
        # Tính gEUD
        a = 1.0 / n
        # Chuẩn hóa diff_volumes
        norm_volumes = diff_volumes / np.sum(diff_volumes)
        # Tính EQD2
        dose_per_fraction = doses / num_fractions
        eqd2 = doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
        
        # Tính gEUD
        geud = np.power(np.sum(norm_volumes * np.power(eqd2, a)), 1.0/a)
        
        # Tính giá trị t
        t = (geud - td50) / (m * td50)
        
        # Tính NTCP
        ntcp = 0.5 * (1 + stats.norm.cdf(t))
    
    elif model.lower() == 'relative_seriality':
        d50 = parameters.get('d50', 50.0)
        gamma50 = parameters.get('gamma50', 2.0)
        seriality = parameters.get('seriality', 0.5)
        alpha_beta = parameters.get('alpha_beta', 3.0)
        
        # Chuẩn hóa diff_volumes
        norm_volumes = diff_volumes / np.sum(diff_volumes)
        
        # Tính EQD2
        dose_per_fraction = doses / num_fractions
        eqd2 = doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
        
        # Tính xác suất biến chứng cho mỗi bin
        e = np.exp(gamma50 * (1 - eqd2/d50))
        p = 1.0 / (1.0 + e)
        
        # Tính NTCP
        ntcp = np.power(np.prod(np.power(1.0 - p, norm_volumes * seriality)), 1.0/seriality)
        ntcp = 1.0 - ntcp
    
    elif model.lower() == 'logit':
        d50 = parameters.get('d50', 50.0)
        k = parameters.get('k', 4.0)
        alpha_beta = parameters.get('alpha_beta', 3.0)
        
        # Chuẩn hóa diff_volumes
        norm_volumes = diff_volumes / np.sum(diff_volumes)
        
        # Tính EQD2
        dose_per_fraction = doses / num_fractions
        eqd2 = doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
        
        # Tính liều trung bình
        mean_dose = np.sum(norm_volumes * eqd2)
        
        # Tính NTCP
        ntcp = 1.0 / (1.0 + np.power(d50/mean_dose, k))
    
    elif model.lower() == 'poisson':
        d50 = parameters.get('d50', 50.0)
        gamma50 = parameters.get('gamma50', 2.0)
        alpha_beta = parameters.get('alpha_beta', 3.0)
        
        # Chuẩn hóa diff_volumes
        norm_volumes = diff_volumes / np.sum(diff_volumes)
        
        # Tính EQD2
        dose_per_fraction = doses / num_fractions
        eqd2 = doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
        
        # Tính liều trung bình
        mean_dose = np.sum(norm_volumes * eqd2)
        
        # Hằng số e0
        e0 = -np.log(np.log(2))
        
        # Tính NTCP
        ntcp = 1.0 - np.exp(-np.exp(e0 + gamma50 * (mean_dose - d50)/d50))
    
    else:
        raise ValueError(f"Unknown NTCP model '{model}'")
    
    return ntcp

class NTCPModels:
    """
    Lớp cung cấp các phương thức tính xác suất biến chứng mô lành (NTCP).
    
    Lớp này bao gồm các mô hình sinh học khác nhau như Lyman-Kutcher-Burman (LKB),
    mô hình Relative Seriality, mô hình Logit, và mô hình Poisson.
    """
    
    @staticmethod
    def calculate_ntcp_lkb(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        td50: float = None,
        n: float = None,
        m: float = None,
        alpha_beta: float = 3.0,
        dose_threshold: Optional[float] = None,
        organ_parameters: Optional[Dict[str, Dict[str, float]]] = None,
        organ_type: Optional[str] = None
    ) -> float:
        """
        Tính toán NTCP dựa trên mô hình Lyman-Kutcher-Burman (LKB).
        
        Xem hàm calculate_ntcp_lkb của module để biết thêm chi tiết.
        """
        return calculate_ntcp_lkb(
            dose_array, structure_mask, num_fractions, td50, n, m, 
            alpha_beta, dose_threshold, organ_parameters, organ_type
        )
    
    @staticmethod
    def calculate_ntcp_relative_seriality(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        d50: float,
        gamma50: float,
        seriality: float,
        alpha_beta: float = 3.0,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán NTCP dựa trên mô hình Relative Seriality.
        
        Xem hàm calculate_ntcp_relative_seriality của module để biết thêm chi tiết.
        """
        return calculate_ntcp_relative_seriality(
            dose_array, structure_mask, num_fractions, d50, gamma50, 
            seriality, alpha_beta, dose_threshold
        )
    
    @staticmethod
    def calculate_ntcp_logit(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        d50: float,
        k: float,
        alpha_beta: float = 3.0,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán NTCP dựa trên mô hình Logit.
        
        Xem hàm calculate_ntcp_logit của module để biết thêm chi tiết.
        """
        return calculate_ntcp_logit(
            dose_array, structure_mask, num_fractions, d50, k, 
            alpha_beta, dose_threshold
        )
    
    @staticmethod
    def calculate_ntcp_poisson(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        d50: float,
        gamma50: float,
        alpha_beta: float = 3.0,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán NTCP dựa trên mô hình Poisson.
        
        Xem hàm calculate_ntcp_poisson của module để biết thêm chi tiết.
        """
        return calculate_ntcp_poisson(
            dose_array, structure_mask, num_fractions, d50, gamma50, 
            alpha_beta, dose_threshold
        )
    
    @staticmethod
    def calculate_cutoff_ntcp(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        tolerance_dose: float,
        critical_volume: float,
        alpha_beta: float = 3.0,
        num_fractions: int = 1
    ) -> float:
        """
        Tính toán NTCP dựa trên mô hình ngưỡng đơn giản.
        
        Xem hàm calculate_cutoff_ntcp của module để biết thêm chi tiết.
        """
        return calculate_cutoff_ntcp(
            dose_array, structure_mask, tolerance_dose, critical_volume, 
            alpha_beta, num_fractions
        )
    
    @staticmethod
    def get_ntcp_constraints(
        organ_type: str,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Lấy các ràng buộc NTCP cho một cơ quan cụ thể.
        
        Xem hàm get_ntcp_constraints của module để biết thêm chi tiết.
        """
        return get_ntcp_constraints(organ_type, endpoint)
    
    @staticmethod
    def calculate_ntcp_for_dvh(
        dvh_data: Dict[str, np.ndarray],
        model: str = 'lkb',
        parameters: Dict[str, float] = None,
        organ_type: Optional[str] = None,
        num_fractions: int = 30
    ) -> float:
        """
        Tính toán NTCP từ dữ liệu DVH sử dụng các mô hình khác nhau.
        
        Xem hàm calculate_ntcp_for_dvh của module để biết thêm chi tiết.
        """
        return calculate_ntcp_for_dvh(
            dvh_data, model, parameters, organ_type, num_fractions
        )
