"""
Module tính toán TCP (Tumor Control Probability) cho đánh giá kế hoạch xạ trị.

Module này cung cấp các hàm để tính toán xác suất kiểm soát khối u (TCP) dựa trên
các mô hình sinh học khác nhau như LQ (Linear-Quadratic), Poisson, v.v.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

def calculate_tcp_lq_poisson(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    num_fractions: int,
    alpha: float = 0.3,
    alpha_beta: float = 10.0,
    clonogenic_density: float = 1e7,
    dose_threshold: Optional[float] = None
) -> float:
    """
    Tính toán TCP dựa trên mô hình LQ-Poisson.

    TCP = exp(-N0 * sum_i(exp(-alpha*EQD2_i)))

    với EQD2_i = D_i * (1 + d_i/(alpha/beta)) / (1 + 2/(alpha/beta))

    Parameters
    ----------
    dose_array : np.ndarray
        Mảng liều 3D (Gy)
    structure_mask : np.ndarray
        Mặt nạ nhị phân 3D cho cấu trúc mục tiêu
    num_fractions : int
        Số phân liều
    alpha : float, optional
        Tham số alpha trong mô hình tuyến tính-bậc hai (Gy^-1), mặc định là 0.3
    alpha_beta : float, optional
        Tỷ lệ alpha/beta (Gy), mặc định là 10.0
    clonogenic_density : float, optional
        Mật độ tế bào gốc (cells/cm^3), mặc định là 1e7
    dose_threshold : Optional[float], optional
        Ngưỡng liều (Gy), chỉ xem xét voxel có liều > ngưỡng, mặc định là None

    Returns
    -------
    float
        Xác suất kiểm soát khối u (0-1)
    """
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Hình dạng mảng liều {dose_array.shape} không khớp với hình dạng mặt nạ cấu trúc {structure_mask.shape}")
    
    # Áp dụng mặt nạ cấu trúc
    tumor_doses = dose_array[structure_mask > 0]
    if len(tumor_doses) == 0:
        logger.warning("Không có voxel nào trong mặt nạ cấu trúc, TCP = 0")
        return 0.0
    
    # Áp dụng ngưỡng liều nếu được chỉ định
    if dose_threshold is not None:
        tumor_doses = tumor_doses[tumor_doses > dose_threshold]
        if len(tumor_doses) == 0:
            logger.warning(f"Không có voxel nào vượt ngưỡng liều {dose_threshold} Gy, TCP = 0")
            return 0.0
    
    # Tính liều trên mỗi phân liều
    dose_per_fraction = tumor_doses / num_fractions
    
    # Tính liều tương đương 2Gy (EQD2)
    eqd2 = tumor_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
    # Tính tổng thể tích khối u (cm³) dựa trên số voxel
    voxel_count = len(tumor_doses)
    voxel_volume_cm3 = 1.0  # Giả sử thể tích mỗi voxel là 1 cm³, cần điều chỉnh nếu có thông tin voxel thực tế
    tumor_volume_cm3 = voxel_count * voxel_volume_cm3
    
    # Tính tổng số tế bào khối u
    total_clonogenic_cells = clonogenic_density * tumor_volume_cm3
    
    # Xác suất sống sót của tế bào sau khi chiếu xạ
    cell_survival_probability = np.exp(-alpha * eqd2)
    
    # Trung bình số tế bào còn sống sau điều trị
    surviving_cells = total_clonogenic_cells * np.mean(cell_survival_probability)
    
    # TCP dựa trên mô hình Poisson
    tcp = np.exp(-surviving_cells)
    
    return float(tcp)

def calculate_tcp_lq_poisson_dvh(
    dvh: Dict[str, np.ndarray],
    num_fractions: int,
    alpha: float = 0.3,
    alpha_beta: float = 10.0,
    clonogenic_density: float = 1e7,
    dose_threshold: Optional[float] = None
) -> float:
    """
    Tính toán TCP từ DVH (Dose-Volume Histogram) dựa trên mô hình LQ-Poisson.
    
    Parameters
    ----------
    dvh : Dict[str, np.ndarray]
        Dict chứa DVH với keys 'dose' và 'volume'
    num_fractions : int
        Số phân liều
    alpha : float, optional
        Tham số alpha trong mô hình tuyến tính-bậc hai (Gy^-1), mặc định là 0.3
    alpha_beta : float, optional
        Tỷ lệ alpha/beta (Gy), mặc định là 10.0
    clonogenic_density : float, optional
        Mật độ tế bào gốc (cells/cm^3), mặc định là 1e7
    dose_threshold : Optional[float], optional
        Ngưỡng liều (Gy), chỉ xem xét voxel có liều > ngưỡng, mặc định là None
        
    Returns
    -------
    float
        Xác suất kiểm soát khối u (0-1)
    """
    if 'dose' not in dvh or 'volume' not in dvh:
        raise ValueError("DVH phải chứa cả keys 'dose' và 'volume'")
    
    doses = dvh['dose']
    volumes = dvh['volume']
    
    if len(doses) != len(volumes):
        raise ValueError(f"Số lượng điểm liều {len(doses)} không khớp với số lượng điểm thể tích {len(volumes)}")
    
    if len(doses) == 0:
        logger.warning("DVH rỗng, TCP = 0")
        return 0.0
    
    # Áp dụng ngưỡng liều nếu được chỉ định
    if dose_threshold is not None:
        mask = doses > dose_threshold
        doses = doses[mask]
        volumes = volumes[mask]
        if len(doses) == 0:
            logger.warning(f"Không có điểm DVH nào vượt ngưỡng liều {dose_threshold} Gy, TCP = 0")
            return 0.0
    
    # Chuẩn hóa thể tích để tổng bằng 1
    volumes = volumes / np.sum(volumes)
    
    # Tính liều trên mỗi phân liều
    dose_per_fraction = doses / num_fractions
    
    # Tính liều tương đương 2Gy (EQD2)
    eqd2 = doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
    # Xác suất sống sót của tế bào sau khi chiếu xạ cho từng bin DVH
    cell_survival_probability = np.exp(-alpha * eqd2)
    
    # Trung bình có trọng số số tế bào còn sống sau điều trị
    weighted_survival = np.sum(cell_survival_probability * volumes)
    
    # Tổng thể tích khối u (cm³) - giả sử DVH đã được chuẩn hóa
    tumor_volume_cm3 = 1.0  # Cần điều chỉnh nếu có thông tin thực tế
    
    # Tổng số tế bào khối u
    total_clonogenic_cells = clonogenic_density * tumor_volume_cm3
    
    # TCP dựa trên mô hình Poisson
    tcp = np.exp(-total_clonogenic_cells * weighted_survival)
    
    return float(tcp)

def calculate_tcp_niemierko(
    eud: float,
    tcd50: float = 60.0,
    gamma50: float = 2.0
) -> float:
    """
    Tính toán TCP dựa trên mô hình Niemierko.
    
    TCP = 1 / (1 + (TCD50/EUD)^(4*gamma50))
    
    Parameters
    ----------
    eud : float
        Liều đồng nhất tương đương (EUD) (Gy)
    tcd50 : float, optional
        Liều kiểm soát khối u cho 50% bệnh nhân (Gy), mặc định là 60.0
    gamma50 : float, optional
        Độ dốc của đường cong liều-đáp ứng tại TCD50, mặc định là 2.0
        
    Returns
    -------
    float
        Xác suất kiểm soát khối u (0-1)
    """
    if eud <= 0:
        logger.warning("EUD phải dương, TCP = 0")
        return 0.0
    
    if tcd50 <= 0:
        logger.warning("TCD50 phải dương, TCP = 0")
        return 0.0
    
    # Tính TCP theo mô hình Niemierko
    exponent = 4 * gamma50
    tcp = 1 / (1 + (tcd50 / eud) ** exponent)
    
    return float(tcp)

def calculate_tcp_logistic(
    dose: float,
    tcd50: float = 60.0,
    gamma50: float = 2.0
) -> float:
    """
    Tính toán TCP dựa trên mô hình logistic.
    
    TCP = 1 / (1 + exp(-4*gamma50*(dose/tcd50 - 1)))
    
    Parameters
    ----------
    dose : float
        Liều (Gy)
    tcd50 : float, optional
        Liều kiểm soát khối u cho 50% bệnh nhân (Gy), mặc định là 60.0
    gamma50 : float, optional
        Độ dốc của đường cong liều-đáp ứng tại TCD50, mặc định là 2.0
        
    Returns
    -------
    float
        Xác suất kiểm soát khối u (0-1)
    """
    if dose < 0:
        logger.warning("Liều không thể âm, TCP = 0")
        return 0.0
    
    if tcd50 <= 0:
        logger.warning("TCD50 phải dương, TCP = 0")
        return 0.0
    
    # Tính TCP theo mô hình logistic
    exponent = -4 * gamma50 * (dose / tcd50 - 1)
    tcp = 1 / (1 + np.exp(exponent))
    
    return float(tcp)

def calculate_tcp_webb(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    alpha_mean: float = 0.3,
    alpha_std: float = 0.1,
    dose_threshold: Optional[float] = None
) -> float:
    """
    Tính toán TCP dựa trên mô hình Webb với tính không đồng nhất của tính nhạy cảm phóng xạ.
    
    Parameters
    ----------
    dose_array : np.ndarray
        Mảng liều 3D (Gy)
    structure_mask : np.ndarray
        Mặt nạ nhị phân 3D cho cấu trúc mục tiêu
    alpha_mean : float, optional
        Giá trị trung bình của tham số alpha (Gy^-1), mặc định là 0.3
    alpha_std : float, optional
        Độ lệch chuẩn của tham số alpha (Gy^-1), mặc định là 0.1
    dose_threshold : Optional[float], optional
        Ngưỡng liều (Gy), chỉ xem xét voxel có liều > ngưỡng, mặc định là None
        
    Returns
    -------
    float
        Xác suất kiểm soát khối u (0-1)
    """
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Hình dạng mảng liều {dose_array.shape} không khớp với hình dạng mặt nạ cấu trúc {structure_mask.shape}")
    
    # Áp dụng mặt nạ cấu trúc
    tumor_doses = dose_array[structure_mask > 0]
    if len(tumor_doses) == 0:
        logger.warning("Không có voxel nào trong mặt nạ cấu trúc, TCP = 0")
        return 0.0
    
    # Áp dụng ngưỡng liều nếu được chỉ định
    if dose_threshold is not None:
        tumor_doses = tumor_doses[tumor_doses > dose_threshold]
        if len(tumor_doses) == 0:
            logger.warning(f"Không có voxel nào vượt ngưỡng liều {dose_threshold} Gy, TCP = 0")
            return 0.0
    
    # Số lượng alpha bins cho tích phân
    num_bins = 50
    alpha_values = np.linspace(max(0, alpha_mean - 3 * alpha_std), alpha_mean + 3 * alpha_std, num_bins)
    
    # Hàm mật độ xác suất Gaussian cho alpha
    def gaussian_pdf(x, mean, std):
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    alpha_pdf = gaussian_pdf(alpha_values, alpha_mean, alpha_std)
    alpha_pdf = alpha_pdf / np.sum(alpha_pdf)  # Chuẩn hóa
    
    # Tính TCP cho từng giá trị alpha
    tcp_values = np.zeros(num_bins)
    for i, alpha in enumerate(alpha_values):
        # Xác suất sống sót của tế bào cho mỗi voxel
        cell_survival_probability = np.exp(-alpha * tumor_doses)
        
        # Giả sử mỗi voxel chứa cùng số lượng tế bào
        voxel_count = len(tumor_doses)
        cells_per_voxel = 1e5 / voxel_count  # Giả sử tổng số tế bào là 1e5
        
        # TCP dựa trên mô hình Poisson
        surviving_cells = cells_per_voxel * np.sum(cell_survival_probability)
        tcp_values[i] = np.exp(-surviving_cells)
    
    # Tích phân TCP trên phân phối alpha
    tcp = np.sum(tcp_values * alpha_pdf)
    
    return float(tcp)

class TCPModels:
    """
    Lớp cung cấp các phương thức tính toán TCP (Tumor Control Probability).
    """
    
    @staticmethod
    def calculate_tcp_lq_poisson(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        alpha: float = 0.3,
        alpha_beta: float = 10.0,
        clonogenic_density: float = 1e7,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán TCP dựa trên mô hình LQ-Poisson.
        
        Xem hàm calculate_tcp_lq_poisson của module để biết thêm chi tiết.
        """
        return calculate_tcp_lq_poisson(
            dose_array, 
            structure_mask, 
            num_fractions, 
            alpha, 
            alpha_beta, 
            clonogenic_density, 
            dose_threshold
        )
    
    @staticmethod
    def calculate_tcp_lq_poisson_dvh(
        dvh: Dict[str, np.ndarray],
        num_fractions: int,
        alpha: float = 0.3,
        alpha_beta: float = 10.0,
        clonogenic_density: float = 1e7,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán TCP từ DVH (Dose-Volume Histogram) dựa trên mô hình LQ-Poisson.
        
        Xem hàm calculate_tcp_lq_poisson_dvh của module để biết thêm chi tiết.
        """
        return calculate_tcp_lq_poisson_dvh(
            dvh, 
            num_fractions, 
            alpha, 
            alpha_beta, 
            clonogenic_density, 
            dose_threshold
        )
    
    @staticmethod
    def calculate_tcp_niemierko(
        eud: float,
        tcd50: float = 60.0,
        gamma50: float = 2.0
    ) -> float:
        """
        Tính toán TCP dựa trên mô hình Niemierko.
        
        Xem hàm calculate_tcp_niemierko của module để biết thêm chi tiết.
        """
        return calculate_tcp_niemierko(eud, tcd50, gamma50)
    
    @staticmethod
    def calculate_tcp_logistic(
        dose: float,
        tcd50: float = 60.0,
        gamma50: float = 2.0
    ) -> float:
        """
        Tính toán TCP dựa trên mô hình logistic.
        
        Xem hàm calculate_tcp_logistic của module để biết thêm chi tiết.
        """
        return calculate_tcp_logistic(dose, tcd50, gamma50)
    
    @staticmethod
    def calculate_tcp_webb(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        alpha_mean: float = 0.3,
        alpha_std: float = 0.1,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán TCP dựa trên mô hình Webb với tính không đồng nhất của tính nhạy cảm phóng xạ.
        
        Xem hàm calculate_tcp_webb của module để biết thêm chi tiết.
        """
        return calculate_tcp_webb(
            dose_array, 
            structure_mask, 
            alpha_mean, 
            alpha_std, 
            dose_threshold
        )
