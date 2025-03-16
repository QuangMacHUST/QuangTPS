"""
Module tính toán liều tương đương chuẩn hóa (EQD2) và biến đổi liều cho đánh giá sinh học.

Module này cung cấp các hàm để chuyển đổi liều giữa các phân liều và phương pháp xạ trị khác nhau,
bao gồm EQD2, BED, và các chuyển đổi tương ứng để đánh giá tác động sinh học chính xác.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger(__name__)

def calculate_eqd2(
    dose: Union[float, np.ndarray],
    fraction_size: Union[float, np.ndarray],
    alpha_beta: float
) -> Union[float, np.ndarray]:
    """
    Tính toán liều tương đương chuẩn hóa 2Gy (EQD2).

    EQD2 = D * (d + α/β) / (2 + α/β)

    với D là tổng liều, d là liều mỗi phân liều, α/β là tỉ lệ alpha/beta.

    Parameters
    ----------
    dose : Union[float, np.ndarray]
        Tổng liều (Gy)
    fraction_size : Union[float, np.ndarray]
        Liều mỗi phân liều (Gy)
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    Union[float, np.ndarray]
        Liều tương đương chuẩn hóa 2Gy (Gy)
    """
    # Kiểm tra đầu vào
    if isinstance(dose, (list, tuple)):
        dose = np.array(dose)
    if isinstance(fraction_size, (list, tuple)):
        fraction_size = np.array(fraction_size)
    
    # Xử lý trường hợp số phân liều = 0 hoặc tổng liều = 0
    if np.any(dose <= 0) or np.any(fraction_size <= 0):
        if isinstance(dose, np.ndarray) or isinstance(fraction_size, np.ndarray):
            result = np.zeros_like(dose if isinstance(dose, np.ndarray) else fraction_size)
            mask = (dose > 0) & (fraction_size > 0)
            result[mask] = dose[mask] * (fraction_size[mask] + alpha_beta) / (2.0 + alpha_beta)
            return result
        else:
            if dose <= 0 or fraction_size <= 0:
                return 0.0
    
    # Tính EQD2
    return dose * (fraction_size + alpha_beta) / (2.0 + alpha_beta)

def calculate_bed(
    dose: Union[float, np.ndarray],
    fraction_size: Union[float, np.ndarray],
    alpha_beta: float
) -> Union[float, np.ndarray]:
    """
    Tính toán liều sinh học hiệu quả (BED - Biologically Effective Dose).
    
    BED = D * (1 + d/(α/β))
    
    với D là tổng liều, d là liều mỗi phân liều, α/β là tỉ lệ alpha/beta.
    
    Parameters
    ----------
    dose : Union[float, np.ndarray]
        Tổng liều (Gy)
    fraction_size : Union[float, np.ndarray]
        Liều mỗi phân liều (Gy)
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    Union[float, np.ndarray]
        Liều sinh học hiệu quả (Gy)
    """
    # Kiểm tra đầu vào
    if isinstance(dose, (list, tuple)):
        dose = np.array(dose)
    if isinstance(fraction_size, (list, tuple)):
        fraction_size = np.array(fraction_size)
    
    # Xử lý trường hợp số phân liều = 0 hoặc tổng liều = 0
    if np.any(dose <= 0) or np.any(fraction_size <= 0):
        if isinstance(dose, np.ndarray) or isinstance(fraction_size, np.ndarray):
            result = np.zeros_like(dose if isinstance(dose, np.ndarray) else fraction_size)
            mask = (dose > 0) & (fraction_size > 0)
            result[mask] = dose[mask] * (1.0 + fraction_size[mask] / alpha_beta)
            return result
        else:
            if dose <= 0 or fraction_size <= 0:
                return 0.0
    
    # Tính BED
    return dose * (1.0 + fraction_size / alpha_beta)

def calculate_hypo_fractionation(
    eqd2: float,
    num_fractions: int,
    alpha_beta: float
) -> float:
    """
    Tính liều cần thiết cho phân liều giảm số lần chiếu xạ (hypofractionation).
    
    Dùng để tính liều mới khi thay đổi số lần chiếu xạ nhưng muốn giữ cùng hiệu quả sinh học.
    
    Parameters
    ----------
    eqd2 : float
        Liều tương đương chuẩn hóa 2Gy (Gy)
    num_fractions : int
        Số phân liều mới
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    float
        Liều mới (Gy) cần chiếu xạ với số phân liều mới
    """
    if eqd2 <= 0 or num_fractions <= 0:
        return 0.0
    
    # Tính liều mỗi phân liều mới (giải phương trình bậc hai)
    a = 1.0
    b = alpha_beta
    c = -eqd2 * (2.0 + alpha_beta) / num_fractions
    
    # Nghiệm dương của phương trình bậc hai
    d = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    # Tổng liều mới
    new_dose = d * num_fractions
    
    return new_dose

def calculate_bioeffect(
    dose: float,
    fraction_size: float,
    alpha_beta: float,
    alpha: float
) -> float:
    """
    Tính hiệu ứng sinh học của liều xạ trị dựa trên mô hình LQ (Linear-Quadratic).
    
    Bioeffect = -ln(Survival Fraction) = alpha*D + beta*D*d
    
    với D là tổng liều, d là liều mỗi phân liều, alpha và beta là các tham số mô hình LQ.
    
    Parameters
    ----------
    dose : float
        Tổng liều (Gy)
    fraction_size : float
        Liều mỗi phân liều (Gy)
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
    alpha : float
        Tham số alpha trong mô hình LQ (Gy^-1)
        
    Returns
    -------
    float
        Hiệu ứng sinh học (không đơn vị)
    """
    if dose <= 0 or fraction_size <= 0 or alpha <= 0:
        return 0.0
    
    # Tính beta từ alpha và alpha/beta
    beta = alpha / alpha_beta
    
    # Tính hiệu ứng sinh học
    bioeffect = alpha * dose + beta * dose * fraction_size
    
    return bioeffect

def calculate_eqd2_from_dvh(
    dvh: Dict[str, Union[List, np.ndarray]],
    num_fractions: int,
    alpha_beta: float
) -> Dict[str, Union[List, np.ndarray]]:
    """
    Chuyển đổi DVH (Dose-Volume Histogram) sang dạng EQD2.
    
    Parameters
    ----------
    dvh : Dict[str, Union[List, np.ndarray]]
        DVH với các key: 'dose' và 'volume'
    num_fractions : int
        Số phân liều
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    Dict[str, Union[List, np.ndarray]]
        DVH chuyển đổi sang dạng EQD2
    """
    if 'dose' not in dvh or 'volume' not in dvh:
        raise ValueError("DVH must contain 'dose' and 'volume' keys")
    
    # Lấy dose và volume từ DVH
    doses = np.array(dvh['dose'])
    volumes = np.array(dvh['volume'])
    
    if len(doses) != len(volumes):
        raise ValueError(f"Length of dose array ({len(doses)}) must match length of volume array ({len(volumes)})")
    
    if num_fractions <= 0:
        raise ValueError(f"Number of fractions must be positive, got {num_fractions}")
    
    # Tính liều mỗi phân liều
    fraction_size = doses / num_fractions
    
    # Tính EQD2
    eqd2_doses = calculate_eqd2(doses, fraction_size, alpha_beta)
    
    # Tạo DVH mới
    eqd2_dvh = {
        'dose': eqd2_doses,
        'volume': volumes
    }
    
    # Sao chép các metadata khác nếu có
    for key in dvh:
        if key not in ['dose', 'volume']:
            eqd2_dvh[key] = dvh[key]
    
    return eqd2_dvh

def convert_fractionation_schedule(
    old_dose: float,
    old_fractions: int,
    new_fractions: int,
    alpha_beta: float
) -> Tuple[float, float]:
    """
    Chuyển đổi liều giữa các lịch trình phân liều khác nhau, giữ nguyên hiệu ứng sinh học.
    
    Parameters
    ----------
    old_dose : float
        Tổng liều cũ (Gy)
    old_fractions : int
        Số phân liều cũ
    new_fractions : int
        Số phân liều mới
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    Tuple[float, float]
        (Tổng liều mới (Gy), Liều mỗi phân liều mới (Gy))
    """
    if old_dose <= 0 or old_fractions <= 0 or new_fractions <= 0:
        return 0.0, 0.0
    
    # Tính liều mỗi phân liều cũ
    old_fraction_size = old_dose / old_fractions
    
    # Tính EQD2
    eqd2 = calculate_eqd2(old_dose, old_fraction_size, alpha_beta)
    
    # Tính liều mới
    new_dose = calculate_hypo_fractionation(eqd2, new_fractions, alpha_beta)
    
    # Tính liều mỗi phân liều mới
    new_fraction_size = new_dose / new_fractions
    
    return new_dose, new_fraction_size

def bed_to_eqd2(
    bed: Union[float, np.ndarray],
    alpha_beta: float
) -> Union[float, np.ndarray]:
    """
    Chuyển đổi từ BED sang EQD2.

    EQD2 = BED / (1 + 2/(α/β))

    Parameters
    ----------
    bed : Union[float, np.ndarray]
        Giá trị BED (Gy)
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    Union[float, np.ndarray]
        Giá trị EQD2 (Gy)
    """
    if alpha_beta <= 0:
        raise ValueError(f"alpha_beta must be positive, got {alpha_beta}")

    # Chuyển đổi từ BED sang EQD2
    return bed / (1.0 + 2.0 / alpha_beta)

def eqd2_to_bed(
    eqd2: Union[float, np.ndarray],
    alpha_beta: float
) -> Union[float, np.ndarray]:
    """
    Chuyển đổi từ EQD2 sang BED.

    BED = EQD2 * (1 + 2/(α/β))

    Parameters
    ----------
    eqd2 : Union[float, np.ndarray]
        Giá trị EQD2 (Gy)
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    Union[float, np.ndarray]
        Giá trị BED (Gy)
    """
    if alpha_beta <= 0:
        raise ValueError(f"alpha_beta must be positive, got {alpha_beta}")

    # Chuyển đổi từ EQD2 sang BED
    return eqd2 * (1.0 + 2.0 / alpha_beta)

def convert_dose_fractionation(
    dose: float,
    num_fractions_original: int,
    num_fractions_new: int,
    alpha_beta: float
) -> float:
    """
    Chuyển đổi liều giữa các phân liều khác nhau.

    Parameters
    ----------
    dose : float
        Tổng liều gốc (Gy)
    num_fractions_original : int
        Số phân liều gốc
    num_fractions_new : int
        Số phân liều mới
    alpha_beta : float
        Tỉ lệ alpha/beta của mô (Gy)
        
    Returns
    -------
    float
        Liều mới để đạt được hiệu quả sinh học tương đương
    """
    if num_fractions_original <= 0 or num_fractions_new <= 0:
        raise ValueError("Number of fractions must be positive")

    if alpha_beta <= 0:
        raise ValueError(f"alpha_beta must be positive, got {alpha_beta}")

    # Tính liều mỗi phân liều gốc
    fraction_size_original = dose / num_fractions_original

    # Tính EQD2
    eqd2 = calculate_eqd2(dose, fraction_size_original, alpha_beta)

    # Tìm liều mỗi phân liều mới (d') sao cho EQD2(D', d') = EQD2
    # EQD2 = D' * (d' + α/β) / (2 + α/β)
    # Giải phương trình: d' = D' / num_fractions_new
    # Thay vào công thức EQD2 và giải D'

    # Từ: eqd2 = D' * (D'/num_fractions_new + α/β) / (2 + α/β)
    # Ta có: eqd2 * (2 + α/β) = D' * (D'/num_fractions_new + α/β)
    # Sắp xếp lại: D'^2/num_fractions_new + α/β*D' - eqd2*(2 + α/β) = 0
    # Đây là phương trình bậc 2 theo D': aD'^2 + bD' + c = 0

    a = 1.0 / num_fractions_new
    b = alpha_beta
    c = -eqd2 * (2.0 + alpha_beta)

    # Giải phương trình bậc 2
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        raise ValueError("Cannot convert dose: discriminant < 0")

    # Lấy nghiệm dương
    new_dose = (-b + np.sqrt(discriminant)) / (2*a)

    if new_dose < 0:
        raise ValueError("Calculated dose is negative")

    return new_dose

def calculate_eqd2_for_volume(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    num_fractions: int,
    alpha_beta: float = 10.0,
    dose_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Tính toán EQD2 cho một phân bố liều 3D.

    Parameters
    ----------
    dose_array : np.ndarray
        Mảng phân bố liều 3D (Gy)
    structure_mask : np.ndarray
        Mảng mask 3D của cấu trúc (1 nếu thuộc cấu trúc, 0 nếu không)
    num_fractions : int
        Số phân liều
    alpha_beta : float, optional
        Tỉ lệ alpha/beta (Gy), mặc định là 10.0
    dose_threshold : float, optional
        Ngưỡng liều để tính EQD2, voxel có liều < ngưỡng sẽ được đặt thành 0
        
    Returns
    -------
    np.ndarray
        Mảng phân bố EQD2 3D (Gy)
    """
    # Kiểm tra mask và dose có cùng kích thước
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")

    # Tạo mảng kết quả
    eqd2_array = np.zeros_like(dose_array)

    # Chỉ xét các voxel trong cấu trúc
    mask = structure_mask > 0
    if dose_threshold is not None:
        mask = mask & (dose_array >= dose_threshold)

    # Liều mỗi phân liều (Gy)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction_size = dose_array / num_fractions

    # Tính EQD2 cho mỗi voxel
    eqd2_array[mask] = calculate_eqd2(dose_array[mask], fraction_size[mask], alpha_beta)

    return eqd2_array

def calculate_standard_fractionation_equivalent(
    dose: float,
    fractions: int,
    alpha_beta: float = 10.0,
    ref_fraction_size: float = 2.0
) -> Tuple[float, int]:
    """
    Chuyển đổi một kế hoạch phân liều sang phân liều chuẩn có hiệu quả sinh học tương đương.

    Parameters
    ----------
    dose : float
        Tổng liều (Gy)
    fractions : int
        Số phân liều
    alpha_beta : float, optional
        Tỉ lệ alpha/beta (Gy), mặc định là 10.0
    ref_fraction_size : float, optional
        Liều mỗi phân liều chuẩn (Gy), mặc định là 2.0
        
    Returns
    -------
    Tuple[float, int]
        (tổng liều chuẩn, số phân liều chuẩn)
    """
    if fractions <= 0:
        raise ValueError("Number of fractions must be positive")

    if alpha_beta <= 0:
        raise ValueError(f"alpha_beta must be positive, got {alpha_beta}")

    if ref_fraction_size <= 0:
        raise ValueError(f"Reference fraction size must be positive, got {ref_fraction_size}")

    # Tính EQD2
    fraction_size = dose / fractions
    eqd2 = calculate_eqd2(dose, fraction_size, alpha_beta)

    # Tính số phân liều chuẩn
    std_fractions = round(eqd2 / ref_fraction_size)

    # Tính tổng liều chuẩn
    std_dose = std_fractions * ref_fraction_size

    return std_dose, std_fractions

def get_alpha_beta_ratio(tissue_type: str) -> float:
    """
    Lấy tỉ lệ alpha/beta cho một loại mô cụ thể.

    Parameters
    ----------
    tissue_type : str
        Loại mô
        
    Returns
    -------
    float
        Tỉ lệ alpha/beta (Gy)
    """
    # Dict tỉ lệ alpha/beta cho các loại mô khác nhau
    # Giá trị dựa trên các nghiên cứu lâm sàng
    alpha_beta_ratios = {
        # Khối u
        'tumor': 10.0,
        'tumor_head_neck': 10.0,
        'tumor_breast': 4.0,
        'tumor_prostate': 1.5,
        'tumor_lung': 10.0,
        'tumor_colorectal': 5.4,
        'tumor_melanoma': 0.6,
        'tumor_glioblastoma': 5.6,
        'tumor_sarcoma': 5.4,
        
        # Mô lành
        'normal': 3.0,
        'cns': 2.0,
        'brain': 2.0,
        'spinal_cord': 2.0,
        'lung': 3.0,
        'heart': 3.0,
        'kidney': 3.0,
        'liver': 3.0,
        'gi_tract': 3.0,
        'rectum': 3.0,
        'bladder': 4.0,
        'skin': 2.5,
        'bone': 3.0,
        'cartilage': 3.0,
        'mucosa': 7.0,
        'salivary_gland': 3.0,
        'parotid': 3.0,
        'oral_mucosa': 10.0,
        'esophagus': 3.0,
        'small_bowel': 3.0,
        'colon': 3.0,
        'optic_nerve': 3.0,
        'optic_chiasm': 3.0,
        'lens': 1.2,
        'retina': 2.9,
        'lacrimal_gland': 3.0,
        'cochlea': 3.0,
        'pituitary': 3.0,
        'thyroid': 3.0,
        'ovary': 3.0,
        'testis': 3.0
    }

    # Kiểm tra loại mô có trong danh sách không
    tissue_type_lower = tissue_type.lower().replace(' ', '_')
    if tissue_type_lower in alpha_beta_ratios:
        return alpha_beta_ratios[tissue_type_lower]
    else:
        # Trả về giá trị mặc định dựa trên loại mô
        if 'tumor' in tissue_type_lower or 'cancer' in tissue_type_lower or 'target' in tissue_type_lower:
            logger.warning(f"Unknown tumor type '{tissue_type}', using default alpha/beta = 10.0")
            return 10.0
        else:
            logger.warning(f"Unknown normal tissue type '{tissue_type}', using default alpha/beta = 3.0")
            return 3.0

class EQD2Calculator:
    """
    Lớp tính toán liều tương đương chuẩn hóa 2Gy (EQD2) và các chuyển đổi liên quan.
    """
    
    @staticmethod   
    def calculate_eqd2(
        dose: Union[float, np.ndarray],
        fraction_size: Union[float, np.ndarray],
        alpha_beta: float
    ) -> Union[float, np.ndarray]:
        """
        Tính toán liều tương đương chuẩn hóa 2Gy (EQD2).
        
        Xem hàm calculate_eqd2 của module để biết thêm chi tiết.
        """
        return calculate_eqd2(dose, fraction_size, alpha_beta)
    
    @staticmethod
    def calculate_bed(
        dose: Union[float, np.ndarray],
        fraction_size: Union[float, np.ndarray],
        alpha_beta: float
    ) -> Union[float, np.ndarray]:
        """
        Tính toán liều sinh học hiệu quả (BED - Biologically Effective Dose).
        
        Xem hàm calculate_bed của module để biết thêm chi tiết.
        """
        return calculate_bed(dose, fraction_size, alpha_beta)

    @staticmethod
    def calculate_hypo_fractionation(
        eqd2: float,
        num_fractions: int,
        alpha_beta: float
    ) -> float:
        """
        Tính liều cần thiết cho phân liều giảm số lần chiếu xạ (hypofractionation).
        
        Xem hàm calculate_hypo_fractionation của module để biết thêm chi tiết.
        """
        return calculate_hypo_fractionation(eqd2, num_fractions, alpha_beta)
    
    @staticmethod
    def calculate_bioeffect(
        dose: float,
        fraction_size: float,
        alpha_beta: float,
        alpha: float
    ) -> float:
        """
        Tính hiệu ứng sinh học của liều xạ trị dựa trên mô hình LQ (Linear-Quadratic).
        
        Xem hàm calculate_bioeffect của module để biết thêm chi tiết.
        """
        return calculate_bioeffect(dose, fraction_size, alpha_beta, alpha)
    
    @staticmethod
    def calculate_eqd2_from_dvh(
        dvh: Dict[str, Union[List, np.ndarray]],
        num_fractions: int,
        alpha_beta: float
    ) -> Dict[str, Union[List, np.ndarray]]:
        """
        Chuyển đổi DVH (Dose-Volume Histogram) sang dạng EQD2.
        
        Xem hàm calculate_eqd2_from_dvh của module để biết thêm chi tiết.
        """
        return calculate_eqd2_from_dvh(dvh, num_fractions, alpha_beta)
    
    @staticmethod
    def convert_fractionation_schedule(
        old_dose: float,
        old_fractions: int,
        new_fractions: int,
        alpha_beta: float
    ) -> Tuple[float, float]:
        """
        Chuyển đổi liều giữa các lịch trình phân liều khác nhau, giữ nguyên hiệu ứng sinh học.
        
        Xem hàm convert_fractionation_schedule của module để biết thêm chi tiết.
        """
        return convert_fractionation_schedule(old_dose, old_fractions, new_fractions, alpha_beta)

    @staticmethod
    def bed_to_eqd2(
        bed: Union[float, np.ndarray],
        alpha_beta: float
    ) -> Union[float, np.ndarray]:
        """
        Chuyển đổi từ BED sang EQD2.
        
        Xem hàm bed_to_eqd2 của module để biết thêm chi tiết.
        """
        return bed_to_eqd2(bed, alpha_beta)

    @staticmethod
    def eqd2_to_bed(
        eqd2: Union[float, np.ndarray],
        alpha_beta: float
    ) -> Union[float, np.ndarray]:
        """
        Chuyển đổi từ EQD2 sang BED.
        
        Xem hàm eqd2_to_bed của module để biết thêm chi tiết.
        """
        return eqd2_to_bed(eqd2, alpha_beta)

    @staticmethod
    def convert_dose_fractionation(
        dose: float,
        num_fractions_original: int,
        num_fractions_new: int,
        alpha_beta: float
    ) -> float:
        """
        Chuyển đổi liều giữa các phân liều khác nhau.
        
        Xem hàm convert_dose_fractionation của module để biết thêm chi tiết.
        """
        return convert_dose_fractionation(dose, num_fractions_original, num_fractions_new, alpha_beta)

    @staticmethod
    def calculate_eqd2_for_volume(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        alpha_beta: float = 10.0,
        dose_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Tính toán EQD2 cho một phân bố liều 3D.
        
        Xem hàm calculate_eqd2_for_volume của module để biết thêm chi tiết.
        """
        return calculate_eqd2_for_volume(dose_array, structure_mask, num_fractions, alpha_beta, dose_threshold)

    @staticmethod
    def calculate_standard_fractionation_equivalent(
        dose: float,
        fractions: int,
        alpha_beta: float = 10.0,
        ref_fraction_size: float = 2.0
    ) -> Tuple[float, int]:
        """
        Chuyển đổi một kế hoạch phân liều sang phân liều chuẩn có hiệu quả sinh học tương đương.
        
        Xem hàm calculate_standard_fractionation_equivalent của module để biết thêm chi tiết.
        """
        return calculate_standard_fractionation_equivalent(dose, fractions, alpha_beta, ref_fraction_size)

    @staticmethod
    def get_alpha_beta_ratio(tissue_type: str) -> float:
        """
        Lấy tỉ lệ alpha/beta cho một loại mô cụ thể.
        
        Xem hàm get_alpha_beta_ratio của module để biết thêm chi tiết.
        """
        return get_alpha_beta_ratio(tissue_type)
