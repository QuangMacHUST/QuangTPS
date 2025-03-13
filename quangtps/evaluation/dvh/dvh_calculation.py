"""
Module tính toán DVH (Dose Volume Histogram) cho đánh giá kế hoạch xạ trị.

Module này cung cấp các hàm để tính toán biểu đồ liều-thể tích (DVH) từ dữ liệu 
phân bố liều và cấu trúc của bệnh nhân. Hỗ trợ cả DVH tích lũy và vi phân.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

def calculate_dvh(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    dose_bins: Optional[np.ndarray] = None,
    max_dose: Optional[float] = None,
    num_bins: int = 1000,
    dose_unit: str = 'Gy',
    volume_type: str = 'relative',
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Tính toán biểu đồ liều-thể tích từ mảng liều và mask cấu trúc.
    
    Parameters:
        dose_array (np.ndarray): Mảng 3D phân bố liều
        structure_mask (np.ndarray): Mảng boolean 3D chỉ ra vị trí của cấu trúc
        dose_bins (np.ndarray, optional): Các khoảng liều sử dụng cho histogram
        max_dose (float, optional): Liều tối đa để tính DVH, nếu None sẽ lấy từ dose_array
        num_bins (int, optional): Số khoảng liều, mặc định 1000
        dose_unit (str, optional): Đơn vị liều, mặc định 'Gy'
        volume_type (str, optional): Loại thể tích: 'relative' (%) hoặc 'absolute' (cc)
        verbose (bool, optional): In thông tin chi tiết
        
    Returns:
        Dict[str, np.ndarray]: Từ điển chứa:
            - 'differential': DVH vi phân
            - 'cumulative': DVH tích lũy
            - 'dose_bins': Các bin liều
            - 'dose_unit': Đơn vị liều
            - 'volume_type': Loại thể tích
            - 'min_dose': Liều nhỏ nhất trong cấu trúc
            - 'max_dose': Liều lớn nhất trong cấu trúc
            - 'mean_dose': Liều trung bình trong cấu trúc
            - 'median_dose': Liều trung vị trong cấu trúc
            - 'modal_dose': Liều có tần suất cao nhất trong cấu trúc
            - 'structure_volume': Thể tích cấu trúc (cc)
            
    Raises:
        ValueError: Nếu kích thước mảng liều và mask cấu trúc không khớp nhau
    """
    # Kiểm tra kích thước của mảng liều và mask
    if dose_array.shape != structure_mask.shape:
        raise ValueError(f"Dose array shape {dose_array.shape} and structure mask shape {structure_mask.shape} do not match")
    
    # Lấy liều trong cấu trúc
    mask = structure_mask > 0
    structure_doses = dose_array[mask]
    
    # Nếu không có voxel nào trong cấu trúc, trả về DVH rỗng
    if len(structure_doses) == 0:
        logger.warning("No voxels found in structure mask. Returning empty DVH.")
        empty_bins = np.linspace(0, 1, num_bins)
        empty_dvh = np.zeros_like(empty_bins)
        return {
            'differential': empty_dvh,
            'cumulative': empty_dvh,
            'dose_bins': empty_bins,
            'dose_unit': dose_unit,
            'volume_type': volume_type,
            'min_dose': 0.0,
            'max_dose': 0.0,
            'mean_dose': 0.0,
            'median_dose': 0.0,
            'modal_dose': 0.0,
            'structure_volume': 0.0,
        }
    
    # Tính thống kê cơ bản
    min_dose = np.min(structure_doses)
    if max_dose is None:
        max_dose = np.max(structure_doses)
    mean_dose = np.mean(structure_doses)
    median_dose = np.median(structure_doses)
    
    # Tính thể tích cấu trúc (voxel)
    num_voxels = np.sum(mask)
    
    # Tạo các bin liều nếu không được cung cấp
    if dose_bins is None:
        # Thêm một lượng nhỏ để đảm bảo max_dose nằm trong bin cuối cùng
        dose_bins = np.linspace(0, max_dose * 1.001, num_bins)
    
    # Tính histogram
    hist, bin_edges = np.histogram(structure_doses, bins=dose_bins)
    
    # Chuẩn hóa histogram thành thể tích tương đối (%) nếu cần
    if volume_type == 'relative':
        hist = hist / num_voxels * 100
    
    # Tính liều có tần suất cao nhất (modal dose)
    modal_dose_idx = np.argmax(hist)
    modal_dose = (bin_edges[modal_dose_idx] + bin_edges[modal_dose_idx+1]) / 2
    
    # Tạo DVH vi phân
    differential_dvh = hist
    
    # Tạo DVH tích lũy
    cumulative_dvh = np.cumsum(hist[::-1])[::-1]
    
    # Điểm giữa của mỗi bin để biểu diễn
    dose_points = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if verbose:
        logger.info(f"DVH calculated for structure with {num_voxels} voxels")
        logger.info(f"Min dose: {min_dose:.2f} {dose_unit}")
        logger.info(f"Max dose: {max_dose:.2f} {dose_unit}")
        logger.info(f"Mean dose: {mean_dose:.2f} {dose_unit}")
        logger.info(f"Median dose: {median_dose:.2f} {dose_unit}")
    
    return {
        'differential': differential_dvh,
        'cumulative': cumulative_dvh,
        'dose_bins': dose_points,
        'dose_unit': dose_unit,
        'volume_type': volume_type,
        'min_dose': min_dose,
        'max_dose': max_dose,
        'mean_dose': mean_dose,
        'median_dose': median_dose,
        'modal_dose': modal_dose,
        'structure_volume': num_voxels,
    }

def calculate_dvh_metrics(
    dvh_data: Dict[str, Any],
    metrics: List[str] = None,
    dose_rx: Optional[float] = None
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá từ dữ liệu DVH.
    
    Parameters:
        dvh_data (Dict[str, Any]): Dữ liệu DVH từ hàm calculate_dvh
        metrics (List[str], optional): Danh sách các chỉ số cần tính
        dose_rx (float, optional): Liều kê đơn, cần thiết cho một số chỉ số
        
    Returns:
        Dict[str, float]: Từ điển chứa các chỉ số và giá trị tương ứng
        
    Raises:
        ValueError: Nếu metric không được hỗ trợ hoặc không có đủ dữ liệu
    """
    # Khởi tạo danh sách metrics mặc định nếu không được cung cấp
    if metrics is None:
        metrics = ['D95', 'D98', 'D50', 'D2', 'V95', 'V100', 'V105']
    
    # Chuẩn bị dữ liệu
    cumulative_dvh = dvh_data['cumulative']
    dose_bins = dvh_data['dose_bins']
    volume_type = dvh_data['volume_type']
    
    # Chuyển đổi sang đơn vị tương đối nếu cần
    if volume_type == 'absolute':
        rel_cumulative_dvh = cumulative_dvh / dvh_data['structure_volume'] * 100
    else:
        rel_cumulative_dvh = cumulative_dvh
    
    # Kết quả
    results = {}
    
    # Tính toán các chỉ số
    for metric in metrics:
        if metric.startswith('D'):
            # Dx - Liều phủ x% thể tích
            try:
                volume_percent = float(metric[1:])
                if volume_percent < 0 or volume_percent > 100:
                    raise ValueError(f"Invalid volume percentage {volume_percent} for metric {metric}")
                
                # Tìm liều tương ứng với thể tích
                results[metric] = _get_dose_at_volume(dose_bins, rel_cumulative_dvh, volume_percent)
                
            except ValueError as e:
                logger.warning(f"Could not calculate {metric}: {str(e)}")
                results[metric] = float('nan')
                
        elif metric.startswith('V') and dose_rx is not None:
            # Vx - Phần trăm thể tích nhận liều ≥ x% của liều kê đơn
            try:
                dose_percent = float(metric[1:])
                if dose_percent < 0:
                    raise ValueError(f"Invalid dose percentage {dose_percent} for metric {metric}")
                
                # Tính liều tương ứng với phần trăm của liều kê đơn
                target_dose = dose_rx * dose_percent / 100
                
                # Tìm thể tích tương ứng với liều
                results[metric] = _get_volume_at_dose(dose_bins, rel_cumulative_dvh, target_dose)
                
            except ValueError as e:
                logger.warning(f"Could not calculate {metric}: {str(e)}")
                results[metric] = float('nan')
        
        # Thêm các thống kê cơ bản từ dvh_data
        elif metric in ['min_dose', 'max_dose', 'mean_dose', 'median_dose', 'modal_dose']:
            results[metric] = dvh_data[metric]
        
        else:
            logger.warning(f"Metric {metric} not supported or dose_rx not provided")
            results[metric] = float('nan')
    
    return results

def _get_dose_at_volume(dose_bins: np.ndarray, cumulative_dvh: np.ndarray, volume_percent: float) -> float:
    """
    Lấy giá trị liều tại thể tích phần trăm cụ thể từ DVH tích lũy.
    
    Parameters:
        dose_bins (np.ndarray): Các bin liều
        cumulative_dvh (np.ndarray): DVH tích lũy
        volume_percent (float): Phần trăm thể tích cần tìm
        
    Returns:
        float: Giá trị liều tại phần trăm thể tích (nội suy tuyến tính)
    """
    # Tìm điểm cắt với volume_percent
    # Chú ý DVH tích lũy có thể hiểu là: phần trăm thể tích nhận liều >= x
    f = np.interp(volume_percent, cumulative_dvh[::-1], dose_bins[::-1])
    return f

def _get_volume_at_dose(dose_bins: np.ndarray, cumulative_dvh: np.ndarray, dose: float) -> float:
    """
    Lấy phần trăm thể tích nhận liều >= giá trị liều cụ thể từ DVH tích lũy.
    
    Parameters:
        dose_bins (np.ndarray): Các bin liều
        cumulative_dvh (np.ndarray): DVH tích lũy
        dose (float): Giá trị liều cần tìm
        
    Returns:
        float: Phần trăm thể tích nhận liều >= dose (nội suy tuyến tính)
    """
    # Tìm phần trăm thể tích tại dose
    f = np.interp(dose, dose_bins, cumulative_dvh)
    return f

def calculate_dvh_from_dose_grid(
    dose_grid: DoseGrid,
    structure_mask: np.ndarray,
    num_bins: int = 1000,
    dose_unit: str = 'Gy',
    volume_type: str = 'relative'
) -> Dict[str, np.ndarray]:
    """
    Tính toán DVH từ đối tượng DoseGrid và mask cấu trúc.
    
    Parameters:
        dose_grid (DoseGrid): Đối tượng DoseGrid chứa phân bố liều
        structure_mask (np.ndarray): Mảng boolean 3D chỉ ra vị trí của cấu trúc
        num_bins (int, optional): Số khoảng liều, mặc định 1000
        dose_unit (str, optional): Đơn vị liều, mặc định 'Gy'
        volume_type (str, optional): Loại thể tích: 'relative' (%) hoặc 'absolute' (cc)
        
    Returns:
        Dict[str, np.ndarray]: Dữ liệu DVH
    """
    # Lấy mảng liều từ dose_grid
    dose_array = dose_grid.get_dose_matrix()
    
    # Chuyển đổi mask cấu trúc nếu cần (đảm bảo cùng kích thước với dose_array)
    if structure_mask.shape != dose_array.shape:
        logger.warning(f"Structure mask shape {structure_mask.shape} does not match dose grid shape {dose_array.shape}")
        # Cần triển khai phương pháp chuyển đổi mask cấu trúc ở đây
        # ...
    
    # Tính DVH
    dvh_data = calculate_dvh(
        dose_array=dose_array,
        structure_mask=structure_mask,
        num_bins=num_bins,
        dose_unit=dose_unit,
        volume_type=volume_type
    )
    
    # Thêm kích thước voxel để tính thể tích thực (cc)
    voxel_size = dose_grid.get_voxel_size()  # mm
    voxel_volume_cc = np.prod(voxel_size) / 1000  # chuyển từ mm³ sang cc
    
    # Cập nhật thể tích cấu trúc
    dvh_data['structure_volume_cc'] = dvh_data['structure_volume'] * voxel_volume_cc
    
    # Cập nhật DVH tuyệt đối nếu cần
    if volume_type == 'absolute':
        dvh_data['differential'] = dvh_data['differential'] * voxel_volume_cc
        dvh_data['cumulative'] = dvh_data['cumulative'] * voxel_volume_cc
    
    return dvh_data

def merge_dvhs(
    dvh_list: List[Dict[str, np.ndarray]],
    weights: Optional[List[float]] = None
) -> Dict[str, np.ndarray]:
    """
    Kết hợp nhiều DVH với trọng số tùy chọn.
    
    Parameters:
        dvh_list (List[Dict]): Danh sách các dvh để kết hợp
        weights (List[float], optional): Trọng số tương ứng, nếu None thì trọng số bằng nhau
        
    Returns:
        Dict[str, np.ndarray]: DVH sau khi kết hợp
        
    Raises:
        ValueError: Nếu trọng số không hợp lệ hoặc DVH không tương thích
    """
    if not dvh_list:
        raise ValueError("Empty dvh_list provided")
    
    n_dvhs = len(dvh_list)
    
    # Kiểm tra weights
    if weights is None:
        weights = [1.0 / n_dvhs] * n_dvhs
    else:
        if len(weights) != n_dvhs:
            raise ValueError(f"Number of weights ({len(weights)}) does not match number of DVHs ({n_dvhs})")
        # Chuẩn hóa weights để tổng = 1
        weights = np.array(weights) / np.sum(weights)
    
    # Xác nhận rằng tất cả DVH có cùng dose_bins
    ref_dvh = dvh_list[0]
    ref_bins = ref_dvh['dose_bins']
    
    for i, dvh in enumerate(dvh_list[1:], start=1):
        if not np.array_equal(dvh['dose_bins'], ref_bins):
            logger.warning(f"DVH {i} has different dose bins. Resampling...")
            # Cần triển khai phương pháp resampling DVH ở đây
            # ...
    
    # Tính DVH kết hợp
    merged_differential = np.zeros_like(ref_dvh['differential'])
    merged_cumulative = np.zeros_like(ref_dvh['cumulative'])
    
    total_volume = 0
    min_doses = []
    max_doses = []
    mean_doses = []
    median_doses = []
    
    for i, dvh in enumerate(dvh_list):
        weight = weights[i]
        if dvh['volume_type'] == 'relative':
            # Lấy thông tin thể tích
            structure_volume = dvh['structure_volume']
            total_volume += structure_volume
            
            # Kết hợp DVH với trọng số
            merged_differential += dvh['differential'] * weight * structure_volume
            merged_cumulative += dvh['cumulative'] * weight * structure_volume
        else:
            # DVH tuyệt đối, cần chuyển đổi
            logger.warning("DVH with absolute volume needs special handling")
            # ...
        
        # Thu thập thông tin thống kê
        min_doses.append(dvh['min_dose'])
        max_doses.append(dvh['max_dose'])
        mean_doses.append(dvh['mean_dose'])
        median_doses.append(dvh['median_dose'])
    
    # Chuẩn hóa lại thành tương đối hoặc tuyệt đối
    if ref_dvh['volume_type'] == 'relative':
        merged_differential = merged_differential / total_volume * 100
        merged_cumulative = merged_cumulative / total_volume * 100
    
    # Tạo DVH kết hợp
    merged_dvh = {
        'differential': merged_differential,
        'cumulative': merged_cumulative,
        'dose_bins': ref_bins,
        'dose_unit': ref_dvh['dose_unit'],
        'volume_type': ref_dvh['volume_type'],
        'min_dose': np.min(min_doses),
        'max_dose': np.max(max_doses),
        'mean_dose': np.average(mean_doses, weights=weights),
        'median_dose': np.average(median_doses, weights=weights),  # Gần đúng
        'structure_volume': total_volume,
    }
    
    return merged_dvh

def subtract_dvhs(
    dvh1: Dict[str, np.ndarray],
    dvh2: Dict[str, np.ndarray],
    absolute_difference: bool = False
) -> Dict[str, np.ndarray]:
    """
    Tính hiệu của hai DVH.
    
    Parameters:
        dvh1 (Dict): DVH thứ nhất
        dvh2 (Dict): DVH thứ hai
        absolute_difference (bool): Nếu True, trả về giá trị tuyệt đối của hiệu
        
    Returns:
        Dict[str, np.ndarray]: DVH hiệu
    """
    # Kiểm tra xem hai DVH có cùng bin không
    if not np.array_equal(dvh1['dose_bins'], dvh2['dose_bins']):
        raise ValueError("DVHs must have the same dose bins for subtraction")
    
    # Kiểm tra thể tích tương đối vs tuyệt đối
    if dvh1['volume_type'] != dvh2['volume_type']:
        raise ValueError("DVHs must have the same volume type (relative or absolute)")
    
    # Tính hiệu
    if absolute_difference:
        diff_differential = np.abs(dvh1['differential'] - dvh2['differential'])
        diff_cumulative = np.abs(dvh1['cumulative'] - dvh2['cumulative'])
    else:
        diff_differential = dvh1['differential'] - dvh2['differential']
        diff_cumulative = dvh1['cumulative'] - dvh2['cumulative']
    
    # Tạo DVH hiệu
    diff_dvh = {
        'differential': diff_differential,
        'cumulative': diff_cumulative,
        'dose_bins': dvh1['dose_bins'],
        'dose_unit': dvh1['dose_unit'],
        'volume_type': dvh1['volume_type'],
        'min_dose': min(dvh1['min_dose'], dvh2['min_dose']),
        'max_dose': max(dvh1['max_dose'], dvh2['max_dose']),
        'mean_dose': abs(dvh1['mean_dose'] - dvh2['mean_dose']) if absolute_difference else dvh1['mean_dose'] - dvh2['mean_dose'],
        'median_dose': abs(dvh1['median_dose'] - dvh2['median_dose']) if absolute_difference else dvh1['median_dose'] - dvh2['median_dose'],
        'structure_volume': dvh1['structure_volume'],  # Giả sử cùng cấu trúc
    }
    
    return diff_dvh
