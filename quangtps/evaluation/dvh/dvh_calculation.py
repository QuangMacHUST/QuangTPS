#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp các hàm tính toán DVH (Dose Volume Histogram) cơ bản.

Module này là cầu nối giữa DVHCalculator và DVHAnalysis, cung cấp các hàm helper
để tính toán và phân tích DVH.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator, DVHData

logger = logging.getLogger(__name__)

def calculate_dvh(dose_array: np.ndarray, structure_mask: np.ndarray, 
                 num_bins: int = 1000, cumulative: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính toán DVH từ mảng liều và mặt nạ cấu trúc.
    
    Parameters
    ----------
    dose_array : np.ndarray
        Mảng phân bố liều
    structure_mask : np.ndarray
        Mặt nạ cấu trúc (1 cho vùng trong cấu trúc, 0 cho vùng ngoài)
    num_bins : int
        Số lượng bin sử dụng cho DVH
    cumulative : bool
        Nếu True, tính DVH tích lũy, ngược lại tính DVH vi phân
        
    Returns
    -------
    tuple
        (dose_bins, volume_bins)
    """
    # Chuyển đổi mảng numpy thành hình ảnh SimpleITK
    dose_image = sitk.GetImageFromArray(dose_array)
    roi_mask = sitk.GetImageFromArray(structure_mask)
    
    # Sử dụng DVHCalculator để tính toán DVH
    calculator = DVHCalculator(num_bins=num_bins)
    dose_bins, volume_bins = calculator.calculate_dvh(dose_image, roi_mask, cumulative)
    
    return dose_bins, volume_bins

def _get_dose_at_volume(dose_bins: np.ndarray, volume_bins: np.ndarray, 
                      percent_volume: float) -> float:
    """
    Lấy giá trị liều tại một phần trăm thể tích nhất định từ dữ liệu DVH.
    
    Parameters
    ----------
    dose_bins : np.ndarray
        Mảng giá trị liều từ DVH
    volume_bins : np.ndarray
        Mảng giá trị thể tích tương ứng từ DVH
    percent_volume : float
        Phần trăm thể tích (0-100)
        
    Returns
    -------
    float
        Liều (Gy) tại phần trăm thể tích đã chỉ định
    """
    if percent_volume < 0 or percent_volume > 100:
        raise ValueError("Percent volume must be between 0 and 100")
    
    # Chuyển đổi phần trăm thành tỷ lệ
    volume_fraction = percent_volume / 100.0
    
    # Nếu volume_bins là phần trăm (0-100), chuyển đổi thành tỷ lệ (0-1)
    if np.max(volume_bins) > 1.0:
        normalized_vol_bins = volume_bins / 100.0
    else:
        normalized_vol_bins = volume_bins
    
    # Nội suy để tìm liều tại thể tích chỉ định
    if volume_fraction <= np.min(normalized_vol_bins):
        return np.max(dose_bins)
    elif volume_fraction >= np.max(normalized_vol_bins):
        return np.min(dose_bins)
    
    # Nội suy tuyến tính
    idx = np.interp(volume_fraction, normalized_vol_bins[::-1], np.arange(len(normalized_vol_bins))[::-1])
    idx_floor = int(np.floor(idx))
    idx_ceil = int(np.ceil(idx))
    
    if idx_floor == idx_ceil:
        return dose_bins[idx_floor]
    
    # Nội suy tuyến tính giữa hai điểm lân cận
    weight_ceil = idx - idx_floor
    weight_floor = 1.0 - weight_ceil
    
    dose = weight_floor * dose_bins[idx_floor] + weight_ceil * dose_bins[idx_ceil]
    
    return dose

def _get_volume_at_dose(dose_bins: np.ndarray, volume_bins: np.ndarray, 
                       dose: float, as_percent: bool = True) -> float:
    """
    Lấy thể tích nhận ít nhất một liều cụ thể từ dữ liệu DVH.
    
    Parameters
    ----------
    dose_bins : np.ndarray
        Mảng giá trị liều từ DVH
    volume_bins : np.ndarray
        Mảng giá trị thể tích tương ứng từ DVH
    dose : float
        Ngưỡng liều (Gy)
    as_percent : bool
        Nếu True, trả về thể tích theo phần trăm
        Nếu False, trả về thể tích tuyệt đối
        
    Returns
    -------
    float
        Thể tích (theo % hoặc đơn vị tuyệt đối) nhận ít nhất liều chỉ định
    """
    if dose < np.min(dose_bins) or dose > np.max(dose_bins):
        if dose < np.min(dose_bins):
            return volume_bins[0]  # Toàn bộ thể tích nếu liều thấp hơn liều tối thiểu
        else:
            return 0.0  # Không có thể tích nếu liều cao hơn liều tối đa
    
    # Nội suy để tìm thể tích tại liều chỉ định
    idx = np.interp(dose, dose_bins, np.arange(len(dose_bins)))
    idx_floor = int(np.floor(idx))
    idx_ceil = int(np.ceil(idx))
    
    if idx_floor == idx_ceil:
        volume = volume_bins[idx_floor]
    else:
        # Nội suy tuyến tính giữa hai điểm lân cận
        weight_ceil = idx - idx_floor
        weight_floor = 1.0 - weight_ceil
        
        volume = weight_floor * volume_bins[idx_floor] + weight_ceil * volume_bins[idx_ceil]
    
    # Trả về kết quả theo định dạng yêu cầu
    return volume

def calculate_dvh_metrics(dvh_data: Dict[str, Any], metrics: List[str] = None, 
                          rx_dose: Optional[float] = None) -> Dict[str, float]:
    """
    Tính toán các chỉ số metrics từ dữ liệu DVH.
    
    Parameters
    ----------
    dvh_data : Dict[str, Any]
        Dữ liệu DVH cần phân tích
    metrics : List[str], optional
        Danh sách các metrics cần tính. Nếu None, sử dụng danh sách mặc định
    rx_dose : float, optional
        Liều kê đơn (Gy), cần thiết cho một số metrics
        
    Returns
    -------
    Dict[str, float]
        Dictionary chứa các metrics đã tính
    """
    if metrics is None:
        # Danh sách metrics mặc định
        metrics = ['Dmax', 'Dmean', 'Dmin', 'D95', 'D90', 'D50', 'V95', 'V90', 'V50']
    
    result = {}
    
    # Lấy dữ liệu DVH cần thiết
    dose_bins = dvh_data.get('dose_bins', [])
    volume_bins = dvh_data.get('volume_bins', [])
    structure_volume = dvh_data.get('structure_volume', 0)
    
    # Đảm bảo dữ liệu hợp lệ
    if len(dose_bins) == 0 or len(volume_bins) == 0 or len(dose_bins) != len(volume_bins):
        logger.warning("Dữ liệu DVH không hợp lệ để tính metrics")
        return result
    
    # Tính các metrics cơ bản
    for metric in metrics:
        if metric.startswith('D') and metric[1:].replace('.', '', 1).isdigit():
            # Metrics dạng Dx (liều ở x% thể tích)
            try:
                percent = float(metric[1:])
                result[metric] = _get_dose_at_volume(dose_bins, volume_bins, percent)
            except (ValueError, IndexError) as e:
                logger.warning(f"Không thể tính {metric}: {str(e)}")
                result[metric] = float('nan')
                
        elif metric.startswith('V') and metric[1:].replace('.', '', 1).isdigit():
            # Metrics dạng Vx (thể tích nhận ít nhất x Gy)
            try:
                dose = float(metric[1:])
                # Nếu có liều kê đơn và định dạng Vx%, chuyển sang liều tuyệt đối
                if rx_dose is not None and "%" in metric:
                    percent = float(metric[1:-1])  # bỏ 'V' và '%'
                    dose = rx_dose * percent / 100
                
                result[metric] = _get_volume_at_dose(dose_bins, volume_bins, dose, True)
            except (ValueError, IndexError) as e:
                logger.warning(f"Không thể tính {metric}: {str(e)}")
                result[metric] = float('nan')
                
        elif metric == 'Dmax':
            result[metric] = np.max(dose_bins) if len(dose_bins) > 0 else float('nan')
            
        elif metric == 'Dmean':
            # Tính liều trung bình từ histogram
            if 'mean_dose' in dvh_data:
                result[metric] = dvh_data['mean_dose']
            else:
                # Tính xấp xỉ từ dose_bins và volume_bins
                total_dose = 0
                total_volume = 0
                for i in range(len(dose_bins) - 1):
                    vol_diff = volume_bins[i] - volume_bins[i + 1]
                    if vol_diff > 0:
                        avg_dose = (dose_bins[i] + dose_bins[i + 1]) / 2
                        total_dose += avg_dose * vol_diff
                        total_volume += vol_diff
                
                result[metric] = total_dose / total_volume if total_volume > 0 else float('nan')
                
        elif metric == 'Dmin':
            # Tìm liều tối thiểu trong vùng không phải 0
            non_zero_doses = [d for i, d in enumerate(dose_bins) if volume_bins[i] > 0]
            result[metric] = min(non_zero_doses) if non_zero_doses else float('nan')
    
    # Tính metrics liên quan đến liều kê đơn nếu có
    if rx_dose is not None:
        if 'V_rx' in metrics:
            result['V_rx'] = _get_volume_at_dose(dose_bins, volume_bins, rx_dose, True)
            
        if 'V_rx_percent' in metrics:
            v_rx = _get_volume_at_dose(dose_bins, volume_bins, rx_dose, False)
            result['V_rx_percent'] = 100 * v_rx / structure_volume if structure_volume > 0 else float('nan')
    
    return result

def calculate_dvh_from_dose_grid(
    dose_grid: np.ndarray,
    structure_mask: np.ndarray,
    num_bins: int = 100,
    volume_type: str = 'relative'
) -> Dict[str, Any]:
    """
    Tính toán DVH từ lưới liều (dose grid) và mặt nạ cấu trúc.
    
    Parameters
    ----------
    dose_grid : np.ndarray
        Mảng chứa dữ liệu liều (Gy)
    structure_mask : np.ndarray
        Mặt nạ nhị phân của cấu trúc (1 trong cấu trúc, 0 ngoài cấu trúc)
    num_bins : int, optional
        Số lượng bin cho histogram
    volume_type : str, optional
        'relative' để xuất thể tích theo %, 'absolute' để xuất thể tích theo cc
        
    Returns
    -------
    Dict[str, Any]
        Dictionary chứa dữ liệu DVH, bao gồm dose_bins, volume_bins, min_dose, mean_dose, max_dose, etc.
    """
    # Kiểm tra đầu vào
    if dose_grid.shape != structure_mask.shape:
        raise ValueError(f"Kích thước dose_grid {dose_grid.shape} và structure_mask {structure_mask.shape} không khớp")
    
    # Tạo mặt nạ cấu trúc boolean
    mask_bool = structure_mask > 0
    
    # Lấy giá trị liều trong cấu trúc
    structure_doses = dose_grid[mask_bool]
    
    if len(structure_doses) == 0:
        logger.warning("Không có voxel nào trong cấu trúc")
        return {
            'dose_bins': np.array([]),
            'volume_bins': np.array([]),
            'min_dose': 0.0,
            'mean_dose': 0.0,
            'max_dose': 0.0,
            'structure_volume': 0.0,
        }
    
    # Tính thể tích cấu trúc (tổng số voxel)
    structure_volume = np.sum(mask_bool)
    
    # Tính các thống kê cơ bản
    min_dose = np.min(structure_doses) if len(structure_doses) > 0 else 0.0
    mean_dose = np.mean(structure_doses) if len(structure_doses) > 0 else 0.0
    max_dose = np.max(structure_doses) if len(structure_doses) > 0 else 0.0
    
    # Tạo bin liều
    if max_dose > min_dose:
        dose_bins = np.linspace(0, max_dose * 1.05, num_bins)
    else:
        # Tránh trường hợp liều đồng nhất
        dose_bins = np.linspace(0, mean_dose * 2, num_bins)
    
    # Tính histogram
    hist, bin_edges = np.histogram(structure_doses, bins=dose_bins)
    
    # Chuyển đổi sang tích lũy (cumulative)
    cum_hist = np.cumsum(hist[::-1])[::-1]
    
    # Chuẩn hóa theo loại thể tích
    if volume_type.lower() == 'relative':
        volume_bins = cum_hist / structure_volume * 100.0  # Theo phần trăm
    else:
        # Nếu có thông tin voxel size, nhân với thể tích thực, hoặc giả định 1mm³ mỗi voxel
        voxel_volume = 1.0  # mm³, giả định
        volume_bins = cum_hist * voxel_volume / 1000.0  # Chuyển thành cc (cm³)
    
    # Tạo kết quả
    dvh_data = {
        'dose_bins': dose_bins[:-1],  # Bỏ bin cuối từ bin_edges
        'volume_bins': volume_bins,
        'min_dose': min_dose,
        'mean_dose': mean_dose,
        'max_dose': max_dose,
        'structure_volume': structure_volume,
        'is_cumulative': True
    }
    
    return dvh_data
