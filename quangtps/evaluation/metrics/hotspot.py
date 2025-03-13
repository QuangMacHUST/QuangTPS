#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các phương thức phân tích điểm nóng (hotspot) và điểm lạnh (coldspot) 
trong phân phối liều xạ trị.
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional
import scipy.ndimage as ndimage


class HotspotAnalysis:
    """
    Lớp phân tích và đánh giá các điểm nóng (hotspot) và điểm lạnh (coldspot) trong phân phối liều.
    Hotspot là vùng nhận liều cao hơn đáng kể so với liều kê.
    Coldspot là vùng nhận liều thấp hơn đáng kể so với liều kê.
    """

    @staticmethod
    def identify_hotspots(dose_array: np.ndarray, reference_dose: float,
                       threshold_percent: float = 110.0, min_volume_cc: float = 0.1,
                       voxel_size_cc: float = 0.001) -> List[Dict]:
        """
        Xác định các điểm nóng trong phân phối liều 3D
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        reference_dose : float
            Liều tham chiếu (thường là liều kê), đơn vị Gy
        threshold_percent : float, optional
            Ngưỡng phần trăm liều để xác định điểm nóng, mặc định 110% liều tham chiếu
        min_volume_cc : float, optional
            Thể tích tối thiểu để coi là điểm nóng có ý nghĩa, đơn vị cm³
        voxel_size_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
            
        Returns
        -------
        List[Dict]
            Danh sách các điểm nóng được phát hiện, mỗi điểm là một từ điển chứa:
            - volume: thể tích của điểm nóng (cm³)
            - max_dose: liều tối đa trong điểm nóng (Gy)
            - mean_dose: liều trung bình trong điểm nóng (Gy)
            - location: tọa độ trung tâm của điểm nóng (x, y, z) (theo chỉ số)
        """
        # Tính ngưỡng liều để xác định điểm nóng
        threshold_dose = reference_dose * threshold_percent / 100.0
        
        # Tạo mask cho vùng vượt ngưỡng
        hotspot_mask = dose_array > threshold_dose
        
        # Đặt nhãn cho các vùng liên thông trong mask
        labeled_array, num_features = ndimage.label(hotspot_mask)
        
        # Tìm thuộc tính của từng vùng liên thông
        hotspots = []
        
        if num_features > 0:
            # Đối với mỗi vùng liên thông
            for i in range(1, num_features + 1):
                # Tạo mask cho vùng liên thông hiện tại
                region_mask = labeled_array == i
                
                # Tính thể tích
                volume_cc = np.sum(region_mask) * voxel_size_cc
                
                # Chỉ xem xét các vùng có thể tích đủ lớn
                if volume_cc >= min_volume_cc:
                    # Lấy giá trị liều trong vùng
                    region_doses = dose_array[region_mask]
                    
                    # Tính liều tối đa và trung bình
                    max_dose = np.max(region_doses)
                    mean_dose = np.mean(region_doses)
                    
                    # Tìm tọa độ trung tâm của vùng
                    coords = np.array(np.where(region_mask)).T
                    center_coords = np.mean(coords, axis=0).astype(int)
                    
                    # Thêm thông tin vào danh sách
                    hotspots.append({
                        'volume': volume_cc,
                        'max_dose': float(max_dose),
                        'mean_dose': float(mean_dose),
                        'location': tuple(center_coords),
                        'percent_over_reference': float((max_dose / reference_dose - 1) * 100)
                    })
        
        # Sắp xếp điểm nóng theo thể tích giảm dần
        hotspots.sort(key=lambda x: x['volume'], reverse=True)
        
        return hotspots

    @staticmethod
    def identify_coldspots(dose_array: np.ndarray, reference_dose: float,
                        threshold_percent: float = 90.0, min_volume_cc: float = 0.1,
                        voxel_size_cc: float = 0.001, target_mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Xác định các điểm lạnh trong phân phối liều 3D
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        reference_dose : float
            Liều tham chiếu (thường là liều kê), đơn vị Gy
        threshold_percent : float, optional
            Ngưỡng phần trăm liều để xác định điểm lạnh, mặc định 90% liều tham chiếu
        min_volume_cc : float, optional
            Thể tích tối thiểu để coi là điểm lạnh có ý nghĩa, đơn vị cm³
        voxel_size_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
        target_mask : np.ndarray, optional
            Mask của thể tích mục tiêu, nếu None thì sẽ xét toàn bộ mảng liều
            
        Returns
        -------
        List[Dict]
            Danh sách các điểm lạnh được phát hiện, mỗi điểm là một từ điển chứa:
            - volume: thể tích của điểm lạnh (cm³)
            - min_dose: liều tối thiểu trong điểm lạnh (Gy)
            - mean_dose: liều trung bình trong điểm lạnh (Gy)
            - location: tọa độ trung tâm của điểm lạnh (x, y, z) (theo chỉ số)
        """
        # Tính ngưỡng liều để xác định điểm lạnh
        threshold_dose = reference_dose * threshold_percent / 100.0
        
        # Tạo mask cho vùng dưới ngưỡng
        if target_mask is not None:
            # Chỉ xét trong thể tích mục tiêu
            coldspot_mask = (dose_array < threshold_dose) & target_mask
        else:
            # Xét toàn bộ mảng liều
            coldspot_mask = dose_array < threshold_dose
        
        # Đặt nhãn cho các vùng liên thông trong mask
        labeled_array, num_features = ndimage.label(coldspot_mask)
        
        # Tìm thuộc tính của từng vùng liên thông
        coldspots = []
        
        if num_features > 0:
            # Đối với mỗi vùng liên thông
            for i in range(1, num_features + 1):
                # Tạo mask cho vùng liên thông hiện tại
                region_mask = labeled_array == i
                
                # Tính thể tích
                volume_cc = np.sum(region_mask) * voxel_size_cc
                
                # Chỉ xem xét các vùng có thể tích đủ lớn
                if volume_cc >= min_volume_cc:
                    # Lấy giá trị liều trong vùng
                    region_doses = dose_array[region_mask]
                    
                    # Tính liều tối thiểu và trung bình
                    min_dose = np.min(region_doses)
                    mean_dose = np.mean(region_doses)
                    
                    # Tìm tọa độ trung tâm của vùng
                    coords = np.array(np.where(region_mask)).T
                    center_coords = np.mean(coords, axis=0).astype(int)
                    
                    # Thêm thông tin vào danh sách
                    coldspots.append({
                        'volume': volume_cc,
                        'min_dose': float(min_dose),
                        'mean_dose': float(mean_dose),
                        'location': tuple(center_coords),
                        'percent_under_reference': float((1 - min_dose / reference_dose) * 100)
                    })
        
        # Sắp xếp điểm lạnh theo thể tích giảm dần
        coldspots.sort(key=lambda x: x['volume'], reverse=True)
        
        return coldspots

    @staticmethod
    def calculate_hotspot_volume_histograms(dose_array: np.ndarray, reference_dose: float,
                                         thresholds: List[float] = None,
                                         voxel_size_cc: float = 0.001) -> Dict[str, float]:
        """
        Tính toán thể tích nhận liều vượt ngưỡng theo các ngưỡng khác nhau
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        reference_dose : float
            Liều tham chiếu (thường là liều kê), đơn vị Gy
        thresholds : List[float], optional
            Danh sách các ngưỡng phần trăm để tính toán
            Mặc định: [100, 105, 110, 115, 120, 125, 130, 140, 150]
        voxel_size_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa thể tích (cm³) cho mỗi ngưỡng liều
        """
        if thresholds is None:
            thresholds = [100, 105, 110, 115, 120, 125, 130, 140, 150]
        
        histogram = {}
        
        for threshold in thresholds:
            threshold_dose = reference_dose * threshold / 100.0
            volume = np.sum(dose_array > threshold_dose) * voxel_size_cc
            histogram[f"V{threshold}%"] = volume
        
        return histogram

    @staticmethod
    def calculate_coldspot_volume_histograms(dose_array: np.ndarray, reference_dose: float,
                                          thresholds: List[float] = None,
                                          voxel_size_cc: float = 0.001,
                                          target_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Tính toán thể tích nhận liều dưới ngưỡng theo các ngưỡng khác nhau
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        reference_dose : float
            Liều tham chiếu (thường là liều kê), đơn vị Gy
        thresholds : List[float], optional
            Danh sách các ngưỡng phần trăm để tính toán
            Mặc định: [95, 90, 85, 80, 70, 60, 50]
        voxel_size_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
        target_mask : np.ndarray, optional
            Mask của thể tích mục tiêu, nếu None thì sẽ xét toàn bộ mảng liều
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa thể tích (cm³) cho mỗi ngưỡng liều
        """
        if thresholds is None:
            thresholds = [95, 90, 85, 80, 70, 60, 50]
        
        histogram = {}
        
        for threshold in thresholds:
            threshold_dose = reference_dose * threshold / 100.0
            
            if target_mask is not None:
                # Chỉ xét trong thể tích mục tiêu
                volume = np.sum((dose_array < threshold_dose) & target_mask) * voxel_size_cc
            else:
                # Xét toàn bộ mảng liều
                volume = np.sum(dose_array < threshold_dose) * voxel_size_cc
                
            histogram[f"V<{threshold}%"] = volume
        
        return histogram

    @staticmethod
    def calculate_relative_hotspots(dose_array: np.ndarray, reference_dose: float,
                                 target_mask: np.ndarray, voxel_size_cc: float = 0.001) -> Dict[str, float]:
        """
        Tính toán các chỉ số liên quan đến điểm nóng tương đối với thể tích mục tiêu
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        reference_dose : float
            Liều tham chiếu (thường là liều kê), đơn vị Gy
        target_mask : np.ndarray
            Mask của thể tích mục tiêu
        voxel_size_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các chỉ số điểm nóng:
            - max_target_dose: Liều tối đa trong thể tích mục tiêu (Gy)
            - max_dose_ratio: Tỷ lệ liều tối đa/liều tham chiếu
            - max_dose_location: Tọa độ của liều tối đa
            - hotspot_volume_cc: Thể tích nhận >110% liều tham chiếu (cm³)
            - hotspot_volume_percent: Phần trăm thể tích mục tiêu nhận >110% liều
        """
        # Tính thể tích mục tiêu
        target_volume_cc = np.sum(target_mask) * voxel_size_cc
        
        # Liều tối đa trong thể tích mục tiêu
        max_target_dose = np.max(dose_array[target_mask])
        
        # Tọa độ của liều tối đa
        max_dose_indices = np.where(dose_array == max_target_dose)
        if len(max_dose_indices[0]) > 0:
            max_dose_location = (int(max_dose_indices[0][0]), 
                                 int(max_dose_indices[1][0]), 
                                 int(max_dose_indices[2][0]))
        else:
            max_dose_location = None
        
        # Tỷ lệ liều tối đa/liều tham chiếu
        max_dose_ratio = max_target_dose / reference_dose
        
        # Thể tích nhận >110% liều tham chiếu
        hotspot_threshold = reference_dose * 1.1
        hotspot_mask = (dose_array > hotspot_threshold) & target_mask
        hotspot_volume_cc = np.sum(hotspot_mask) * voxel_size_cc
        
        # Phần trăm thể tích mục tiêu nhận >110% liều
        if target_volume_cc > 0:
            hotspot_volume_percent = (hotspot_volume_cc / target_volume_cc) * 100
        else:
            hotspot_volume_percent = 0
        
        return {
            'max_target_dose': float(max_target_dose),
            'max_dose_ratio': float(max_dose_ratio),
            'max_dose_location': max_dose_location,
            'hotspot_volume_cc': float(hotspot_volume_cc),
            'hotspot_volume_percent': float(hotspot_volume_percent)
        }

    @staticmethod
    def generate_spots_report(hotspots: List[Dict], coldspots: List[Dict], 
                           reference_dose: float) -> Dict:
        """
        Tạo báo cáo tổng hợp về các điểm nóng và điểm lạnh
        
        Parameters
        ----------
        hotspots : List[Dict]
            Danh sách các điểm nóng
        coldspots : List[Dict]
            Danh sách các điểm lạnh
        reference_dose : float
            Liều tham chiếu, đơn vị Gy
            
        Returns
        -------
        Dict
            Báo cáo tổng hợp về các điểm nóng và điểm lạnh
        """
        report = {
            'reference_dose': reference_dose,
            'hotspots': {
                'count': len(hotspots),
                'total_volume': sum(h['volume'] for h in hotspots),
                'max_dose': max([h['max_dose'] for h in hotspots]) if hotspots else 0,
                'max_dose_percent': (max([h['max_dose'] for h in hotspots]) / reference_dose * 100) if hotspots else 0,
                'details': hotspots
            },
            'coldspots': {
                'count': len(coldspots),
                'total_volume': sum(c['volume'] for c in coldspots),
                'min_dose': min([c['min_dose'] for c in coldspots]) if coldspots else 0,
                'min_dose_percent': (min([c['min_dose'] for c in coldspots]) / reference_dose * 100) if coldspots else 0,
                'details': coldspots
            }
        }
        
        return report

    @staticmethod
    def interpret_hotspots(hotspots: List[Dict], reference_dose: float) -> str:
        """
        Diễn giải mức độ nghiêm trọng của các điểm nóng
        
        Parameters
        ----------
        hotspots : List[Dict]
            Danh sách các điểm nóng
        reference_dose : float
            Liều tham chiếu, đơn vị Gy
            
        Returns
        -------
        str
            Diễn giải mức độ nghiêm trọng
        """
        if not hotspots:
            return "Không phát hiện điểm nóng đáng kể."
        
        max_dose_percent = max([h['max_dose'] for h in hotspots]) / reference_dose * 100
        total_volume = sum(h['volume'] for h in hotspots)
        
        if max_dose_percent > 130:
            severity = "nghiêm trọng"
        elif max_dose_percent > 120:
            severity = "đáng lo ngại"
        elif max_dose_percent > 110:
            severity = "cần lưu ý"
        else:
            severity = "chấp nhận được"
        
        return (f"Phát hiện {len(hotspots)} điểm nóng với tổng thể tích {total_volume:.2f} cm³. "
                f"Điểm nóng có liều cao nhất là {max_dose_percent:.1f}% liều tham chiếu, "
                f"được đánh giá là {severity}.")

    @staticmethod
    def interpret_coldspots(coldspots: List[Dict], reference_dose: float) -> str:
        """
        Diễn giải mức độ nghiêm trọng của các điểm lạnh
        
        Parameters
        ----------
        coldspots : List[Dict]
            Danh sách các điểm lạnh
        reference_dose : float
            Liều tham chiếu, đơn vị Gy
            
        Returns
        -------
        str
            Diễn giải mức độ nghiêm trọng
        """
        if not coldspots:
            return "Không phát hiện điểm lạnh đáng kể."
        
        min_dose_percent = min([c['min_dose'] for c in coldspots]) / reference_dose * 100
        total_volume = sum(c['volume'] for c in coldspots)
        
        if min_dose_percent < 80:
            severity = "nghiêm trọng"
        elif min_dose_percent < 85:
            severity = "đáng lo ngại"
        elif min_dose_percent < 90:
            severity = "cần lưu ý"
        else:
            severity = "chấp nhận được"
        
        return (f"Phát hiện {len(coldspots)} điểm lạnh với tổng thể tích {total_volume:.2f} cm³. "
                f"Điểm lạnh có liều thấp nhất là {min_dose_percent:.1f}% liều tham chiếu, "
                f"được đánh giá là {severity}.")
