#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module này chứa các phương thức để đánh giá biên và ranh giới của phân phối liều.
Đánh giá biên liều đóng vai trò quan trọng trong việc đảm bảo phủ liều chính xác cho mục tiêu 
đồng thời bảo vệ các cơ quan nguy cấp xung quanh.
"""

import numpy as np
from scipy import ndimage
from typing import List, Dict, Tuple, Optional, Union


class EdgeAnalysis:
    """
    Lớp chứa các phương thức để đánh giá các đặc điểm của biên liều.
    """
    
    @staticmethod
    def calculate_edge_width(dose_array: np.ndarray, 
                           high_dose_percent: float = 80.0, 
                           low_dose_percent: float = 20.0,
                           reference_dose: float = None,
                           voxel_size_mm: List[float] = None) -> Dict[str, float]:
        """
        Tính toán độ rộng biên liều giữa hai mức liều phần trăm.
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        high_dose_percent : float, optional
            Phần trăm liều cao dùng để xác định biên liều, mặc định 80%
        low_dose_percent : float, optional
            Phần trăm liều thấp dùng để xác định biên liều, mặc định 20%
        reference_dose : float, optional
            Liều tham chiếu sử dụng để tính phần trăm. Nếu None, sử dụng giá trị lớn nhất trong mảng
        voxel_size_mm : List[float], optional
            Kích thước voxel theo mỗi chiều [dx, dy, dz] tính bằng mm. Nếu None, giả định [1, 1, 1]
        
        Returns
        -------
        Dict[str, float]
            Từ điển chứa giá trị độ rộng biên liều theo các hướng và giá trị trung bình
        """
        if reference_dose is None:
            reference_dose = np.max(dose_array)
            
        if voxel_size_mm is None:
            voxel_size_mm = [1.0, 1.0, 1.0]
            
        # Tính ngưỡng liều
        high_threshold = reference_dose * high_dose_percent / 100.0
        low_threshold = reference_dose * low_dose_percent / 100.0
        
        # Tạo mask cho vùng liều cao và liều thấp
        high_dose_mask = dose_array >= high_threshold
        low_dose_mask = dose_array >= low_threshold
        
        # Tính khoảng cách trung bình từ vùng liều cao đến biên của vùng liều thấp
        distance_map = ndimage.distance_transform_edt(~high_dose_mask, sampling=voxel_size_mm)
        distances = distance_map[low_dose_mask & ~high_dose_mask]
        
        # Nếu không có vùng chuyển tiếp, trả về giá trị mặc định
        if len(distances) == 0:
            return {
                'edge_width_mean': 0.0,
                'edge_width_min': 0.0,
                'edge_width_max': 0.0,
                'edge_width_std': 0.0
            }
        
        return {
            'edge_width_mean': float(np.mean(distances)),
            'edge_width_min': float(np.min(distances)),
            'edge_width_max': float(np.max(distances)),
            'edge_width_std': float(np.std(distances))
        }
    
    @staticmethod
    def calculate_dose_falloff(dose_array: np.ndarray, 
                             target_mask: np.ndarray,
                             distance_mm: float = 10.0,
                             reference_dose: float = None,
                             voxel_size_mm: List[float] = None) -> Dict[str, float]:
        """
        Tính toán độ giảm liều theo khoảng cách từ biên của mục tiêu.
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        target_mask : np.ndarray
            Mảng binary chỉ ra vùng mục tiêu
        distance_mm : float, optional
            Khoảng cách tính từ biên mục tiêu (mm) để đánh giá độ giảm liều, mặc định 10mm
        reference_dose : float, optional
            Liều tham chiếu sử dụng để tính phần trăm. Nếu None, sử dụng giá trị trung bình trong mục tiêu
        voxel_size_mm : List[float], optional
            Kích thước voxel theo mỗi chiều [dx, dy, dz] tính bằng mm. Nếu None, giả định [1, 1, 1]
        
        Returns
        -------
        Dict[str, float]
            Từ điển chứa thông tin về độ giảm liều
        """
        if voxel_size_mm is None:
            voxel_size_mm = [1.0, 1.0, 1.0]
            
        if reference_dose is None:
            reference_dose = np.mean(dose_array[target_mask])
        
        # Tính khoảng cách từ biên mục tiêu
        distance_map = ndimage.distance_transform_edt(~target_mask, sampling=voxel_size_mm)
        
        # Lấy liều trung bình tại biên mục tiêu
        edge_mask = ndimage.binary_dilation(target_mask, iterations=1) & ~target_mask
        if np.sum(edge_mask) == 0:
            edge_dose = reference_dose
        else:
            edge_dose = np.mean(dose_array[edge_mask])
        
        # Lấy các voxel ở khoảng cách distance_mm từ biên mục tiêu
        distance_mask = (distance_map <= distance_mm) & (distance_map > 0)
        if np.sum(distance_mask) == 0:
            distance_dose = 0.0
        else:
            distance_dose = np.mean(dose_array[distance_mask])
        
        # Tính độ giảm liều tuyệt đối và phần trăm
        absolute_falloff = edge_dose - distance_dose
        percentage_falloff = 100 * (1 - distance_dose / edge_dose) if edge_dose > 0 else 0.0
        falloff_rate = absolute_falloff / distance_mm if distance_mm > 0 else 0.0
        
        return {
            'edge_dose': float(edge_dose),
            'dose_at_distance': float(distance_dose),
            'absolute_falloff': float(absolute_falloff),
            'percentage_falloff': float(percentage_falloff),
            'falloff_rate_per_mm': float(falloff_rate)
        }
    
    @staticmethod
    def analyze_dose_gradient_orientations(dose_array: np.ndarray,
                                        target_mask: np.ndarray,
                                        voxel_size_mm: List[float] = None) -> Dict[str, Union[float, Dict]]:
        """
        Phân tích gradient liều theo các hướng không gian.
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        target_mask : np.ndarray
            Mảng binary chỉ ra vùng mục tiêu
        voxel_size_mm : List[float], optional
            Kích thước voxel theo mỗi chiều [dx, dy, dz] tính bằng mm. Nếu None, giả định [1, 1, 1]
        
        Returns
        -------
        Dict[str, Union[float, Dict]]
            Từ điển chứa thông tin về độ dốc gradient theo các hướng
        """
        if voxel_size_mm is None:
            voxel_size_mm = [1.0, 1.0, 1.0]
        
        # Tính gradient theo mỗi hướng, đã chuẩn hóa theo kích thước voxel
        dx, dy, dz = np.gradient(dose_array)
        dx = dx / voxel_size_mm[0]
        dy = dy / voxel_size_mm[1]
        dz = dz / voxel_size_mm[2]
        
        # Tạo mask cho vùng bìa rộng 5mm từ biên mục tiêu
        distance_map = ndimage.distance_transform_edt(~target_mask, sampling=voxel_size_mm)
        edge_region_mask = (distance_map <= 5.0) & (distance_map > 0)
        
        # Tính magnitude của gradient
        gradient_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Chỉ tính gradient trong vùng bìa
        if np.sum(edge_region_mask) == 0:
            return {
                'mean_gradient': 0.0,
                'max_gradient': 0.0,
                'directional_gradients': {
                    'x_direction': {'mean': 0.0, 'max': 0.0, 'std': 0.0},
                    'y_direction': {'mean': 0.0, 'max': 0.0, 'std': 0.0},
                    'z_direction': {'mean': 0.0, 'max': 0.0, 'std': 0.0}
                }
            }
        
        # Tính các thống kê cho gradient trong vùng bìa
        edge_gradients = gradient_magnitude[edge_region_mask]
        edge_dx = dx[edge_region_mask]
        edge_dy = dy[edge_region_mask]
        edge_dz = dz[edge_region_mask]
        
        return {
            'mean_gradient': float(np.mean(edge_gradients)),
            'max_gradient': float(np.max(edge_gradients)),
            'directional_gradients': {
                'x_direction': {
                    'mean': float(np.mean(np.abs(edge_dx))),
                    'max': float(np.max(np.abs(edge_dx))),
                    'std': float(np.std(np.abs(edge_dx)))
                },
                'y_direction': {
                    'mean': float(np.mean(np.abs(edge_dy))),
                    'max': float(np.max(np.abs(edge_dy))),
                    'std': float(np.std(np.abs(edge_dy)))
                },
                'z_direction': {
                    'mean': float(np.mean(np.abs(edge_dz))),
                    'max': float(np.max(np.abs(edge_dz))),
                    'std': float(np.std(np.abs(edge_dz)))
                }
            }
        }
    
    @staticmethod
    def calculate_edge_conformity(dose_array: np.ndarray,
                                target_mask: np.ndarray,
                                reference_dose: float,
                                edge_distance_mm: float = 2.0,
                                voxel_size_mm: List[float] = None) -> Dict[str, float]:
        """
        Tính toán chỉ số đồng dạng biên (Edge Conformity Index).
        
        Chỉ số này đánh giá mức độ tương ứng giữa biên liều quy định và biên của thể tích mục tiêu.
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        target_mask : np.ndarray
            Mảng binary chỉ ra vùng mục tiêu
        reference_dose : float
            Liều tham chiếu sử dụng để đánh giá
        edge_distance_mm : float, optional
            Khoảng cách từ biên mục tiêu để xác định vùng biên (mm), mặc định 2mm
        voxel_size_mm : List[float], optional
            Kích thước voxel theo mỗi chiều [dx, dy, dz] tính bằng mm. Nếu None, giả định [1, 1, 1]
        
        Returns
        -------
        Dict[str, float]
            Từ điển chứa chỉ số đồng dạng biên và các thông số liên quan
        """
        if voxel_size_mm is None:
            voxel_size_mm = [1.0, 1.0, 1.0]
        
        # Tạo mask chỉ ra biên của mục tiêu với một khoảng cách nhất định
        distance_map_inside = ndimage.distance_transform_edt(target_mask, sampling=voxel_size_mm)
        distance_map_outside = ndimage.distance_transform_edt(~target_mask, sampling=voxel_size_mm)
        
        target_edge_mask = (distance_map_inside <= edge_distance_mm) & target_mask
        outside_edge_mask = (distance_map_outside <= edge_distance_mm) & ~target_mask
        
        # Tạo mask cho vùng liều tham chiếu
        reference_dose_mask = dose_array >= reference_dose
        
        # Tính toán thể tích các vùng
        voxel_volume = np.prod(voxel_size_mm) / 1000.0  # chuyển đổi sang cm³
        
        # Thể tích mục tiêu bên trong khoảng cách edge
        v_target_edge = np.sum(target_edge_mask) * voxel_volume
        
        # Thể tích mục tiêu biên nhận đủ liều tham chiếu
        v_target_edge_covered = np.sum(target_edge_mask & reference_dose_mask) * voxel_volume
        
        # Thể tích ngoài mục tiêu nằm trong khoảng cách edge
        v_outside_edge = np.sum(outside_edge_mask) * voxel_volume
        
        # Thể tích ngoài mục tiêu biên nhận liều tham chiếu
        v_outside_edge_dose = np.sum(outside_edge_mask & reference_dose_mask) * voxel_volume
        
        # Tính chỉ số đồng dạng biên
        target_edge_coverage = v_target_edge_covered / v_target_edge if v_target_edge > 0 else 0.0
        outside_edge_sparing = 1.0 - (v_outside_edge_dose / v_outside_edge) if v_outside_edge > 0 else 1.0
        
        # Chỉ số đồng dạng biên tổng quát
        edge_conformity_index = (target_edge_coverage + outside_edge_sparing) / 2.0
        
        return {
            'target_edge_coverage': float(target_edge_coverage),
            'outside_edge_sparing': float(outside_edge_sparing),
            'edge_conformity_index': float(edge_conformity_index),
            'v_target_edge': float(v_target_edge),
            'v_target_edge_covered': float(v_target_edge_covered),
            'v_outside_edge': float(v_outside_edge),
            'v_outside_edge_dose': float(v_outside_edge_dose)
        }
    
    @staticmethod
    def identify_edge_irregularities(dose_array: np.ndarray,
                                   target_mask: np.ndarray,
                                   reference_dose: float,
                                   edge_distance_mm: float = 5.0,
                                   voxel_size_mm: List[float] = None) -> Dict[str, Union[float, List[Dict]]]:
        """
        Xác định các bất thường trong phân phối liều tại biên của mục tiêu.
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        target_mask : np.ndarray
            Mảng binary chỉ ra vùng mục tiêu
        reference_dose : float
            Liều tham chiếu sử dụng để đánh giá
        edge_distance_mm : float, optional
            Khoảng cách từ biên mục tiêu để xác định vùng biên (mm), mặc định 5mm
        voxel_size_mm : List[float], optional
            Kích thước voxel theo mỗi chiều [dx, dy, dz] tính bằng mm. Nếu None, giả định [1, 1, 1]
        
        Returns
        -------
        Dict[str, Union[float, List[Dict]]]
            Từ điển chứa thông tin về các bất thường tại biên liều
        """
        if voxel_size_mm is None:
            voxel_size_mm = [1.0, 1.0, 1.0]
        
        # Tính gradient
        dx, dy, dz = np.gradient(dose_array)
        gradient_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Tạo mask cho vùng biên
        distance_map = ndimage.distance_transform_edt(~target_mask, sampling=voxel_size_mm)
        edge_region_mask = (distance_map <= edge_distance_mm) & (distance_map > 0)
        
        if np.sum(edge_region_mask) == 0:
            return {
                'irregularity_score': 0.0,
                'irregularities': []
            }
        
        # Tính các thống kê của gradient tại vùng biên
        edge_gradients = gradient_magnitude[edge_region_mask]
        mean_gradient = np.mean(edge_gradients)
        std_gradient = np.std(edge_gradients)
        
        # Xác định vùng có gradient bất thường (2 độ lệch chuẩn so với trung bình)
        threshold = mean_gradient + 2 * std_gradient
        irregular_mask = (gradient_magnitude > threshold) & edge_region_mask
        
        # Đếm và xác định vị trí các bất thường
        labeled_arrays, num_features = ndimage.label(irregular_mask)
        irregularities = []
        
        for i in range(1, num_features + 1):
            feature_mask = labeled_arrays == i
            
            # Tính toán thể tích của bất thường
            volume_cc = np.sum(feature_mask) * np.prod(voxel_size_mm) / 1000.0
            
            # Tính giá trị liều và gradient trung bình
            mean_dose = np.mean(dose_array[feature_mask])
            mean_grad = np.mean(gradient_magnitude[feature_mask])
            
            # Xác định vị trí trung tâm của bất thường
            coords = np.argwhere(feature_mask)
            center = np.mean(coords, axis=0)
            
            irregularities.append({
                'volume_cc': float(volume_cc),
                'mean_dose': float(mean_dose),
                'dose_percent': float(mean_dose / reference_dose * 100) if reference_dose > 0 else 0.0,
                'mean_gradient': float(mean_grad),
                'center_position': [float(center[0]), float(center[1]), float(center[2])]
            })
        
        # Tính "điểm bất thường" tổng thể
        irregularity_score = num_features * (std_gradient / mean_gradient) if mean_gradient > 0 else 0.0
        
        return {
            'irregularity_score': float(irregularity_score),
            'num_irregularities': num_features,
            'mean_gradient': float(mean_gradient),
            'std_gradient': float(std_gradient),
            'irregularities': irregularities
        }
    
    @staticmethod
    def generate_edge_analysis_report(dose_array: np.ndarray,
                                    target_mask: np.ndarray,
                                    reference_dose: float,
                                    voxel_size_mm: List[float] = None) -> Dict:
        """
        Tạo báo cáo tổng hợp về phân tích biên liều cho một kế hoạch.
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        target_mask : np.ndarray
            Mảng binary chỉ ra vùng mục tiêu
        reference_dose : float
            Liều tham chiếu (Gy)
        voxel_size_mm : List[float], optional
            Kích thước voxel theo mỗi chiều [dx, dy, dz] tính bằng mm. Nếu None, giả định [1, 1, 1]
        
        Returns
        -------
        Dict
            Báo cáo tổng hợp về phân tích biên liều
        """
        if voxel_size_mm is None:
            voxel_size_mm = [1.0, 1.0, 1.0]
            
        # Phân tích độ rộng biên
        edge_width_data = EdgeAnalysis.calculate_edge_width(
            dose_array, 80.0, 20.0, reference_dose, voxel_size_mm
        )
        
        # Phân tích độ giảm liều
        falloff_data = EdgeAnalysis.calculate_dose_falloff(
            dose_array, target_mask, 10.0, reference_dose, voxel_size_mm
        )
        
        # Phân tích gradient theo hướng
        gradient_data = EdgeAnalysis.analyze_dose_gradient_orientations(
            dose_array, target_mask, voxel_size_mm
        )
        
        # Tính chỉ số đồng dạng biên
        edge_conformity_data = EdgeAnalysis.calculate_edge_conformity(
            dose_array, target_mask, reference_dose, 2.0, voxel_size_mm
        )
        
        # Xác định bất thường tại biên
        irregularities_data = EdgeAnalysis.identify_edge_irregularities(
            dose_array, target_mask, reference_dose, 5.0, voxel_size_mm
        )
        
        # Tổng hợp báo cáo
        report = {
            'edge_width': edge_width_data,
            'dose_falloff': falloff_data,
            'dose_gradient': gradient_data,
            'edge_conformity': edge_conformity_data,
            'edge_irregularities': irregularities_data
        }
        
        # Thêm đánh giá tổng quát
        report['evaluation'] = EdgeAnalysis.interpret_edge_results(report)
        
        return report
    
    @staticmethod
    def interpret_edge_results(report: Dict) -> Dict[str, str]:
        """
        Diễn giải kết quả phân tích biên liều.
        
        Parameters
        ----------
        report : Dict
            Báo cáo kết quả phân tích biên
            
        Returns
        -------
        Dict[str, str]
            Từ điển chứa diễn giải cho từng khía cạnh của biên liều
        """
        interpretations = {}
        
        # Diễn giải độ rộng biên
        edge_width = report['edge_width']['edge_width_mean']
        if edge_width < 2:
            interpretations['edge_width'] = "Biên liều rất dốc, chuyển tiếp liều nhanh."
        elif edge_width < 5:
            interpretations['edge_width'] = "Biên liều dốc, độ chuyển tiếp liều phù hợp."
        elif edge_width < 10:
            interpretations['edge_width'] = "Biên liều trung bình, độ chuyển tiếp liều vừa phải."
        else:
            interpretations['edge_width'] = "Biên liều thoải, độ chuyển tiếp liều chậm."
        
        # Diễn giải độ giảm liều
        percentage_falloff = report['dose_falloff']['percentage_falloff']
        if percentage_falloff > 80:
            interpretations['dose_falloff'] = "Độ giảm liều rất tốt, bảo vệ tối ưu cho mô lành."
        elif percentage_falloff > 60:
            interpretations['dose_falloff'] = "Độ giảm liều tốt, bảo vệ đáng kể cho mô lành."
        elif percentage_falloff > 40:
            interpretations['dose_falloff'] = "Độ giảm liều trung bình, cần xem xét cải thiện."
        else:
            interpretations['dose_falloff'] = "Độ giảm liều thấp, cần cải thiện để bảo vệ mô lành tốt hơn."
        
        # Diễn giải đồng dạng biên
        edge_conformity = report['edge_conformity']['edge_conformity_index']
        if edge_conformity > 0.9:
            interpretations['edge_conformity'] = "Đồng dạng biên liều xuất sắc, phân phối liều rất phù hợp với hình dạng mục tiêu."
        elif edge_conformity > 0.8:
            interpretations['edge_conformity'] = "Đồng dạng biên liều tốt, phân phối liều phù hợp với hình dạng mục tiêu."
        elif edge_conformity > 0.7:
            interpretations['edge_conformity'] = "Đồng dạng biên liều chấp nhận được, có thể cải thiện thêm."
        else:
            interpretations['edge_conformity'] = "Đồng dạng biên liều kém, cần cải thiện sự phù hợp giữa phân phối liều và hình dạng mục tiêu."
        
        # Diễn giải các bất thường
        num_irregularities = report['edge_irregularities']['num_irregularities']
        if num_irregularities == 0:
            interpretations['irregularities'] = "Không phát hiện bất thường tại biên liều."
        elif num_irregularities < 3:
            interpretations['irregularities'] = f"Phát hiện {num_irregularities} vùng bất thường nhỏ tại biên liều."
        elif num_irregularities < 6:
            interpretations['irregularities'] = f"Phát hiện {num_irregularities} vùng bất thường đáng chú ý tại biên liều, cần xem xét."
        else:
            interpretations['irregularities'] = f"Phát hiện {num_irregularities} vùng bất thường đáng kể tại biên liều, cần đánh giá kỹ và cải thiện kế hoạch."
        
        # Đánh giá tổng quát
        average_score = (
            min(1, edge_width / 5) * 0.2 +
            (percentage_falloff / 100) * 0.3 +
            edge_conformity * 0.3 +
            max(0, 1 - min(1, num_irregularities / 5)) * 0.2
        )
        
        if average_score > 0.9:
            interpretations['overall'] = "Đánh giá biên liều: Xuất sắc. Biên liều rõ ràng, dốc và đồng dạng tốt với mục tiêu."
        elif average_score > 0.8:
            interpretations['overall'] = "Đánh giá biên liều: Rất tốt. Biên liều có đặc tính phù hợp cho mục đích điều trị."
        elif average_score > 0.7:
            interpretations['overall'] = "Đánh giá biên liều: Tốt. Biên liều đáp ứng các yêu cầu cơ bản nhưng có thể cải thiện."
        elif average_score > 0.6:
            interpretations['overall'] = "Đánh giá biên liều: Khá. Biên liều có một số hạn chế cần được xem xét."
        else:
            interpretations['overall'] = "Đánh giá biên liều: Cần cải thiện. Biên liều không đáp ứng tốt các tiêu chí mong muốn."
        
        return interpretations 