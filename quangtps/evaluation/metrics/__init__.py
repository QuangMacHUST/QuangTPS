#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module metrics chứa các chỉ số đánh giá kế hoạch xạ trị.
"""

from quangtps.evaluation.metrics.conformity import ConformityIndices
from quangtps.evaluation.metrics.homogeneity import HomogeneityIndices
from quangtps.evaluation.metrics.gradient import GradientIndices
from quangtps.evaluation.metrics.paddick import PaddickIndices
from quangtps.evaluation.metrics.hotspot import HotspotAnalysis
from quangtps.evaluation.metrics.integral import IntegralDoseAnalysis
from quangtps.evaluation.metrics.edge import EdgeAnalysis

__all__ = [
    'ConformityIndices',
    'HomogeneityIndices',
    'GradientIndices',
    'PaddickIndices',
    'HotspotAnalysis',
    'IntegralDoseAnalysis',
    'EdgeAnalysis',
    'EvaluationMetrics'
]


class EvaluationMetrics:
    """
    Lớp tổng hợp để tính toán và trả về tất cả các chỉ số đánh giá kế hoạch xạ trị.
    """

    @staticmethod
    def calculate_conformity_indices(tv_ri: float, tv: float, v_ri: float) -> dict:
        """
        Tính toán tất cả các chỉ số đồng dạng và trả về dưới dạng từ điển
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        dict
            Từ điển chứa tất cả các chỉ số đồng dạng đã tính
        """
        return ConformityIndices.calculate_all_metrics(tv_ri, tv, v_ri)

    @staticmethod
    def calculate_homogeneity_indices(d_max: float, d_min: float, d2: float, d5: float, 
                                   d50: float, d95: float, d98: float, d_ref: float, 
                                   d_mean: float) -> dict:
        """
        Tính toán tất cả các chỉ số đồng nhất và trả về dưới dạng từ điển
        
        Parameters
        ----------
        d_max : float
            Liều tối đa trong thể tích mục tiêu (Gy hoặc %)
        d_min : float
            Liều tối thiểu trong thể tích mục tiêu (Gy hoặc %)
        d2 : float
            Liều nhận bởi 2% thể tích mục tiêu (Gy hoặc %)
        d5 : float
            Liều nhận bởi 5% thể tích mục tiêu (Gy hoặc %)
        d50 : float
            Liều nhận bởi 50% thể tích mục tiêu (Gy hoặc %)
        d95 : float
            Liều nhận bởi 95% thể tích mục tiêu (Gy hoặc %)
        d98 : float
            Liều nhận bởi 98% thể tích mục tiêu (Gy hoặc %)
        d_ref : float
            Liều tham chiếu (Gy hoặc %)
        d_mean : float
            Liều trung bình (Gy hoặc %)
            
        Returns
        -------
        dict
            Từ điển chứa tất cả các chỉ số đồng nhất đã tính
        """
        return HomogeneityIndices.calculate_all_metrics(
            d_max, d_min, d2, d5, d50, d95, d98, d_ref, d_mean
        )

    @staticmethod
    def calculate_gradient_indices(v_ref: float, v_half: float, d_ref: float = None, 
                                r_eff: float = None) -> dict:
        """
        Tính toán các chỉ số gradient phổ biến và trả về dưới dạng từ điển
        
        Parameters
        ----------
        v_ref : float
            Thể tích nhận đầy đủ liều tham chiếu (cm³)
        v_half : float
            Thể tích nhận một nửa liều tham chiếu (cm³)
        d_ref : float, optional
            Liều tham chiếu (Gy)
        r_eff : float, optional
            Bán kính hiệu dụng ngoài thể tích tham chiếu (cm)
            
        Returns
        -------
        dict
            Từ điển chứa các chỉ số gradient đã tính
        """
        return GradientIndices.calculate_all_metrics(v_ref, v_half, d_ref, r_eff)

    @staticmethod
    def calculate_paddick_metrics(tv_ri: float, tv: float, v_ri: float, 
                               v_half: float = None) -> dict:
        """
        Tính toán và báo cáo tất cả các chỉ số Paddick liên quan
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
        v_half : float, optional
            Thể tích nhận một nửa liều tham chiếu (cm³)
            
        Returns
        -------
        dict
            Từ điển chứa tất cả các chỉ số Paddick
        """
        return PaddickIndices.paddick_metrics_report(tv_ri, tv, v_ri, v_half)
        
    @staticmethod
    def analyze_hotspots(dose_array, reference_dose, threshold_percent=110.0, 
                      min_volume_cc=0.1, voxel_size_cc=0.001) -> dict:
        """
        Xác định và phân tích các điểm nóng trong phân phối liều
        
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
        dict
            Từ điển chứa thông tin về các điểm nóng được phát hiện
        """
        # Xác định các điểm nóng
        hotspots = HotspotAnalysis.identify_hotspots(
            dose_array, reference_dose, threshold_percent, min_volume_cc, voxel_size_cc
        )
        
        # Tính toán thể tích nhận liều vượt ngưỡng
        hotspot_volumes = HotspotAnalysis.calculate_hotspot_volume_histograms(
            dose_array, reference_dose, None, voxel_size_cc
        )
        
        # Diễn giải các điểm nóng
        interpretation = HotspotAnalysis.interpret_hotspots(hotspots, reference_dose)
        
        return {
            'hotspots': hotspots,
            'hotspot_volumes': hotspot_volumes,
            'interpretation': interpretation
        }
        
    @staticmethod
    def analyze_coldspots(dose_array, reference_dose, target_mask=None, threshold_percent=90.0, 
                       min_volume_cc=0.1, voxel_size_cc=0.001) -> dict:
        """
        Xác định và phân tích các điểm lạnh trong phân phối liều
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        reference_dose : float
            Liều tham chiếu (thường là liều kê), đơn vị Gy
        target_mask : np.ndarray, optional
            Mask của thể tích mục tiêu, nếu None thì sẽ xét toàn bộ mảng liều
        threshold_percent : float, optional
            Ngưỡng phần trăm liều để xác định điểm lạnh, mặc định 90% liều tham chiếu
        min_volume_cc : float, optional
            Thể tích tối thiểu để coi là điểm lạnh có ý nghĩa, đơn vị cm³
        voxel_size_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
            
        Returns
        -------
        dict
            Từ điển chứa thông tin về các điểm lạnh được phát hiện
        """
        # Xác định các điểm lạnh
        coldspots = HotspotAnalysis.identify_coldspots(
            dose_array, reference_dose, threshold_percent, min_volume_cc, 
            voxel_size_cc, target_mask
        )
        
        # Tính toán thể tích nhận liều dưới ngưỡng
        coldspot_volumes = HotspotAnalysis.calculate_coldspot_volume_histograms(
            dose_array, reference_dose, None, voxel_size_cc, target_mask
        )
        
        # Diễn giải các điểm lạnh
        interpretation = HotspotAnalysis.interpret_coldspots(coldspots, reference_dose)
        
        return {
            'coldspots': coldspots,
            'coldspot_volumes': coldspot_volumes,
            'interpretation': interpretation
        }
        
    @staticmethod
    def calculate_integral_dose(dose_array, structure_masks, reference_dose, 
                             voxel_volume_cc=0.001, densities=None) -> dict:
        """
        Tính toán và phân tích liều tích phân cho các cấu trúc
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều, đơn vị Gy
        structure_masks : Dict[str, np.ndarray]
            Từ điển chứa mask của các cấu trúc với khóa là tên cấu trúc
        reference_dose : float
            Liều tham chiếu (thường là liều kê), đơn vị Gy
        voxel_volume_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
        densities : Dict[str, float], optional
            Từ điển chứa mật độ của các cấu trúc với khóa là tên cấu trúc
            Nếu không cung cấp, sử dụng mật độ mặc định là 1.0 g/cm³
            
        Returns
        -------
        dict
            Từ điển chứa kết quả phân tích liều tích phân
        """
        # Tính liều tích phân cho từng cấu trúc
        integral_dose_data = IntegralDoseAnalysis.calculate_multiple_integral_doses(
            dose_array, structure_masks, voxel_volume_cc, densities
        )
        
        # Tạo báo cáo
        report = IntegralDoseAnalysis.generate_integral_dose_report(integral_dose_data)
        
        # Tính liều tích phân cho mô lành
        if 'target' in structure_masks and 'body' in structure_masks:
            normal_tissue_data = IntegralDoseAnalysis.calculate_normal_tissue_integral_dose(
                dose_array, structure_masks['target'], structure_masks['body'], 
                voxel_volume_cc
            )
            
            # Diễn giải liều mô lành
            interpretation = IntegralDoseAnalysis.interpret_normal_tissue_dose(normal_tissue_data)
            
            return {
                'integral_dose_data': integral_dose_data,
                'normal_tissue_data': normal_tissue_data,
                'report': report,
                'interpretation': interpretation
            }
        
        return {
            'integral_dose_data': integral_dose_data,
            'report': report
        }

    @staticmethod
    def analyze_dose_edges(dose_array, target_mask, reference_dose,
                        voxel_size_mm=None) -> dict:
        """
        Phân tích các đặc tính biên liều và đánh giá chất lượng của biên liều
        
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
        dict
            Từ điển chứa kết quả phân tích và đánh giá biên liều
        """
        # Tạo báo cáo phân tích biên liều đầy đủ
        edge_report = EdgeAnalysis.generate_edge_analysis_report(
            dose_array, target_mask, reference_dose, voxel_size_mm
        )
        
        # Trích xuất các thông tin chính để tạo báo cáo rút gọn
        summary = {
            'edge_width_mean': edge_report['edge_width']['edge_width_mean'],
            'falloff_percentage': edge_report['dose_falloff']['percentage_falloff'],
            'mean_gradient': edge_report['dose_gradient']['mean_gradient'],
            'edge_conformity': edge_report['edge_conformity']['edge_conformity_index'],
            'irregularity_score': edge_report['edge_irregularities']['irregularity_score'],
            'num_irregularities': edge_report['edge_irregularities']['num_irregularities'],
            'interpretations': edge_report['evaluation']
        }
        
        # Tạo báo cáo đầy đủ và rút gọn
        return {
            'summary': summary,
            'detailed_report': edge_report
        }

    @staticmethod
    def calculate_all_plan_metrics(plan_data: dict) -> dict:
        """
        Tính toán tất cả các chỉ số đánh giá kế hoạch từ dữ liệu kế hoạch
        
        Parameters
        ----------
        plan_data : dict
            Từ điển chứa dữ liệu kế hoạch bao gồm các thông tin liều và thể tích cần thiết
            Phải chứa các khóa sau:
            - tv_ri: Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
            - tv: Thể tích mục tiêu tổng (cm³)
            - v_ri: Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            - v_half: Thể tích nhận một nửa liều tham chiếu (cm³)
            - d_max: Liều tối đa trong thể tích mục tiêu (Gy hoặc %)
            - d_min: Liều tối thiểu trong thể tích mục tiêu (Gy hoặc %)
            - d2: Liều nhận bởi 2% thể tích mục tiêu (Gy hoặc %)
            - d5: Liều nhận bởi 5% thể tích mục tiêu (Gy hoặc %)
            - d50: Liều nhận bởi 50% thể tích mục tiêu (Gy hoặc %)
            - d95: Liều nhận bởi 95% thể tích mục tiêu (Gy hoặc %)
            - d98: Liều nhận bởi 98% thể tích mục tiêu (Gy hoặc %)
            - d_ref: Liều tham chiếu (Gy hoặc %)
            - d_mean: Liều trung bình (Gy hoặc %)
            - r_eff: Bán kính hiệu dụng ngoài thể tích tham chiếu (cm), tùy chọn
            - dose_array: Mảng 3D chứa dữ liệu liều (tùy chọn)
            - structure_masks: Từ điển chứa mask của các cấu trúc (tùy chọn)
            - voxel_size_mm: Kích thước voxel theo mỗi chiều [dx, dy, dz] (mm), tùy chọn
            
        Returns
        -------
        dict
            Từ điển chứa tất cả các chỉ số đánh giá kế hoạch
        """
        metrics = {}
        
        # Lấy dữ liệu từ từ điển
        tv_ri = plan_data.get('tv_ri')
        tv = plan_data.get('tv')
        v_ri = plan_data.get('v_ri')
        v_half = plan_data.get('v_half')
        d_max = plan_data.get('d_max')
        d_min = plan_data.get('d_min')
        d2 = plan_data.get('d2')
        d5 = plan_data.get('d5')
        d50 = plan_data.get('d50')
        d95 = plan_data.get('d95')
        d98 = plan_data.get('d98')
        d_ref = plan_data.get('d_ref')
        d_mean = plan_data.get('d_mean')
        r_eff = plan_data.get('r_eff')
        dose_array = plan_data.get('dose_array')
        structure_masks = plan_data.get('structure_masks')
        voxel_size_mm = plan_data.get('voxel_size_mm')
        
        # Tính toán các chỉ số đồng dạng
        if all(x is not None for x in [tv_ri, tv, v_ri]):
            metrics['conformity_indices'] = EvaluationMetrics.calculate_conformity_indices(
                tv_ri, tv, v_ri
            )
            
            # Tính toán các chỉ số Paddick
            metrics['paddick_indices'] = EvaluationMetrics.calculate_paddick_metrics(
                tv_ri, tv, v_ri, v_half
            )
            
        # Tính toán các chỉ số đồng nhất
        if all(x is not None for x in [d_max, d_min, d2, d5, d50, d95, d98, d_ref, d_mean]):
            metrics['homogeneity_indices'] = EvaluationMetrics.calculate_homogeneity_indices(
                d_max, d_min, d2, d5, d50, d95, d98, d_ref, d_mean
            )
            
        # Tính toán các chỉ số gradient
        if all(x is not None for x in [v_ri, v_half]):
            metrics['gradient_indices'] = EvaluationMetrics.calculate_gradient_indices(
                v_ri, v_half, d_ref, r_eff
            )
        
        # Nếu có mảng liều và liều tham chiếu, phân tích điểm nóng/lạnh
        if dose_array is not None and d_ref is not None:
            metrics['hotspot_analysis'] = EvaluationMetrics.analyze_hotspots(
                dose_array, d_ref
            )
            
            if structure_masks is not None and 'target' in structure_masks:
                metrics['coldspot_analysis'] = EvaluationMetrics.analyze_coldspots(
                    dose_array, d_ref, structure_masks['target']
                )
                
                # Phân tích biên liều
                metrics['edge_analysis'] = EvaluationMetrics.analyze_dose_edges(
                    dose_array, structure_masks['target'], d_ref, voxel_size_mm
                )
                
            # Nếu có mask cấu trúc, tính liều tích phân
            if structure_masks is not None:
                metrics['integral_dose'] = EvaluationMetrics.calculate_integral_dose(
                    dose_array, structure_masks, d_ref
                )
            
        return metrics

    @staticmethod
    def get_interpretations(metrics: dict) -> dict:
        """
        Diễn giải tất cả các chỉ số đánh giá kế hoạch
        
        Parameters
        ----------
        metrics : dict
            Từ điển chứa các chỉ số đánh giá kế hoạch
            
        Returns
        -------
        dict
            Từ điển chứa diễn giải của các chỉ số
        """
        interpretations = {}
        
        # Diễn giải các chỉ số đồng dạng
        if 'conformity_indices' in metrics:
            conf_metrics = metrics['conformity_indices']
            if 'CI_RTOG' in conf_metrics:
                interpretations['CI_RTOG'] = ConformityIndices.get_rtog_evaluation(
                    conf_metrics['CI_RTOG']
                )
                
        # Diễn giải các chỉ số Paddick
        if 'paddick_indices' in metrics:
            interpretations.update(
                PaddickIndices.interpret_paddick_metrics(metrics['paddick_indices'])
            )
            
        # Diễn giải các chỉ số đồng nhất
        if 'homogeneity_indices' in metrics:
            hom_metrics = metrics['homogeneity_indices']
            if 'HI_ICRU83' in hom_metrics:
                interpretations['HI_ICRU83'] = HomogeneityIndices.interpret_icru83(
                    hom_metrics['HI_ICRU83']
                )
            if 'HI_RTOG' in hom_metrics:
                interpretations['HI_RTOG'] = HomogeneityIndices.interpret_rtog(
                    hom_metrics['HI_RTOG']
                )
                
        # Diễn giải các chỉ số gradient
        if 'gradient_indices' in metrics:
            grad_metrics = metrics['gradient_indices']
            if 'GI_Paddick' in grad_metrics:
                interpretations['GI_Paddick'] = GradientIndices.interpret_gi_paddick(
                    grad_metrics['GI_Paddick']
                )
                
        # Diễn giải điểm nóng/lạnh
        if 'hotspot_analysis' in metrics:
            interpretations['Hotspots'] = metrics['hotspot_analysis'].get('interpretation', '')
            
        if 'coldspot_analysis' in metrics:
            interpretations['Coldspots'] = metrics['coldspot_analysis'].get('interpretation', '')
            
        # Diễn giải liều tích phân
        if 'integral_dose' in metrics and 'interpretation' in metrics['integral_dose']:
            interpretations['Integral_Dose'] = metrics['integral_dose']['interpretation']
            
        # Diễn giải phân tích biên liều
        if 'edge_analysis' in metrics and 'summary' in metrics['edge_analysis']:
            if 'interpretations' in metrics['edge_analysis']['summary']:
                edge_interp = metrics['edge_analysis']['summary']['interpretations']
                if 'overall' in edge_interp:
                    interpretations['Edge_Analysis'] = edge_interp['overall']
                if 'edge_width' in edge_interp:
                    interpretations['Edge_Width'] = edge_interp['edge_width']
                if 'dose_falloff' in edge_interp:
                    interpretations['Dose_Falloff'] = edge_interp['dose_falloff']
                if 'edge_conformity' in edge_interp:
                    interpretations['Edge_Conformity'] = edge_interp['edge_conformity']
                if 'irregularities' in edge_interp:
                    interpretations['Edge_Irregularities'] = edge_interp['irregularities']
                
        return interpretations
