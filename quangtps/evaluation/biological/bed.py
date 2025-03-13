#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các phương thức tính toán liều sinh học hiệu quả (BED - Biologically Effective Dose).

BED là chỉ số quan trọng trong đánh giá kế hoạch xạ trị, sử dụng để so sánh các phương pháp 
phân liều khác nhau. BED dựa trên mô hình tuyến tính-bậc hai (Linear-Quadratic model) 
của tổn thương tế bào do bức xạ.
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional


class BiologicalEffectiveDose:
    """
    Lớp tính toán liều sinh học hiệu quả (BED) và các chỉ số sinh học liên quan.
    """

    @staticmethod
    def calculate_bed(dose: float, fractions: int, alpha_beta: float) -> float:
        """
        Tính toán liều sinh học hiệu quả (BED) dựa trên mô hình LQ.
        
        BED = nd * (1 + d/(α/β))
        
        Trong đó:
        - n: số phân liều
        - d: liều mỗi phân liều (Gy)
        - α/β: tỷ lệ alpha/beta, đặc trưng cho loại mô (Gy)
        
        Parameters
        ----------
        dose : float
            Tổng liều (Gy)
        fractions : int
            Số phân liều
        alpha_beta : float
            Tỷ lệ alpha/beta của mô (Gy)
            
        Returns
        -------
        float
            Liều sinh học hiệu quả (BED), đơn vị Gy
            
        References
        ----------
        Hall, E.J., Giaccia, A.J. (2006) Radiobiology for the Radiologist, 6th ed.
        Philadelphia: Lippincott Williams and Wilkins.
        """
        if fractions <= 0:
            raise ValueError("Số phân liều phải lớn hơn 0")
            
        if alpha_beta <= 0:
            raise ValueError("Tỷ lệ alpha/beta phải lớn hơn 0")
            
        # Liều mỗi phân liều
        dose_per_fraction = dose / fractions
        
        # Tính BED
        bed = dose * (1 + dose_per_fraction / alpha_beta)
        
        return float(bed)

    @staticmethod
    def bed_to_total_dose(bed: float, fractions: int, alpha_beta: float) -> float:
        """
        Chuyển đổi từ BED sang tổng liều vật lý.
        
        Parameters
        ----------
        bed : float
            Liều sinh học hiệu quả (Gy)
        fractions : int
            Số phân liều
        alpha_beta : float
            Tỷ lệ alpha/beta của mô (Gy)
            
        Returns
        -------
        float
            Tổng liều vật lý (Gy)
        """
        if fractions <= 0:
            raise ValueError("Số phân liều phải lớn hơn 0")
            
        if alpha_beta <= 0:
            raise ValueError("Tỷ lệ alpha/beta phải lớn hơn 0")
            
        # Giải phương trình BED = nd * (1 + d/(α/β))
        # Trong đó d = D/n (liều mỗi phân liều)
        
        # BED = D * (1 + D/(n*α/β))
        # Giải phương trình bậc hai: D² + (n*α/β)*D - BED*(n*α/β) = 0
        
        a = 1
        b = fractions * alpha_beta
        c = -bed * fractions * alpha_beta
        
        # Phương trình ax² + bx + c = 0, x = D
        delta = b**2 - 4*a*c
        
        if delta < 0:
            raise ValueError("Không thể tính tổng liều từ BED này")
            
        # Chỉ lấy nghiệm dương
        total_dose = (-b + np.sqrt(delta)) / (2*a)
        
        return float(total_dose)

    @staticmethod
    def calculate_dose_per_fraction(bed: float, alpha_beta: float, fractions: int) -> float:
        """
        Tính liều mỗi phân liều từ BED.
        
        Parameters
        ----------
        bed : float
            Liều sinh học hiệu quả (Gy)
        alpha_beta : float
            Tỷ lệ alpha/beta của mô (Gy)
        fractions : int
            Số phân liều
            
        Returns
        -------
        float
            Liều mỗi phân liều (Gy)
        """
        if fractions <= 0:
            raise ValueError("Số phân liều phải lớn hơn 0")
            
        # Tính tổng liều trước
        total_dose = BiologicalEffectiveDose.bed_to_total_dose(bed, fractions, alpha_beta)
        
        # Sau đó tính liều mỗi phân liều
        dose_per_fraction = total_dose / fractions
        
        return float(dose_per_fraction)

    @staticmethod
    def calculate_equivalent_fractions(dose1: float, fractions1: int, 
                                    dose2: float, alpha_beta: float) -> int:
        """
        Tính số phân liều cần thiết để liều thứ hai đạt được cùng BED với liều thứ nhất.
        
        Parameters
        ----------
        dose1 : float
            Tổng liều thứ nhất (Gy)
        fractions1 : int
            Số phân liều của phương án thứ nhất
        dose2 : float
            Tổng liều thứ hai (Gy)
        alpha_beta : float
            Tỷ lệ alpha/beta của mô (Gy)
            
        Returns
        -------
        int
            Số phân liều cần thiết cho tổng liều thứ hai
        """
        # Tính BED cho phương án thứ nhất
        bed1 = BiologicalEffectiveDose.calculate_bed(dose1, fractions1, alpha_beta)
        
        # Tìm số phân liều n2 sao cho BED2 = BED1
        # BED = nd * (1 + d/(α/β))
        # Với d = D/n
        
        # Giải phương trình: D2 * (1 + D2/(n2*α/β)) = BED1
        # Biến đổi: D2²/(n2*α/β) + D2 - BED1 = 0
        # => n2 = D2²/(α/β * (BED1 - D2))
        
        if dose2 >= bed1:
            raise ValueError("Tổng liều thứ hai phải nhỏ hơn BED của phương án thứ nhất")
            
        n2 = (dose2**2) / (alpha_beta * (bed1 - dose2))
        
        # Làm tròn lên để đảm bảo BED đủ
        return int(np.ceil(n2))

    @staticmethod
    def calculate_log_cell_kill(bed: float, alpha: float = 0.3) -> float:
        """
        Tính toán log cell kill dựa trên liều sinh học hiệu quả (BED).
        
        Parameters
        ----------
        bed : float
            Liều sinh học hiệu quả (Gy)
        alpha : float, optional
            Hệ số alpha trong mô hình tuyến tính-bậc hai (Gy^-1)
            
        Returns
        -------
        float
            Log cell kill
        """
        return alpha * bed

    @staticmethod
    def calculate_bed_for_tumor_control(tcp: float, alpha: float = 0.3, 
                                     num_clonogens: float = 1e7) -> float:
        """
        Tính BED cần thiết để đạt được xác suất kiểm soát khối u (TCP) nhất định.
        
        Parameters
        ----------
        tcp : float
            Xác suất kiểm soát khối u (0-1)
        alpha : float, optional
            Hệ số alpha trong mô hình tuyến tính-bậc hai (Gy^-1)
        num_clonogens : float, optional
            Số tế bào nhân được ban đầu trong khối u
            
        Returns
        -------
        float
            BED cần thiết (Gy)
        """
        if tcp <= 0 or tcp >= 1:
            raise ValueError("TCP phải nằm trong khoảng (0, 1)")
            
        # Tính log cell kill cần thiết
        log_cell_kill = np.log(num_clonogens) + np.log(-np.log(1 - tcp))
        
        # Tính BED
        bed = log_cell_kill / alpha
        
        return float(bed)

    @staticmethod
    def convert_bed_between_fraction_schemes(bed1: float, fractions1: int, 
                                          fractions2: int, alpha_beta: float) -> float:
        """
        Chuyển đổi BED giữa các phương án phân liều khác nhau.
        
        Parameters
        ----------
        bed1 : float
            BED ban đầu (Gy)
        fractions1 : int
            Số phân liều ban đầu
        fractions2 : int
            Số phân liều mới
        alpha_beta : float
            Tỷ lệ alpha/beta của mô (Gy)
            
        Returns
        -------
        float
            BED tương đương với phương án phân liều mới (Gy)
        """
        # Tính tổng liều vật lý từ BED ban đầu
        total_dose1 = BiologicalEffectiveDose.bed_to_total_dose(bed1, fractions1, alpha_beta)
        
        # Tính liều mỗi phân liều
        dose_per_fraction1 = total_dose1 / fractions1
        
        # Tính tổng liều mới
        total_dose2 = dose_per_fraction1 * fractions2
        
        # Tính BED mới
        bed2 = BiologicalEffectiveDose.calculate_bed(total_dose2, fractions2, alpha_beta)
        
        return float(bed2)

    @staticmethod
    def bed_with_repair(dose: float, fractions: int, alpha_beta: float, 
                      halftime: float, treatment_time: float) -> float:
        """
        Tính BED có tính đến sửa chữa trong quá trình chiếu xạ dài.
        
        Parameters
        ----------
        dose : float
            Tổng liều (Gy)
        fractions : int
            Số phân liều
        alpha_beta : float
            Tỷ lệ alpha/beta của mô (Gy)
        halftime : float
            Thời gian bán rã sửa chữa tổn thương bậc hai (giờ)
        treatment_time : float
            Thời gian chiếu xạ mỗi phân liều (giờ)
            
        Returns
        -------
        float
            BED có tính đến sửa chữa (Gy)
        """
        if fractions <= 0:
            raise ValueError("Số phân liều phải lớn hơn 0")
            
        if halftime <= 0:
            raise ValueError("Thời gian bán rã phải lớn hơn 0")
            
        # Liều mỗi phân liều
        dose_per_fraction = dose / fractions
        
        # Hệ số sửa chữa (g)
        repair_factor = 2 * halftime / treatment_time * (1 - (1 - np.exp(-0.693 * treatment_time / halftime)) / (0.693 * treatment_time / halftime))
        
        # Tính BED có sửa chữa
        bed = dose * (1 + (dose_per_fraction / alpha_beta) * repair_factor)
        
        return float(bed)

    @staticmethod
    def bed_from_dvh(dvh_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                   structure_name: str, 
                   fractions: int, 
                   alpha_beta: float) -> Dict[str, float]:
        """
        Tính liều sinh học hiệu quả (BED) từ dữ liệu DVH.
        
        Parameters
        ----------
        dvh_data : Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dữ liệu DVH, key là tên cấu trúc, value là tuple (dose_bins, volume_pcts)
        structure_name : str
            Tên cấu trúc cần tính BED
        fractions : int
            Số phân liều
        alpha_beta : float
            Tỷ lệ alpha/beta của mô (Gy)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thông số BED:
            - min_bed: BED tối thiểu
            - max_bed: BED tối đa
            - mean_bed: BED trung bình
            - median_bed: BED trung vị
            - bed_d90: BED nhận bởi 90% thể tích
            - bed_d95: BED nhận bởi 95% thể tích
            - bed_d98: BED nhận bởi 98% thể tích
        """
        if structure_name not in dvh_data:
            raise ValueError(f"Cấu trúc '{structure_name}' không có trong dữ liệu DVH")
            
        # Lấy dữ liệu DVH của cấu trúc
        dose_bins, volume_pcts = dvh_data[structure_name]
        
        # Chuyển đổi mỗi giá trị liều sang BED
        bed_values = np.zeros_like(dose_bins)
        for i, dose in enumerate(dose_bins):
            try:
                # Giả sử rằng các giá trị trong dose_bins là liều tích lũy,
                # cần chia cho số phân liều để có liều mỗi phân liều
                dose_per_fraction = dose / fractions
                bed_values[i] = dose * (1 + dose_per_fraction / alpha_beta)
            except:
                bed_values[i] = 0
        
        # Tính các thông số BED
        # Lưu ý: DVH tích lũy có đồ thị giảm dần theo liều
        min_bed = np.min(bed_values[bed_values > 0]) if np.any(bed_values > 0) else 0
        max_bed = np.max(bed_values)
        
        # Tính BED trung bình (phải tính từ DVH vi phân)
        # Chuyển DVH tích lũy sang vi phân
        dvh_diff = np.abs(np.diff(np.append(volume_pcts, 0)))
        mean_bed = np.sum(bed_values[:-1] * dvh_diff) / 100  # Chia 100 vì volume_pcts là phần trăm
        
        # Nội suy để tìm BED trung vị và các điểm BED khác
        # Đảo ngược dữ liệu DVH để nội suy (vì DVH tích lũy giảm dần theo liều)
        sorted_indices = np.argsort(bed_values)
        sorted_bed = bed_values[sorted_indices]
        sorted_vol = volume_pcts[sorted_indices]
        
        # Nội suy tuyến tính để tìm BED ở các phần trăm thể tích
        median_bed = np.interp(50, sorted_vol, sorted_bed)
        bed_d90 = np.interp(90, sorted_vol, sorted_bed)
        bed_d95 = np.interp(95, sorted_vol, sorted_bed)
        bed_d98 = np.interp(98, sorted_vol, sorted_bed)
        
        return {
            'min_bed': float(min_bed),
            'max_bed': float(max_bed),
            'mean_bed': float(mean_bed),
            'median_bed': float(median_bed),
            'bed_d90': float(bed_d90),
            'bed_d95': float(bed_d95),
            'bed_d98': float(bed_d98)
        }

    @staticmethod
    def generate_bed_report(dvh_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                         structures: List[str],
                         fractions: int,
                         alpha_beta_values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Tạo báo cáo BED toàn diện cho nhiều cấu trúc.
        
        Parameters
        ----------
        dvh_data : Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dữ liệu DVH, key là tên cấu trúc, value là tuple (dose_bins, volume_pcts)
        structures : List[str]
            Danh sách tên cấu trúc cần tính BED
        fractions : int
            Số phân liều
        alpha_beta_values : Dict[str, float]
            Từ điển chứa giá trị alpha/beta cho mỗi cấu trúc
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Báo cáo BED cho mỗi cấu trúc
        """
        report = {}
        
        for structure in structures:
            if structure not in dvh_data:
                continue
                
            # Lấy giá trị alpha/beta cho cấu trúc
            if structure in alpha_beta_values:
                alpha_beta = alpha_beta_values[structure]
            else:
                # Giá trị mặc định
                alpha_beta = 10.0 if "PTV" in structure or "GTV" in structure or "CTV" in structure else 3.0
                
            # Tính BED cho cấu trúc
            bed_data = BiologicalEffectiveDose.bed_from_dvh(
                dvh_data, structure, fractions, alpha_beta
            )
            
            # Thêm thông tin alpha/beta
            bed_data['alpha_beta'] = alpha_beta
            
            # Thêm vào báo cáo
            report[structure] = bed_data
            
        return report

    @staticmethod
    def interpret_bed_results(bed_data: Dict[str, float], 
                           is_tumor: bool = False) -> str:
        """
        Diễn giải kết quả BED.
        
        Parameters
        ----------
        bed_data : Dict[str, float]
            Dữ liệu BED từ hàm bed_from_dvh
        is_tumor : bool, optional
            True nếu cấu trúc là khối u, False nếu là cơ quan nguy cấp
            
        Returns
        -------
        str
            Diễn giải kết quả BED
        """
        if is_tumor:
            # Diễn giải cho khối u
            min_bed = bed_data.get('min_bed', 0)
            d98_bed = bed_data.get('bed_d98', 0)
            
            if min_bed < 60:
                return "BED thấp, có thể không đủ để kiểm soát khối u."
            elif min_bed < 70:
                return "BED trung bình, khả năng kiểm soát khối u ở mức vừa phải."
            elif min_bed < 80:
                return "BED khá tốt, khả năng kiểm soát khối u cao."
            else:
                return "BED rất cao, khả năng kiểm soát khối u rất tốt."
        else:
            # Diễn giải cho cơ quan nguy cấp
            max_bed = bed_data.get('max_bed', 0)
            
            if max_bed < 50:
                return "BED thấp, rủi ro biến chứng thấp."
            elif max_bed < 80:
                return "BED trung bình, rủi ro biến chứng ở mức vừa phải."
            elif max_bed < 100:
                return "BED cao, rủi ro biến chứng đáng kể."
            else:
                return "BED rất cao, rủi ro biến chứng nghiêm trọng." 