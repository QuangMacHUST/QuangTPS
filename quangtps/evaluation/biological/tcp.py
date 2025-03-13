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
class TCPModels:
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
    
        Parameters:
            dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
            structure_mask (np.ndarray): Mảng mask 3D của khối u
            num_fractions (int): Số phân liều
            alpha (float, optional): Hệ số alpha trong mô hình LQ (Gy^-1)
            alpha_beta (float, optional): Tỉ lệ alpha/beta (Gy)
            clonogenic_density (float, optional): Mật độ tế bào sinh ung thư (cells/cm^3)
            dose_threshold (float, optional): Ngưỡng liều để tính TCP, voxel có liều < ngưỡng sẽ bị bỏ qua
    
        Returns:
            float: Giá trị TCP (0-1)
        """
        # Kiểm tra mask và dose có cùng kích thước
        if dose_array.shape != structure_mask.shape:
            raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
        # Chỉ xét các voxel trong khối u
        mask = structure_mask > 0
        if dose_threshold is not None:
            mask = mask & (dose_array >= dose_threshold)
    
        # Nếu không có voxel nào thỏa điều kiện
        if not np.any(mask):
            logger.warning("No valid voxels found for TCP calculation")
            return 0.0
    
        # Lấy liều tại các voxel trong khối u
        tumor_doses = dose_array[mask]
    
        # Liều mỗi phân liều (Gy)
        dose_per_fraction = tumor_doses / num_fractions
    
        # Tính EQD2 cho mỗi voxel
        eqd2 = tumor_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
        # Tính số tế bào sống sót theo mô hình LQ
        survival = np.exp(-alpha * eqd2)
    
        # Tính thể tích khối u (cm^3)
        # Giả định spacing là 1mm x 1mm x 1mm
        tumor_volume = np.sum(mask) / 1000.0  # Chuyển từ mm^3 sang cm^3
    
        # Tính số tế bào sinh ung thư ban đầu
        initial_cells = clonogenic_density * tumor_volume
    
        # Tính số tế bào sinh ung thư sống sót trung bình
        surviving_cells = initial_cells * np.mean(survival)
    
        # Tính TCP theo mô hình Poisson
        tcp = np.exp(-surviving_cells)
    
        return tcp

    def calculate_tcp_lq_poisson_dvh(
        dvh_data: Dict[str, np.ndarray],
        num_fractions: int,
        alpha: float = 0.3,
        alpha_beta: float = 10.0,
        clonogenic_density: float = 1e7,
        tumor_volume: float = None
    ) -> float:
        """
        Tính toán TCP dựa trên DVH và mô hình LQ-Poisson.
    
        Parameters:
            dvh_data (dict): Dict chứa dữ liệu DVH với các key:
                - 'dose': Mảng giá trị liều (Gy)
                - 'volume': Mảng giá trị thể tích (% hoặc cc)
                - 'relative_volume': True nếu thể tích là phần trăm
            num_fractions (int): Số phân liều
            alpha (float, optional): Hệ số alpha trong mô hình LQ (Gy^-1)
            alpha_beta (float, optional): Tỉ lệ alpha/beta (Gy)
            clonogenic_density (float, optional): Mật độ tế bào sinh ung thư (cells/cm^3)
            tumor_volume (float, optional): Thể tích khối u (cm^3), cần thiết nếu DVH sử dụng thể tích tương đối
    
        Returns:
            float: Giá trị TCP (0-1)
        """
        # Lấy dữ liệu từ DVH
        doses = dvh_data['dose']
        volumes = dvh_data['volume']
        is_relative = dvh_data.get('relative_volume', True)
    
        # Nếu là DVH tích lũy, chuyển thành DVH vi phân
        if dvh_data.get('type', 'cumulative') == 'cumulative':
            # Tính delta volume
            diff_volumes = np.abs(np.diff(volumes, append=0))
        else:
            diff_volumes = volumes
    
        # Nếu thể tích là phần trăm, cần chuyển sang thể tích tuyệt đối (cm^3)
        if is_relative:
            if tumor_volume is None:
                raise ValueError("Tumor volume must be provided when DVH uses relative volume")
            actual_volumes = diff_volumes * tumor_volume / 100.0
        else:
            # Giả định volumes đã là cm^3
            actual_volumes = diff_volumes
    
        # Liều mỗi phân liều (Gy)
        dose_per_fraction = doses / num_fractions
    
        # Tính EQD2 cho mỗi bin
        eqd2 = doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)

        # Tính số tế bào sống sót theo mô hình LQ
        survival = np.exp(-alpha * eqd2)
    
        # Tính số tế bào sinh ung thư ban đầu trong mỗi bin
        initial_cells = clonogenic_density * actual_volumes
    
        # Tính số tế bào sinh ung thư sống sót
        surviving_cells = np.sum(initial_cells * survival)
    
        # Tính TCP theo mô hình Poisson
        tcp = np.exp(-surviving_cells)
    
        return tcp

    def calculate_tcp_niemierko(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        tcd50: float = 50.0,
        gamma50: float = 2.0,
        alpha_beta: float = 10.0,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán TCP dựa trên mô hình Niemierko.
    
        TCP = 1 / (1 + (TCD50/EUD)^(4*gamma50))
    
        với EUD = (sum_i(v_i * D_i^a))^(1/a)
    
        Parameters:
            dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
            structure_mask (np.ndarray): Mảng mask 3D của khối u
            num_fractions (int): Số phân liều
            tcd50 (float, optional): Liều cần thiết để đạt TCP = 50% (Gy)
            gamma50 (float, optional): Độ dốc của đường cong liều-đáp ứng tại TCP = 50%
            alpha_beta (float, optional): Tỉ lệ alpha/beta (Gy)
            dose_threshold (float, optional): Ngưỡng liều để tính TCP, voxel có liều < ngưỡng sẽ bị bỏ qua
    
        Returns:
            float: Giá trị TCP (0-1)
        """
        # Kiểm tra mask và dose có cùng kích thước
        if dose_array.shape != structure_mask.shape:
            raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
        # Chỉ xét các voxel trong khối u
        mask = structure_mask > 0
        if dose_threshold is not None:
            mask = mask & (dose_array >= dose_threshold)
    
        # Nếu không có voxel nào thỏa điều kiện
        if not np.any(mask):
            logger.warning("No valid voxels found for TCP calculation")
            return 0.0
    
        # Lấy liều tại các voxel trong khối u
        tumor_doses = dose_array[mask]
    
        # Liều mỗi phân liều (Gy)
        dose_per_fraction = tumor_doses / num_fractions
    
        # Tính EQD2 cho mỗi voxel
        eqd2 = tumor_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
        # Tính tỉ lệ thể tích của mỗi voxel (tất cả voxel có cùng kích thước)
        num_voxels = np.sum(mask)
        volume_fraction = 1.0 / num_voxels
    
        # Tham số a cho tính EUD của khối u
        a = 1.0  # Giá trị dương, đặc trưng cho khối u
    
        # Tính EUD theo công thức Niemierko
        eud = np.power(np.sum(volume_fraction * np.power(eqd2, a)), 1.0/a)
    
        # Tính TCP theo mô hình Niemierko
        tcp = 1.0 / (1.0 + np.power(tcd50/eud, 4.0*gamma50))
    
        return tcp

    def calculate_tcp_logistic(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        d50: float = 50.0,
        k: float = 5.0,
        alpha_beta: float = 10.0,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán TCP dựa trên mô hình hàm Logistic.

        TCP = 1 / (1 + exp(-4*gamma50/D50 * (D - D50)))
    
        Parameters:
            dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
            structure_mask (np.ndarray): Mảng mask 3D của khối u
            num_fractions (int): Số phân liều
            d50 (float, optional): Liều cần thiết để đạt TCP = 50% (Gy)
            k (float, optional): Tham số k trong mô hình Logistic, liên quan đến độ dốc
            alpha_beta (float, optional): Tỉ lệ alpha/beta (Gy)
            dose_threshold (float, optional): Ngưỡng liều để tính TCP, voxel có liều < ngưỡng sẽ bị bỏ qua
    
        Returns:
            float: Giá trị TCP (0-1)
        """
        # Kiểm tra mask và dose có cùng kích thước
        if dose_array.shape != structure_mask.shape:
            raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
        # Chỉ xét các voxel trong khối u
        mask = structure_mask > 0
        if dose_threshold is not None:
            mask = mask & (dose_array >= dose_threshold)
    
        # Nếu không có voxel nào thỏa điều kiện
        if not np.any(mask):
            logger.warning("No valid voxels found for TCP calculation")
            return 0.0
    
        # Lấy liều tại các voxel trong khối u
        tumor_doses = dose_array[mask]
    
        # Liều mỗi phân liều (Gy)
        dose_per_fraction = tumor_doses / num_fractions
    
        # Tính EQD2 cho mỗi voxel
        eqd2 = tumor_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
        # Tính liều trung bình
        mean_dose = np.mean(eqd2)
    
        # Tính TCP theo mô hình Logistic
        tcp = 1.0 / (1.0 + np.exp(-4 * k * (mean_dose - d50) / d50))
    
        return tcp

    def calculate_tcp_webb(
        dose_array: np.ndarray,
        structure_mask: np.ndarray,
        num_fractions: int,
        alpha: float = 0.3,
        alpha_std: float = 0.1,
        alpha_beta: float = 10.0,
        clonogenic_density: float = 1e7,
        dose_threshold: Optional[float] = None
    ) -> float:
        """
        Tính toán TCP dựa trên mô hình Webb có tính đến tính không đồng nhất của tế bào.
    
        Parameters:
            dose_array (np.ndarray): Mảng phân bố liều 3D (Gy)
            structure_mask (np.ndarray): Mảng mask 3D của khối u
            num_fractions (int): Số phân liều
            alpha (float, optional): Giá trị trung bình của hệ số alpha (Gy^-1)
            alpha_std (float, optional): Độ lệch chuẩn của hệ số alpha (Gy^-1)
            alpha_beta (float, optional): Tỉ lệ alpha/beta (Gy)
            clonogenic_density (float, optional): Mật độ tế bào sinh ung thư (cells/cm^3)
            dose_threshold (float, optional): Ngưỡng liều để tính TCP, voxel có liều < ngưỡng sẽ bị bỏ qua
    
        Returns:
            float: Giá trị TCP (0-1)
        """
        # Kiểm tra mask và dose có cùng kích thước
        if dose_array.shape != structure_mask.shape:
            raise ValueError(f"Dose array shape {dose_array.shape} does not match structure mask shape {structure_mask.shape}")
    
        # Chỉ xét các voxel trong khối u
        mask = structure_mask > 0
        if dose_threshold is not None:
            mask = mask & (dose_array >= dose_threshold)
    
        # Nếu không có voxel nào thỏa điều kiện
        if not np.any(mask):
            logger.warning("No valid voxels found for TCP calculation")
            return 0.0
    
        # Lấy liều tại các voxel trong khối u
        tumor_doses = dose_array[mask]
    
        # Liều mỗi phân liều (Gy)
        dose_per_fraction = tumor_doses / num_fractions
    
        # Tính EQD2 cho mỗi voxel
        eqd2 = tumor_doses * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
    
        # Tính thể tích khối u (cm^3)
        # Giả định spacing là 1mm x 1mm x 1mm
        tumor_volume = np.sum(mask) / 1000.0  # Chuyển từ mm^3 sang cm^3
    
        # Tính số tế bào sinh ung thư ban đầu
        initial_cells = clonogenic_density * tumor_volume
    
        # Tính TCP với phân bố Gaussian của alpha
        # Sử dụng phương pháp Monte Carlo để tính tích phân
        num_samples = 1000
        alpha_samples = np.random.normal(alpha, alpha_std, num_samples)
    
        # Tính TCP cho mỗi giá trị alpha
        tcp_samples = np.zeros(num_samples)
        for i, a in enumerate(alpha_samples):
            if a <= 0:
                continue  # Bỏ qua nếu alpha <= 0
        
            # Tính số tế bào sống sót theo mô hình LQ
            survival = np.exp(-a * eqd2)
        
            # Tính số tế bào sinh ung thư sống sót trung bình
            surviving_cells = initial_cells * np.mean(survival)
        
            # Tính TCP theo mô hình Poisson
            tcp_samples[i] = np.exp(-surviving_cells)
    
    # Tính TCP trung bình
        tcp = np.mean(tcp_samples)
    
        return tcp
