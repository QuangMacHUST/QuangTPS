#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp các hàm tính toán hệ số Dice và các phép đo tương tự.

Module này cung cấp các chức năng để tính toán độ tương đồng giữa các contour
sử dụng hệ số Dice và các phép đo khác. Các hàm này đặc biệt hữu ích trong việc
đánh giá kế hoạch thích ứng và so sánh các contour tự động với contour tham chiếu.
"""

import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any, Set, Sequence

from quangtps.core.types import Structure


def calculate_dice_coefficient(
    structure_a: Union[Structure, np.ndarray],
    structure_b: Union[Structure, np.ndarray],
    mask: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
) -> float:
    """
    Tính toán hệ số Dice giữa hai cấu trúc hoặc mặt nạ nhị phân.

    Hệ số Dice được tính theo công thức: 2*|A∩B|/(|A|+|B|)
    Hệ số này có giá trị trong khoảng [0, 1], với 1 là trùng khớp hoàn toàn.

    Parameters
    ----------
    structure_a : Union[Structure, np.ndarray]
        Cấu trúc đầu tiên hoặc mặt nạ nhị phân
    structure_b : Union[Structure, np.ndarray]
        Cấu trúc thứ hai hoặc mặt nạ nhị phân để so sánh
    mask : Optional[np.ndarray], optional
        Vùng cần so sánh (nếu không phải toàn bộ ảnh), mặc định là None
    tolerance : float, optional
        Giá trị dung sai để tránh lỗi chia cho 0, mặc định là 1e-6

    Returns
    -------
    float
        Hệ số Dice trong khoảng [0, 1]

    Raises
    ------
    ValueError
        Nếu cấu trúc đầu vào không tương thích
    """
    # Chuyển đổi cấu trúc thành mảng nhị phân
    if isinstance(structure_a, Structure):
        mask_a = structure_a.get_binary_mask()
    else:
        mask_a = structure_a.astype(bool)

    if isinstance(structure_b, Structure):
        mask_b = structure_b.get_binary_mask()
    else:
        mask_b = structure_b.astype(bool)

    # Kiểm tra kích thước
    if mask_a.shape != mask_b.shape:
        raise ValueError("Cấu trúc đầu vào phải có cùng kích thước")

    # Áp dụng mask nếu có
    if mask is not None:
        if mask.shape != mask_a.shape:
            raise ValueError("Mask phải có cùng kích thước với cấu trúc")
        mask_a = np.logical_and(mask_a, mask)
        mask_b = np.logical_and(mask_b, mask)

    # Tính giao và tổng
    intersection = np.logical_and(mask_a, mask_b).sum()
    sum_masks = mask_a.sum() + mask_b.sum()

    # Tính hệ số Dice
    if sum_masks < tolerance:
        return 0.0  # Trường hợp cả hai mặt nạ đều rỗng

    return 2.0 * intersection / (sum_masks + tolerance)


def calculate_jaccard_index(
    structure_a: Union[Structure, np.ndarray],
    structure_b: Union[Structure, np.ndarray],
    mask: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
) -> float:
    """
    Tính toán chỉ số Jaccard (IoU - Intersection over Union) giữa hai cấu trúc.

    Chỉ số Jaccard được tính theo công thức: |A∩B|/|A∪B|
    Chỉ số này có giá trị trong khoảng [0, 1], với 1 là trùng khớp hoàn toàn.

    Parameters
    ----------
    structure_a : Union[Structure, np.ndarray]
        Cấu trúc đầu tiên hoặc mặt nạ nhị phân
    structure_b : Union[Structure, np.ndarray]
        Cấu trúc thứ hai hoặc mặt nạ nhị phân để so sánh
    mask : Optional[np.ndarray], optional
        Vùng cần so sánh (nếu không phải toàn bộ ảnh), mặc định là None
    tolerance : float, optional
        Giá trị dung sai để tránh lỗi chia cho 0, mặc định là 1e-6

    Returns
    -------
    float
        Chỉ số Jaccard trong khoảng [0, 1]
    """
    # Chuyển đổi cấu trúc thành mảng nhị phân
    if isinstance(structure_a, Structure):
        mask_a = structure_a.get_binary_mask()
    else:
        mask_a = structure_a.astype(bool)

    if isinstance(structure_b, Structure):
        mask_b = structure_b.get_binary_mask()
    else:
        mask_b = structure_b.astype(bool)

    # Kiểm tra kích thước
    if mask_a.shape != mask_b.shape:
        raise ValueError("Cấu trúc đầu vào phải có cùng kích thước")

    # Áp dụng mask nếu có
    if mask is not None:
        if mask.shape != mask_a.shape:
            raise ValueError("Mask phải có cùng kích thước với cấu trúc")
        mask_a = np.logical_and(mask_a, mask)
        mask_b = np.logical_and(mask_b, mask)

    # Tính giao và hợp
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    # Tính chỉ số Jaccard
    if union < tolerance:
        return 0.0  # Trường hợp cả hai mặt nạ đều rỗng

    return intersection / (union + tolerance)


def calculate_hausdorff_distance(
    structure_a: Union[Structure, np.ndarray],
    structure_b: Union[Structure, np.ndarray],
    voxel_spacing: Optional[Tuple[float, float, float]] = None,
    percentile: float = 95.0,
) -> float:
    """
    Tính khoảng cách Hausdorff giữa hai cấu trúc.

    Khoảng cách Hausdorff là khoảng cách lớn nhất từ một điểm trong tập hợp A
    đến điểm gần nhất trong tập hợp B.

    Parameters
    ----------
    structure_a : Union[Structure, np.ndarray]
        Cấu trúc đầu tiên hoặc mặt nạ nhị phân
    structure_b : Union[Structure, np.ndarray]
        Cấu trúc thứ hai hoặc mặt nạ nhị phân để so sánh
    voxel_spacing : Optional[Tuple[float, float, float]], optional
        Khoảng cách voxel (mm) trong không gian 3D, mặc định là None (1mm)
    percentile : float, optional
        Phần trăm để tính khoảng cách Hausdorff, mặc định là 95

    Returns
    -------
    float
        Khoảng cách Hausdorff theo đơn vị mm

    Raises
    ------
    ImportError
        Nếu scipy không được cài đặt
    """
    try:
        from scipy.ndimage import distance_transform_edt
        from scipy.stats import scoreatpercentile
    except ImportError:
        raise ImportError(
            "Phụ thuộc scipy không được cài đặt. Cài đặt với 'pip install scipy'"
        )

    # Chuyển đổi cấu trúc thành mảng nhị phân
    if isinstance(structure_a, Structure):
        mask_a = structure_a.get_binary_mask()
        if voxel_spacing is None:
            voxel_spacing = structure_a.get_voxel_spacing()
    else:
        mask_a = structure_a.astype(bool)

    if isinstance(structure_b, Structure):
        mask_b = structure_b.get_binary_mask()
        if voxel_spacing is None:
            voxel_spacing = structure_b.get_voxel_spacing()
    else:
        mask_b = structure_b.astype(bool)

    # Kiểm tra kích thước
    if mask_a.shape != mask_b.shape:
        raise ValueError("Cấu trúc đầu vào phải có cùng kích thước")

    # Mặc định voxel spacing nếu không được cung cấp
    if voxel_spacing is None:
        voxel_spacing = (1.0, 1.0, 1.0)

    # Tính bản đồ khoảng cách từ A đến B và ngược lại
    dist_a_to_b = distance_transform_edt(~mask_b, sampling=voxel_spacing)
    dist_b_to_a = distance_transform_edt(~mask_a, sampling=voxel_spacing)

    # Lấy khoảng cách từ A đến B
    a_to_b_distances = dist_a_to_b[mask_a]
    if len(a_to_b_distances) == 0:
        a_to_b_distances = np.array([0.0])

    # Lấy khoảng cách từ B đến A
    b_to_a_distances = dist_b_to_a[mask_b]
    if len(b_to_a_distances) == 0:
        b_to_a_distances = np.array([0.0])

    # Tính khoảng cách Hausdorff theo phần trăm
    hausdorff_a_to_b = scoreatpercentile(a_to_b_distances, percentile)
    hausdorff_b_to_a = scoreatpercentile(b_to_a_distances, percentile)

    # Khoảng cách Hausdorff là giá trị lớn nhất của hai khoảng cách
    return max(hausdorff_a_to_b, hausdorff_b_to_a)


def calculate_volume_overlap_metrics(
    structure_a: Union[Structure, np.ndarray],
    structure_b: Union[Structure, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Tính toán nhiều phép đo đánh giá độ trùng khớp giữa hai cấu trúc.

    Parameters
    ----------
    structure_a : Union[Structure, np.ndarray]
        Cấu trúc đầu tiên hoặc mặt nạ nhị phân
    structure_b : Union[Structure, np.ndarray]
        Cấu trúc thứ hai hoặc mặt nạ nhị phân để so sánh
    mask : Optional[np.ndarray], optional
        Vùng cần so sánh, mặc định là None

    Returns
    -------
    Dict[str, float]
        Từ điển chứa các phép đo khác nhau:
        - 'dice': Hệ số Dice
        - 'jaccard': Chỉ số Jaccard
        - 'vol_diff': Hiệu phần trăm thể tích
        - 'sensitivity': Độ nhạy (recall)
        - 'specificity': Độ đặc hiệu
        - 'precision': Độ chính xác
    """
    # Chuyển đổi cấu trúc thành mảng nhị phân
    if isinstance(structure_a, Structure):
        mask_a = structure_a.get_binary_mask()
    else:
        mask_a = structure_a.astype(bool)

    if isinstance(structure_b, Structure):
        mask_b = structure_b.get_binary_mask()
    else:
        mask_b = structure_b.astype(bool)

    # Kiểm tra kích thước
    if mask_a.shape != mask_b.shape:
        raise ValueError("Cấu trúc đầu vào phải có cùng kích thước")

    # Áp dụng mask nếu có
    if mask is not None:
        if mask.shape != mask_a.shape:
            raise ValueError("Mask phải có cùng kích thước với cấu trúc")
        mask_a = np.logical_and(mask_a, mask)
        mask_b = np.logical_and(mask_b, mask)

    # Tính các thành phần cơ bản
    tp = np.logical_and(mask_a, mask_b).sum()  # True positive
    fp = np.logical_and(mask_a, ~mask_b).sum()  # False positive
    fn = np.logical_and(~mask_a, mask_b).sum()  # False negative
    tn = np.logical_and(~mask_a, ~mask_b).sum()  # True negative

    # Tính các phép đo
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    vol_a = mask_a.sum()
    vol_b = mask_b.sum()
    vol_diff_percent = 100 * abs(vol_a - vol_b) / vol_b if vol_b > 0 else float("inf")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "dice": dice,
        "jaccard": jaccard,
        "vol_diff": vol_diff_percent,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }


def batch_calculate_dice(
    ref_structures: Dict[str, Structure], eval_structures: Dict[str, Structure]
) -> Dict[str, float]:
    """
    Tính toán hệ số Dice cho nhiều cặp cấu trúc cùng một lúc.

    Parameters
    ----------
    ref_structures : Dict[str, Structure]
        Từ điển các cấu trúc tham chiếu với khóa là tên cấu trúc
    eval_structures : Dict[str, Structure]
        Từ điển các cấu trúc cần đánh giá với khóa là tên cấu trúc

    Returns
    -------
    Dict[str, float]
        Từ điển chứa hệ số Dice cho mỗi cấu trúc
    """
    dice_scores = {}

    # Xử lý cho các cấu trúc chung
    common_structs = set(ref_structures.keys()).intersection(
        set(eval_structures.keys())
    )

    for struct_name in common_structs:
        try:
            dice_score = calculate_dice_coefficient(
                ref_structures[struct_name], eval_structures[struct_name]
            )
            dice_scores[struct_name] = dice_score
        except Exception as e:
            dice_scores[struct_name] = float("nan")
            print(f"Lỗi khi tính toán hệ số Dice cho {struct_name}: {str(e)}")

    return dice_scores
