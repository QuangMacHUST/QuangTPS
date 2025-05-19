from enum import Enum, auto


class GoalType(Enum):
    """
    Định nghĩa các loại mục tiêu lâm sàng cho đánh giá kế hoạch xạ trị.

    Các loại mục tiêu này tương thích với hệ thống Eclipse của Varian,
    bao gồm đầy đủ các loại mục tiêu cho cả mục tiêu và OAR.
    """

    # Mục tiêu liều-thể tích và thể tích-liều
    DOSE_VOLUME = auto()  # Dx% ≥/≤ y Gy: Liều nhận bởi x% thể tích
    VOLUME_DOSE = auto()  # Vx Gy ≥/≤ y%: Thể tích nhận ít nhất x Gy

    # Các chỉ số thống kê liều
    MEAN_DOSE = auto()  # Dmean ≥/≤ x Gy: Liều trung bình
    MAX_DOSE = auto()  # Dmax ≥/≤ x Gy: Liều tối đa (thường là D0.03cc)
    MIN_DOSE = auto()  # Dmin ≥/≤ x Gy: Liều tối thiểu (thường là D98% hoặc D99%)
    MEDIAN_DOSE = auto()  # D50% ≥/≤ x Gy: Liều trung vị

    # Các chỉ số đánh giá kế hoạch tiên tiến
    HOMOGENEITY_INDEX = auto()  # HI = (D2% - D98%) / D50%: Chỉ số đồng nhất
    CONFORMITY_INDEX = auto()  # CI = (V95% / VPTV): Chỉ số phù hợp
    GRADIENT_INDEX = auto()  # GI = (V50% / V100%): Chỉ số độ dốc
    PADDICK_CI = auto()  # Paddick CI = (TVPIV)² / (TV × PIV)

    # Liều sinh học tương đương (BED) và các chỉ số radiobiological
    BED = auto()  # Liều sinh học tương đương
    EQD2 = auto()  # Liều tương đương 2Gy
    EUD = auto()  # Liều đồng nhất tương đương
    TCP = auto()  # Xác suất kiểm soát khối u
    NTCP = auto()  # Xác suất biến chứng mô bình thường

    # Các mục tiêu phức tạp
    DOSE_FALLOFF = auto()  # Độ giảm liều
    HOT_SPOT = auto()  # Điểm nóng liều cao
    COLD_SPOT = auto()  # Điểm lạnh liều thấp

    # Các chỉ số cho SRS/SBRT
    RTOG_CONFORMITY_INDEX = auto()  # RTOG CI = PIV / TV
    GLOBAL_MAX_DOSE = auto()  # Liều tối đa toàn bộ kế hoạch

    # Các chỉ số tích hợp
    COMPOSITE_INDEX = auto()  # Chỉ số tổng hợp từ nhiều chỉ số khác

    def __str__(self):
        """Trả về biểu diễn chuỗi thân thiện với người dùng."""
        return {
            GoalType.DOSE_VOLUME: "Liều-Thể tích (Dx%)",
            GoalType.VOLUME_DOSE: "Thể tích-Liều (Vx Gy)",
            GoalType.MEAN_DOSE: "Liều trung bình",
            GoalType.MAX_DOSE: "Liều tối đa",
            GoalType.MIN_DOSE: "Liều tối thiểu",
            GoalType.MEDIAN_DOSE: "Liều trung vị",
            GoalType.HOMOGENEITY_INDEX: "Chỉ số đồng nhất",
            GoalType.CONFORMITY_INDEX: "Chỉ số phù hợp",
            GoalType.GRADIENT_INDEX: "Chỉ số độ dốc",
            GoalType.PADDICK_CI: "Chỉ số phù hợp Paddick",
            GoalType.BED: "Liều sinh học tương đương",
            GoalType.EQD2: "Liều tương đương 2Gy",
            GoalType.EUD: "Liều đồng nhất tương đương",
            GoalType.TCP: "Xác suất kiểm soát khối u",
            GoalType.NTCP: "Xác suất biến chứng mô bình thường",
            GoalType.DOSE_FALLOFF: "Độ giảm liều",
            GoalType.HOT_SPOT: "Điểm nóng",
            GoalType.COLD_SPOT: "Điểm lạnh",
            GoalType.RTOG_CONFORMITY_INDEX: "Chỉ số phù hợp RTOG",
            GoalType.GLOBAL_MAX_DOSE: "Liều tối đa toàn cục",
            GoalType.COMPOSITE_INDEX: "Chỉ số tổng hợp",
        }[self]
