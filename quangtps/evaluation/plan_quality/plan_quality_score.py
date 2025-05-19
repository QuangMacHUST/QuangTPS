from enum import Enum, auto


class PlanQualityScore(Enum):
    """
    Định nghĩa các mức đánh giá chất lượng kế hoạch xạ trị.

    Các mức đánh giá tương thích với hệ thống Eclipse của Varian,
    sử dụng các màu sắc và mức độ khác nhau để đánh giá kế hoạch.
    """

    EXCELLENT = auto()  # Xuất sắc: đạt hoặc vượt tất cả mục tiêu lâm sàng quan trọng
    GOOD = auto()  # Tốt: đạt phần lớn mục tiêu lâm sàng quan trọng
    ACCEPTABLE = (
        auto()
    )  # Chấp nhận được: đạt các mục tiêu cơ bản, một số mục tiêu có thể không đạt
    MARGINAL = auto()  # Cận biên: một số mục tiêu quan trọng không đạt
    POOR = auto()  # Kém: nhiều mục tiêu quan trọng không đạt
    NOT_EVALUATED = auto()  # Chưa đánh giá

    def __str__(self):
        """Trả về biểu diễn chuỗi thân thiện với người dùng."""
        return {
            PlanQualityScore.EXCELLENT: "Xuất sắc",
            PlanQualityScore.GOOD: "Tốt",
            PlanQualityScore.ACCEPTABLE: "Chấp nhận được",
            PlanQualityScore.MARGINAL: "Cận biên",
            PlanQualityScore.POOR: "Kém",
            PlanQualityScore.NOT_EVALUATED: "Chưa đánh giá",
        }[self]

    def get_color(self):
        """Trả về mã màu dạng hex cho mức đánh giá."""
        return {
            PlanQualityScore.EXCELLENT: "#4CAF50",  # Xanh lá đậm
            PlanQualityScore.GOOD: "#8BC34A",  # Xanh lá nhạt
            PlanQualityScore.ACCEPTABLE: "#FFEB3B",  # Vàng
            PlanQualityScore.MARGINAL: "#FF9800",  # Cam
            PlanQualityScore.POOR: "#F44336",  # Đỏ
            PlanQualityScore.NOT_EVALUATED: "#9E9E9E",  # Xám
        }[self]

    def get_score_value(self):
        """Trả về giá trị số cho mức đánh giá."""
        return {
            PlanQualityScore.EXCELLENT: 95.0,
            PlanQualityScore.GOOD: 85.0,
            PlanQualityScore.ACCEPTABLE: 75.0,
            PlanQualityScore.MARGINAL: 65.0,
            PlanQualityScore.POOR: 50.0,
            PlanQualityScore.NOT_EVALUATED: 0.0,
        }[self]

    @classmethod
    def from_percentage(cls, percentage):
        """
        Chuyển đổi giá trị phần trăm thành mức đánh giá.

        Args:
            percentage (float): Phần trăm mục tiêu đạt được (0-100)

        Returns:
            PlanQualityScore: Mức đánh giá tương ứng
        """
        if percentage >= 95:
            return cls.EXCELLENT
        elif percentage >= 85:
            return cls.GOOD
        elif percentage >= 75:
            return cls.ACCEPTABLE
        elif percentage >= 65:
            return cls.MARGINAL
        elif percentage > 0:
            return cls.POOR
        else:
            return cls.NOT_EVALUATED
