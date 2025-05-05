# Changelog

## [Unreleased]

### Added
- Thêm module margin mới trong quangtps/segmentation/contour/margin.py:
  - Triển khai MarginGenerator với 4 loại margin: đồng đều, không đồng đều, vòng và bề mặt
  - Tối ưu hóa bằng OpenCV (cv2) để xử lý nhanh hơn
  - Hỗ trợ pixel spacing cho độ chính xác cao
- Tạo MarginToolWidget trong quangtps/segmentation/contour/margin_tool_widget.py:
  - Giao diện người dùng hiện đại cho công cụ margin
  - Hỗ trợ xem trước kết quả margin
  - Hỗ trợ tạo cấu trúc mới hoặc cập nhật cấu trúc hiện có
  - Việt hóa hoàn toàn giao diện
- Cải thiện DVHWidget trong quangtps/ui/widgets/dvh_widget.py:
  - Thêm hàm get_structure_color để lấy màu chính xác từ cấu trúc
  - Hỗ trợ so sánh nhiều kế hoạch xạ trị
  - Thêm bảng thống kê với Dmin, Dmax, Dmean, D95, V20Gy, V30Gy
  - Hỗ trợ xuất biểu đồ dưới dạng PNG, PDF và SVG
  - Việt hóa hoàn toàn giao diện

### Changed
- Cải tiến thuật toán AAA (Anisotropic Analytical Algorithm):
  - Khắc phục lỗi thừa kế trong lớp AAADoseCalculation
  - Triển khai tính toán song song để tăng tốc 3-4 lần
  - Cải thiện kernel chuyển đổi TERMA sang liều
- Cập nhật xử lý ngoại lệ trong script_to_update_project.py:
  - Thêm xử lý ngoại lệ khi import PyQt5 và matplotlib
  - Tạo các lớp dự phòng khi thiếu thư viện

### Fixed
- Sửa lỗi trong margin_tool_widget.py khi tạo cấu trúc mới
- Sửa lỗi trong DVHWidget khi cập nhật biểu đồ với dữ liệu mới
- Sửa lỗi trong AAA.py khi tính toán liều với giá trị âm

## [0.1.0] - 2023-05-15

### Added
- Phiên bản đầu tiên của QuangTPS
- Các tính năng cơ bản của hệ thống lập kế hoạch xạ trị