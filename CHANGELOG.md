# Nhật ký thay đổi QuangTPS

Tất cả các thay đổi đáng chú ý trong dự án sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án này tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.6] - 2025-05-25

### Added
- Tích hợp AcurosXB GPU cho tính toán liều:
  - Tạo module `quangtps/dose/algorithms/acuros_gpu.py` mới
  - Tích hợp tính toán song song trên GPU với CUDA
  - Tối ưu hóa giải LBTE để đạt hiệu suất cao
  - Hỗ trợ cả chế độ dự phòng CPU nếu không có GPU
  - Ứng dụng các CUDA kernel tùy chỉnh để tăng tốc các hoạt động quan trọng

- Nâng cấp hệ thống tính toán liều trong `quangtps/dose/dose_engine.py`:
  - Hỗ trợ GPU cho các thuật toán tính toán liều
  - Thêm các API đăng ký thuật toán tính toán liều
  - Cung cấp khả năng kiểm tra sự hỗ trợ GPU cho mỗi thuật toán
  - Tự động tải và đăng ký các thuật toán có sẵn
  - Cải thiện quản lý lỗi trong quá trình tính toán

- Cập nhật dialog tính toán liều cho phép người dùng chọn GPU/CPU:
  - Hiển thị các thuật toán hỗ trợ GPU
  - Tùy chọn chế độ tính toán (CPU/GPU)
  - Tùy chọn số luồng xử lý
  - Báo cáo tốc độ tính toán và tăng tốc so với CPU

### Fixed
- Sửa lỗi memory leak khi tính toán liều lặp lại nhiều lần
- Đảm bảo giải phóng bộ nhớ GPU sau khi hoàn thành tính toán
- Xử lý các trường hợp ngoại lệ khi không tìm thấy GPU
- Tối ưu hóa quản lý bộ nhớ cho các thuật toán tính toán nặng

### Changed
- Cải thiện hiệu suất tổng thể cho tính toán liều
- Thêm chế độ tính toán chính xác cao cho Acuros XB
- Cập nhật tài liệu hướng dẫn sử dụng GPU cho tính toán liều

## [0.6.4] - 2025-05-22

### Added
- Tạo module thuật toán Monte Carlo trong `quangtps/dose/algorithms/montecarlo.py`:
  - Triển khai lớp MonteCarloAlgorithm để tính toán liều chính xác hơn
  - Thêm các phương thức mô phỏng vật lý hạt như photon và electron
  - Hỗ trợ tính toán đa luồng để tăng hiệu suất
  - Tính toán và hiển thị độ không chắc chắn của kết quả Monte Carlo

- Tạo module tính toán chỉ số chất lượng kế hoạch trong `quangtps/evaluation/metrics/quality_metrics.py`:
  - Triển khai các chỉ số CI, HI, GI cho đánh giá kế hoạch xạ trị
  - Thêm tính năng tính toán các chỉ số sinh học TCP, NTCP
  - Hỗ trợ đánh giá toàn diện kế hoạch điều trị

### Fixed
- Sửa lỗi trong `quangtps/optimization/constraints.py` liên quan đến các tham số DVH
- Sửa lỗi xung đột hàm `create_optimizer` trong `quangtps/optimization/__init__.py`

### Changed
- Cải thiện tính ổn định của hệ thống tối ưu hóa
- Nâng cao khả năng tính toán DVH với nhiều phương pháp dự phòng
- Cập nhật tài liệu README.md với thông tin về phiên bản và tính năng mới

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