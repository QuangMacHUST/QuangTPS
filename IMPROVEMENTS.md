# Plan Quality Evaluation Improvements

## Overview

We have implemented a comprehensive plan quality evaluation system for QuangTPS, mimicking the functionality available in Eclipse TPS. The system includes clinical protocol management, goal evaluation, and robust reporting capabilities.

## Key Features

### Clinical Protocol Management

- **Protocol Storage**: Protocols are stored as JSON files in the `protocols` directory
- **Protocol Dialog**: A dialog for selecting, importing, and exporting protocols
- **Protocol Editor**: A dedicated editor for creating and modifying protocols, with support for:
  - Editing protocol details (name, description)
  - Adding, editing, and removing clinical goals
  - Importing goals from other protocols

### Plan Quality Evaluation

- **Goal Evaluation**: Each clinical goal is evaluated against actual dose metrics
- **Scoring System**: Overall, target, and OAR scores calculated based on goal achievements
- **Visual Feedback**: Color-coded indicators for passed, acceptable, and failed goals

### Reporting

- **HTML Reports**: Generate comprehensive HTML reports showing:
  - Overall quality scores
  - Individual goal results
  - Progress bars and visual indicators
  - Summary assessment

## Integration with QuangTPS

The plan quality functionality is fully integrated with the existing evaluation tab, allowing users to:

1. Select clinical protocols from a dropdown menu or dedicated dialog
2. View plan quality assessment in a dedicated tab
3. Generate and export plan quality reports
4. Edit protocols directly from the interface

## Sample Protocols

Two sample protocols are included:

1. **Head and Neck**: Containing goals for PTVs, spinal cord, brainstem, parotids, and other OARs
2. **Prostate**: Containing goals for PTV, rectum, bladder, femoral heads, and penile bulb

## Eclipse-Like Experience

The interface closely resembles Eclipse's plan quality evaluation system, with:

- Similar scoring visualization
- Progress bars for goal achievement
- Color-coded indicators (green, orange, red)
- Detailed goal display table

## Technical Implementation

- **Modular Design**: Separated protocol management, evaluation, and UI components
- **Exception Handling**: Robust error handling for missing components
- **Clean Integration**: Minimal changes to existing code while adding significant functionality

# Kế hoạch cải thiện QuangTPS

## Cải tiến mới nhất (v0.8.0)

### Hiển thị liều 3D nâng cao và tích hợp Eclipse-style
- [x] Tạo module hiển thị liều 3D mới với đồ họa VTK cao cấp
  - Hiển thị chất lượng cao của isodose surfaces với màu sắc tùy chỉnh theo kiểu Eclipse
  - Hỗ trợ nhiều chế độ hiển thị: surface, volume rendering, MIP, X-Ray
  - Hệ thống góc nhìn tiêu chuẩn (anterior, posterior, left, right, superior, inferior)
  - Tương tác trực tiếp với isodose levels và cấu trúc 3D
  - Hệ thống dự phòng đa lớp với matplotlib khi VTK không khả dụng

- [x] Tích hợp hiển thị 3D với tab External Beam Planning
  - Kết nối liền mạch dữ liệu liều, cấu trúc và hiển thị 3D
  - Giao diện Eclipse-style với các chức năng điều khiển trực quan
  - Xử lý ngoại lệ và dự phòng để đảm bảo hoạt động ổn định

- [x] Khắc phục lỗi trong các module hiện có
  - Sửa lỗi định nghĩa lớp trùng lặp và tham số hàm không hợp lệ
  - Cải thiện cơ chế dự phòng khi thiếu module
  - Thiết lập nền tảng cho phiên bản tiếp theo

## Nhiệm vụ ưu tiên cao
- [x] Hoàn thành module cơ bản cho lập kế hoạch xạ trị
- [x] Hoàn thiện giao diện người dùng chính
- [x] Tích hợp các thuật toán tính toán liều cơ bản
- [x] Hỗ trợ định dạng DICOM RT đầy đủ
- [x] Cải thiện hệ thống báo cáo và xuất kết quả
- [x] Nâng cao hệ thống lập kế hoạch thích ứng (adaptive planning)
- [x] Tăng cường hệ thống xử lý lỗi và ngoại lệ
- [ ] Cải thiện hiệu suất tính toán liều cho kích thước lớn
- [ ] Hoàn thiện hệ thống tối ưu hóa đa tiêu chí (MCO)
- [ ] Triển khai đầy đủ hệ thống đánh giá chất lượng kế hoạch

## Nhiệm vụ ưu tiên trung bình
- [x] Thêm các công cụ để đánh giá kế hoạch
- [x] Tạo hệ thống template linh hoạt
- [x] Cải thiện hiệu suất tính toán trên CPU
- [x] Tích hợp thuật toán Monte Carlo trên GPU
- [x] Nâng cao độ tin cậy của hệ thống
- [x] Cải thiện trải nghiệm người dùng và giao diện
- [x] Xây dựng hệ thống dự đoán thay đổi giải phẫu
- [x] Hoàn thiện module lập kế hoạch mạnh mẽ (robust planning)
- [ ] Triển khai công cụ phân tích thống kê cao cấp
- [ ] Thêm khả năng dự đoán kết quả điều trị

## Nhiệm vụ dài hạn
- [x] Xây dựng hệ thống plugin mở rộng
- [x] Phát triển kho thuật toán tính liều nâng cao
- [ ] Tạo công cụ tối ưu hóa dựa trên học máy
- [ ] Phát triển hệ thống đào tạo tích hợp
- [ ] Xây dựng hệ thống phê duyệt kế hoạch tự động
- [ ] Cải thiện khả năng tương thích với các hệ thống khác
- [ ] Phát triển hệ thống hoạch định phác đồ tự động

## Kế hoạch phát triển tiếp theo (v0.8.1 - v0.9.0)

### Ưu tiên cao
- [ ] Tổ chức thư viện nâng cao MCO (Multi-Criteria Optimization) tương tự Eclipse:
  - Khả năng tạo và đánh giá nhanh nhiều kế hoạch tối ưu với Pareto surface
  - Giao diện trực quan với thanh trượt để điều chỉnh trọng số giữa các mục tiêu
  - Tối ưu hóa thực theo thời gian thực với kéo thả đường đồng liều
  - Tích hợp với RapidPlan (hoặc tương đương) để tự động tạo kế hoạch ban đầu

- [ ] Cải thiện hiệu suất GPU cho tính toán liều và hiển thị:
  - API thống nhất cho tính toán GPU/CPU với dự phòng thông minh
  - Tối ưu hóa thuật toán Monte Carlo với CUDA/OpenCL
  - Giảm sử dụng bộ nhớ cho dữ liệu lớn với kỹ thuật streaming và tải theo yêu cầu
  - Tối ưu hóa hiển thị 3D với ray casting GPU và các kỹ thuật tăng tốc khác

- [ ] Hoàn thiện bộ công cụ đánh giá kế hoạch:
  - Giao diện so sánh kế hoạch tương tự Eclipse
  - Phân tích DVH với tính toán tự động các chỉ số lâm sàng (EUD, TCP/NTCP, V95, D98...)
  - Hệ thống kiểm tra kế hoạch (Plan Checker) với quy tắc có thể tùy chỉnh
  - Báo cáo chất lượng kế hoạch tự động theo nhiều định dạng

### Mô phỏng Eclipse của Varian
Dựa trên nghiên cứu về Eclipse của Varian, các tính năng chính sau cần được triển khai:

1. **RapidPlan** - Hệ thống lập kế hoạch dựa trên kiến thức (Knowledge-based planning)
2. **MCO Navigator** - Giao diện tối ưu hóa đa tiêu chí với Pareto surface explorer
3. **RT Peer Review** - Hệ thống đánh giá và phê duyệt kế hoạch tương tác
4. **Plan Checker** - Công cụ kiểm tra kế hoạch tự động với quy tắc có thể tùy chỉnh
5. **RapidArc Dynamic** - Công cụ tối ưu hóa VMAT với mô hình lá động (dynamic leaf)

Tất cả các tính năng này cần được phát triển trong các phiên bản tiếp theo để đạt được tính năng đầy đủ và giao diện tương tự Eclipse của Varian.