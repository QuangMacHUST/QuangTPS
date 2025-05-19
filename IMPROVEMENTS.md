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

## Cải tiến mới nhất (v0.8.9)

### Hoàn thiện module đánh giá kế hoạch xạ trị tích hợp phong cách Eclipse

- [x] Tạo module báo cáo kế hoạch đa định dạng trong `quangtps/reporting/plan_report_generator.py`:
  - [x] Hỗ trợ tạo báo cáo PDF chuyên nghiệp với DVH, bảng mục tiêu và thông tin kế hoạch
  - [x] Tạo báo cáo HTML tương tác với khả năng hiển thị/lọc dữ liệu linh hoạt
  - [x] Xuất dữ liệu dạng CSV để phân tích ngoại tuyến hoặc nhập vào phần mềm khác
  - [x] So sánh nhiều kế hoạch trong cùng báo cáo với đồ thị chồng

- [x] Chuẩn hóa định nghĩa các loại mục tiêu lâm sàng trong `quangtps/evaluation/protocols/goal_type.py`:
  - [x] Các loại mục tiêu cơ bản (DOSE_VOLUME, VOLUME_DOSE) tương thích Eclipse
  - [x] Chỉ số thống kê liều (MEAN_DOSE, MAX_DOSE, MIN_DOSE)
  - [x] Chỉ số đánh giá nâng cao (HOMOGENEITY_INDEX, CONFORMITY_INDEX)
  - [x] Chỉ số đánh giá sinh học (TCP, NTCP, EUD)

- [x] Tạo hệ thống điểm đánh giá kế hoạch trong `quangtps/evaluation/plan_quality/plan_quality_score.py`:
  - [x] Mức điểm phù hợp Eclipse (EXCELLENT, GOOD, ACCEPTABLE, MARGINAL, POOR)
  - [x] Chuyển đổi tỷ lệ đạt mục tiêu thành điểm số tự động
  - [x] Mã màu trực quan từ đỏ (kém) đến xanh lá (xuất sắc)

- [x] Phát triển bộ đánh giá kế hoạch trong `quangtps/evaluation/plan_quality/plan_quality_evaluator.py`:
  - [x] Phương thức evaluate_with_protocol() đánh giá theo protocol lâm sàng
  - [x] Tính toán điểm số riêng cho mục tiêu (PTV) và cơ quan nguy cấp (OAR)
  - [x] Tính toán chỉ số HI, CI tự động cho đánh giá nâng cao
  - [x] Xuất kết quả đánh giá sang nhiều định dạng (JSON, CSV)

### Nâng cao độ tin cậy và tính ổn định của hệ thống

- [x] Cải thiện xử lý ngoại lệ và cơ chế dự phòng:
  - [x] Xử lý lỗi import với try-except và fallback class
  - [x] Tạo các lớp giả mạch khi module không khả dụng
  - [x] Cải thiện quản lý tài nguyên và giải phóng bộ nhớ
  - [x] Báo cáo lỗi chi tiết hơn để dễ dàng gỡ lỗi

- [x] Nâng cao khả năng hoạt động khi thiếu thư viện bên ngoài:
  - [x] Dự phòng khi không có weasyprint để tạo PDF
  - [x] Xử lý khi matplotlib không khả dụng
  - [x] Dự phòng khi không thể import numpy, pandas
  - [x] Đảm bảo hoạt động cơ bản ngay cả khi thiếu module nâng cao

## Cải tiến mới nhất (v0.8.8)

### Tạo module đánh giá kế hoạch xạ trị toàn diện

- [x] Tạo module `quangtps/ui/plan_evaluation_report_tab.py` với giao diện đánh giá toàn diện:
  - [x] Thiết kế giao diện phong cách Eclipse với bố cục tối ưu
  - [x] Tích hợp DVH, đánh giá mục tiêu lâm sàng, thống kê liều trong một giao diện
  - [x] Hiển thị điểm đánh giá với thang điểm và màu sắc trực quan
  - [x] Tích hợp chọn và chỉnh sửa protocol lâm sàng

- [x] Tạo `quangtps/ui/plan_comparison_dialog.py` để so sánh nhiều kế hoạch:
  - [x] Hiển thị DVH của nhiều kế hoạch trên cùng một biểu đồ
  - [x] Bảng so sánh các chỉ số đánh giá giữa các kế hoạch
  - [x] Phân tích chênh lệch giữa các kế hoạch
  - [x] Đề xuất kế hoạch tối ưu theo các tiêu chí khác nhau

- [x] Phát triển báo cáo kế hoạch đầy đủ:
  - [x] Xuất báo cáo PDF với đầy đủ thông tin kế hoạch và đánh giá
  - [x] Tạo báo cáo HTML tương tác với biểu đồ có thể điều chỉnh
  - [x] Xuất dữ liệu dạng CSV để phân tích ngoài hệ thống
  - [x] Tạo PDF tự động theo mẫu có sẵn

### Cải thiện tích hợp với các thành phần hiện có

- [x] Kết nối liền mạch với các thành phần core:
  - [x] Truy vấn dữ liệu từ DVHWidget và ClinicalGoalsWidget
  - [x] Tích hợp với protocol_manager để quản lý mục tiêu lâm sàng
  - [x] Kết nối với dose_grid và dose_calculator để truy xuất dữ liệu liều
  - [x] Tận dụng structure_manager để truy cập thông tin cấu trúc

- [x] Tích hợp với giao diện người dùng hiện có:
  - [x] Đảm bảo phong cách eclipse_style_theme nhất quán
  - [x] Duy trì tính liên tục khi chuyển đổi giữa các tab
  - [x] Sử dụng các biểu tượng và theme màu sắc tương tự
  - [x] Tối ưu hóa giao diện cho cả màn hình độ phân giải thấp và cao

## Cải tiến mới nhất (v0.8.7)

### Tích hợp hiển thị liều 3D vào External Beam Planning tab

- [x] Thay thế placeholder 3D hiện tại bằng DoseVisualization3D đầy đủ chức năng:
  - [x] Sử dụng VTK để hiển thị phân phối liều 3D với hiệu suất cao
  - [x] Hỗ trợ nhiều chế độ hiển thị: surface, volume rendering, MIP
  - [x] Hiển thị cấu trúc giải phẫu 3D với độ trong suốt có thể điều chỉnh
  - [x] Hỗ trợ dự phòng khi VTK không khả dụng

- [x] Kết nối tự động giữa dữ liệu liều, cấu trúc giải phẫu và hiển thị 3D:
  - [x] Tự động cập nhật hiển thị khi dữ liệu liều hoặc cấu trúc thay đổi
  - [x] Đồng bộ hóa giữa hiển thị liều 3D và DoseVolumeHistogram
  - [x] Kết nối với cấu trúc giải phẫu từ StructureManager
  - [x] Tự động chuyển đổi giữa các định dạng dữ liệu liều khác nhau

- [x] Tối ưu hóa hiển thị isodose surface với các mức liều có thể tùy chỉnh:
  - [x] Hiển thị các bề mặt isodose với màu sắc tương tự Eclipse
  - [x] Cho phép người dùng tùy chỉnh mức liều và độ trong suốt
  - [x] Tối ưu hóa hiệu suất hiển thị với kỹ thuật mesh decimation
  - [x] Cải thiện chất lượng hình ảnh với smooth shading

### Cải thiện giao diện External Beam Planning tab phong cách Eclipse

- [x] Thiết kế lại đầy đủ tab theo phong cách Eclipse với layout tối ưu:
  - [x] Panel trái chứa các tùy chọn lập kế hoạch, chùm tia và mục tiêu
  - [x] Panel phải hiển thị kết quả: liều 3D, DVH và bảng thông tin
  - [x] Splitter có thể điều chỉnh để thay đổi kích thước panel
  - [x] Bố cục thống nhất và chuyên nghiệp

- [x] Chuyển đổi từ combo box chọn chế độ sang các radio button trực quan hơn:
  - [x] Radio button cho các chế độ Forward Planning, Inverse Planning, và MCO
  - [x] Hiển thị các tùy chọn phù hợp với từng chế độ
  - [x] Điều chỉnh giao diện tự động khi chuyển đổi chế độ
  - [x] Hỗ trợ chuyển đổi linh hoạt giữa các chế độ lập kế hoạch

- [x] Tách riêng thành phần quản lý chùm tia và mục tiêu tối ưu hóa vào các tab con:
  - [x] Tab beams quản lý chùm tia, hướng, MLC, và thông số vật lý
  - [x] Tab objectives quản lý các mục tiêu tối ưu hóa và ràng buộc
  - [x] Tab settings chứa các thiết lập tối ưu hóa và tính toán liều
  - [x] Tab rồng cho MCO khi chọn chế độ tối ưu hóa đa tiêu chí

- [x] Thêm trạng thái và thông báo tiến trình rõ ràng:
  - [x] Status bar hiển thị thông tin trạng thái hiện tại
  - [x] Progress bar cho các tác vụ dài như tính toán liều
  - [x] Thông báo lỗi và cảnh báo với màu sắc trực quan
  - [x] Logging chi tiết để theo dõi quá trình lập kế hoạch

### Tăng cường kết nối dữ liệu giữa các component

- [x] Đồng bộ hóa giữa hiển thị liều 3D và DVH:
  - [x] Tự động cập nhật cả hai khi phân phối liều thay đổi
  - [x] Hiển thị cùng màu sắc và cấu trúc trên cả hai widget
  - [x] Liên kết chọn cấu trúc giữa panel cấu trúc và DVH
  - [x] Đồng bộ hóa isodose levels giữa hiển thị 2D và 3D

- [x] Đồng bộ hóa giữa hiển thị cấu trúc trong 3D và panel mục tiêu tối ưu hóa:
  - [x] Hiển thị cùng màu sắc cho cấu trúc trong tất cả các panel
  - [x] Cập nhật trạng thái hiển thị khi thay đổi visibility
  - [x] Tự động làm nổi bật cấu trúc đang được chọn
  - [x] Hiển thị các mục tiêu tối ưu hóa cho cấu trúc đang chọn

- [x] Đảm bảo cập nhật dữ liệu khi thay đổi chế độ lập kế hoạch:
  - [x] Lưu trạng thái hiển thị khi chuyển đổi giữa các chế độ
  - [x] Cập nhật UI phù hợp với chế độ đang chọn
  - [x] Chuyển đổi giữa các thuật toán tối ưu hóa phù hợp
  - [x] Đồng bộ hóa dữ liệu giữa các thành phần

- [x] Thêm hiển thị/ẩn tab MCO Navigator tự động:
  - [x] Hiển thị MCO Navigator khi chọn chế độ MCO
  - [x] Ẩn khi chuyển về chế độ Forward hoặc Inverse
  - [x] Lưu trạng thái MCO Navigator khi ẩn/hiện
  - [x] Tự động chuyển đổi layout phù hợp với chế độ

### Nâng cao độ tin cậy và khả năng phục hồi

- [x] Tích hợp xử lý lỗi toàn diện:
  - [x] Try-except cho các thao tác với thư viện bên ngoài
  - [x] Logging chi tiết cho quá trình debug
  - [x] Thông báo lỗi thân thiện với người dùng
  - [x] Khôi phục trạng thái trước khi xảy ra lỗi

- [x] Thêm các placeholder có thể hiển thị khi thành phần gặp lỗi:
  - [x] Widget thay thế khi VTK không khả dụng
  - [x] Hiển thị thông báo lỗi và hướng dẫn khắc phục
  - [x] Chức năng giới hạn vẫn hoạt động khi thiếu thành phần
  - [x] Khả năng tải lại thành phần khi có sẵn

- [x] Sử dụng chế độ dự phòng:
  - [x] Dự phòng khi VTK không khả dụng
  - [x] Dự phòng khi PyQt5 không khả dụng
  - [x] Dự phòng khi không có GPU cho tính toán nhanh
  - [x] Các thuật toán thay thế khi thuật toán chính không khả dụng

- [x] Cải thiện logging:
  - [x] Ghi log chi tiết về quá trình khởi tạo và sử dụng thành phần
  - [x] Thông tin môi trường hệ thống và thư viện
  - [x] Stack trace chi tiết khi xảy ra lỗi
  - [x] Tùy chọn gửi log lỗi để cải thiện phần mềm

## Cải tiến mới nhất (v0.8.6)

### Hoàn thiện module phân tích độ bền vững (quangtps/evaluation/robustness)

- [x] Cập nhật constructor của lớp RobustnessAnalyzer:
  - [x] Thêm tham số setup_uncertainty (mm)
  - [x] Thêm tham số range_uncertainty (%)
  - [x] Thêm tham số num_scenarios
  - [x] Thêm tùy chọn dose_grid

- [x] Bổ sung các phương thức cốt lõi cho lớp RobustnessResult:
  - [x] get_structure_dvhs()
  - [x] get_target_coverage_data()
  - [x] get_evaluation_metrics()
  - [x] get_spatial_analysis_data()

- [x] Thêm các tính năng xuất kết quả phân tích:
  - [x] export_to_csv()
  - [x] export_to_excel()
  - [x] create_pdf_report()
  - [x] create_html_report()

- [x] Cải thiện phân tích dữ liệu:
  - [x] Tính toán thống kê D95, Dmax và biến thiên
  - [x] Đánh giá chỉ số biến thiên và độ phù hợp
  - [x] Phân tích không gian với bản đồ nhiệt

### Tạo tính năng phân tích độ bền vững theo phong cách Eclipse

- [x] Chuẩn hóa giao diện phân tích độ bền vững:
  - [x] Thiết kế theo phong cách Eclipse hiện đại
  - [x] Tab phân tích độ bền vững đầy đủ chức năng
  - [x] Widget độc lập có thể tái sử dụng
  - [x] Biểu đồ DVH bands tương tác cao

- [x] Tích hợp giữa UI và module phân tích cốt lõi:
  - [x] Kết nối các thành phần (tab, widget, analyzer)
  - [x] Cập nhật tự động khi thay đổi tham số
  - [x] Hiển thị tiến trình phân tích
  - [x] Hiển thị kết quả trực quan

- [x] Kết nối các thành phần:
  - [x] robust_analysis_tab.py
  - [x] robust_analysis_widget.py
  - [x] robustness_visualization.py
  - [x] robustness_analyzer.py

## Kế hoạch cải tiến (v0.9.0)

### Đã lên kế hoạch: Hoàn thiện hệ thống tối ưu hóa đa tiêu chí (MCO)

- [ ] Cải thiện MCO Navigator theo phong cách Eclipse:
  - [ ] Giao diện nâng cao cho Pareto Navigator
  - [ ] Tương tác trực quan với bề mặt Pareto
  - [ ] Hiển thị trực quan giải pháp tối ưu với nhiều màu sắc
  - [ ] Slide bar để di chuyển trên bề mặt Pareto

- [ ] Phát triển thuật toán tối ưu MCO hiệu suất cao:
  - [ ] Tạo các giải pháp Pareto nhanh hơn với các thuật toán tiên tiến
  - [ ] Hỗ trợ tính toán song song trên GPU khi có thể
  - [ ] Tăng cường chất lượng giải pháp Pareto
  - [ ] Giảm thời gian tạo các giải pháp

- [ ] Cải thiện giao diện người dùng MCO:
  - [ ] Hiển thị trực quan mối quan hệ giữa các mục tiêu
  - [ ] Nhiều chế độ hiển thị đồ thị (2D, 3D, Matrix)
  - [ ] Tùy chọn đánh trọng số các mục tiêu
  - [ ] Lịch sử khám phá giải pháp

### Đã lên kế hoạch: Cải thiện hiệu suất tính toán liều

- [ ] Tối ưu hóa thuật toán tính liều cho kích thước lớn:
  - [ ] Cải thiện hiệu suất thuật toán Collapsed Cone
  - [ ] Tối ưu hóa Monte Carlo CPU cho hiệu suất tốt hơn
  - [ ] Bổ sung Monte Carlo GPU với hỗ trợ nhiều nền tảng
  - [ ] Giảm yêu cầu bộ nhớ cho tính toán liều quy mô lớn

- [ ] Tích hợp tính toán phân tán:
  - [ ] Hỗ trợ tính toán trên nhiều CPU/GPU
  - [ ] Thêm tính năng tính toán qua mạng
  - [ ] Cải thiện quản lý tài nguyên hệ thống
  - [ ] Tối ưu hóa phân phối công việc

### Đã lên kế hoạch: Phát triển mô hình đánh giá sinh học

- [ ] Tích hợp mô hình sinh học nâng cao:
  - [ ] LQ, gEUD, TCP, NTCP
  - [ ] Mô hình Lyman-Kutcher-Burman
  - [ ] Mô hình Niemierko
  - [ ] Mô hình phản ứng mô với phân đoạn liều

- [ ] Đánh giá tổn thương mô:
  - [ ] Dự đoán tác dụng phụ
  - [ ] Phân tích rủi ro dựa trên các mô hình lâm sàng
  - [ ] Hiển thị bản đồ nhiệt tổn thương
  - [ ] Báo cáo rủi ro lâm sàng

### Đã lên kế hoạch: Nâng cao trực quan hóa dữ liệu lâm sàng

- [ ] Cải thiện hiển thị DVH:
  - [ ] Nhiều chế độ hiển thị DVH (tích lũy, vi phân, EUD)
  - [ ] Tùy chỉnh màu sắc và kiểu hiển thị
  - [ ] Xuất dữ liệu DVH nhiều định dạng
  - [ ] Tính năng phân tích nâng cao trực tiếp trên biểu đồ

- [ ] Nâng cấp hiển thị 3D:
  - [ ] Hiển thị đa thể thức (multimodality)
  - [ ] Nhiều chế độ kết xuất 3D nâng cao
  - [ ] Tương tác trực tiếp với mô 3D
  - [ ] Hiển thị dữ liệu 4D (thời gian)