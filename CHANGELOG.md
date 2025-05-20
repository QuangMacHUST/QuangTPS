# Changelog

Tất cả những thay đổi đáng chú ý của dự án QuangTPS sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án này tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.11] - 2023-09-15

### Thêm mới
- Nâng cấp MPR viewer trong Structure tab với hiển thị dữ liệu hình ảnh thực:
  - Tích hợp đầy đủ MPRViewer với hiển thị đa mặt phẳng (axial, sagittal, coronal)
  - Thanh công cụ vẽ contour với các tùy chọn: vẽ, xóa, cọ và smart brush
  - Hỗ trợ hoàn tác/làm lại các thao tác vẽ contour
  - Hiển thị và tương tác với cấu trúc giải phẫu trên các mặt phẳng MPR

### Cải tiến
- Nâng cao trải nghiệm người dùng khi phân đoạn cấu trúc:
  - Xử lý sự kiện chuột để vẽ contour trực tiếp trên MPR viewer
  - Đồng bộ hóa hiển thị giữa danh sách cấu trúc và MPR viewer
  - Cập nhật tự động hiển thị khi chỉnh sửa hoặc thay đổi lựa chọn cấu trúc
  - Hiển thị trực quan trạng thái "đã chọn" cho cấu trúc trong MPR viewer

### Sửa lỗi
- Khắc phục lỗi không hiển thị cấu trúc trong MPR viewer khi thay đổi lựa chọn
- Xử lý các trường hợp ngoại lệ khi dữ liệu hình ảnh không hợp lệ
- Cải thiện xử lý ngoại lệ khi các module phụ thuộc không khả dụng
- Đảm bảo làm sạch tài nguyên khi đóng Structure tab

## [0.8.10] - 2023-09-10

### Thêm mới
- Hoàn thiện tính năng phân đoạn tự động trong Structure tab:
  - Tích hợp đầy đủ AutoSegmentationEngine vào phương thức `auto_segment()`
  - Giao diện người dùng cho lựa chọn và thiết lập tham số phân đoạn
  - Hiển thị tiến trình thực hiện với ProgressDialog và ProgressBar
  - Tích hợp kết quả phân đoạn vào quy trình lập kế hoạch

### Cải tiến
- Tăng cường tính ổn định cho module MCO:
  - Khắc phục lỗi trong hàm plot_pareto_front_3d của mco_module.py
  - Thêm xử lý ngoại lệ cho colormap khi matplotlib.cm.viridis không khả dụng
  - Giảm thiểu rủi ro lỗi runtime khi hiển thị bề mặt Pareto 3D
  - Đảm bảo khả năng sử dụng trên nhiều môi trường khác nhau

### Sửa lỗi
- Sửa lỗi trong mco_module.py khi biến cmap và norm được sử dụng trước khi khởi tạo
- Cải thiện xử lý ngoại lệ khi thiếu thư viện PyQt5 hoặc VTK
- Khắc phục vấn đề cập nhật giao diện người dùng sau quá trình phân đoạn tự động

## [0.8.9] - 2023-09-05

### Thêm mới
- Tạo module báo cáo đánh giá kế hoạch trong `quangtps/reporting/plan_report_generator.py`:
  - Hỗ trợ tạo báo cáo đánh giá kế hoạch xạ trị với nhiều định dạng xuất (PDF, HTML, CSV)
  - Báo cáo chứa thông tin kế hoạch, DVH, kết quả đánh giá mục tiêu và chỉ số nâng cao
  - Hỗ trợ so sánh nhiều kế hoạch trong cùng báo cáo

- Định nghĩa chuẩn cho các loại mục tiêu lâm sàng trong `quangtps/evaluation/protocols/goal_type.py`:
  - Các loại mục tiêu cơ bản như DOSE_VOLUME, VOLUME_DOSE
  - Chỉ số thống kê liều như MEAN_DOSE, MAX_DOSE, MIN_DOSE
  - Chỉ số đánh giá nâng cao như HOMOGENEITY_INDEX, CONFORMITY_INDEX, GRADIENT_INDEX
  - Chỉ số đánh giá sinh học như TCP, NTCP, EUD

- Tạo hệ thống điểm đánh giá kế hoạch trong `quangtps/evaluation/plan_quality/plan_quality_score.py`:
  - Các mức đánh giá EXCELLENT, GOOD, ACCEPTABLE, MARGINAL, POOR
  - Phương thức chuyển đổi từ phần trăm đạt mục tiêu sang mức đánh giá
  - Ánh xạ màu sắc trực quan cho từng mức đánh giá

### Cải tiến
- Phát triển bộ đánh giá kế hoạch xạ trị toàn diện:
  - Lớp PlanQualityEvaluator đánh giá kế hoạch theo protocol lâm sàng
  - Tính toán các chỉ số đánh giá cho mục tiêu và cơ quan nguy cấp riêng biệt
  - Hỗ trợ xuất kết quả đánh giá sang nhiều định dạng để phân tích
  - Tính toán chỉ số đồng nhất và phù hợp cho đánh giá nâng cao

- Tăng cường xử lý ngoại lệ và tính ổn định:
  - Cơ chế dự phòng khi các module không khả dụng
  - Xử lý lỗi import thông minh với cơ chế fallback
  - Báo cáo lỗi chi tiết để dễ dàng gỡ lỗi
  - Tăng cường khả năng phục hồi khi thiếu các thư viện bên ngoài

### Sửa lỗi
- Khắc phục các lỗi trong module đánh giá kế hoạch:
  - Sửa lỗi tham chiếu đến các thành viên enum không tồn tại
  - Sửa lỗi import module báo cáo kế hoạch
  - Cải thiện xử lý khi không có DVHAnalyzer

## [0.8.8] - 2023-08-30

### Thêm mới
- Tạo module đánh giá kế hoạch xạ trị toàn diện trong `quangtps/ui/plan_evaluation_report_tab.py`:
  - Giao diện phong cách Eclipse cho đánh giá kế hoạch xạ trị chuyên nghiệp
  - Báo cáo đánh giá kế hoạch với khả năng xuất PDF, HTML và CSV
  - Hiển thị tích hợp của DVH, đánh giá mục tiêu lâm sàng và thống kê liều
  - Tính năng so sánh kế hoạch để phân tích nhiều kế hoạch cùng lúc

### Cải tiến
- Nâng cao trải nghiệm đánh giá kế hoạch:
  - Hiển thị điểm đánh giá tổng thể, PTV và OAR với mã màu trực quan
  - Quản lý và chỉnh sửa protocol lâm sàng ngay trong giao diện
  - Đánh giá kế hoạch tự động với kết quả màu sắc trực quan
  - Tích hợp chọn cấu trúc từ bảng mục tiêu với hiển thị DVH
  - Bố cục thông minh với splitter có thể tùy chỉnh giữa các panel

- Tích hợp với các module hiện có:
  - Kết nối liền mạch với DVHWidget, ProtocolManager và Clinical Goals
  - Tận dụng chức năng của PlanQualityWidget với giao diện mới
  - Đảm bảo tương thích với cả hệ thống Eclipse-style theme
  - Xử lý ngoại lệ toàn diện khi các thành phần không khả dụng

## [0.8.7] - 2023-08-25

### Thêm mới
- Tích hợp hoàn chỉnh hiển thị liều 3D vào External Beam Planning tab:
  - Kết nối tự động giữa dữ liệu liều, cấu trúc giải phẫu và hiển thị 3D
  - Cơ chế tự động chuyển đổi dữ liệu liều từ nhiều định dạng khác nhau
  - Hiển thị đồng bộ liều 3D và cấu trúc giải phẫu trong một khung nhìn thống nhất

### Cải tiến
- Thiết kế lại giao diện External Beam Planning theo phong cách Eclipse:
  - Layout tối ưu với splitter có thể điều chỉnh giữa các panel
  - Chuyển đổi từ combo box sang radio button cho chọn chế độ lập kế hoạch
  - Tổ chức các tính năng thành các tab con logic và trực quan hơn
  - Thêm status bar cung cấp thông tin trạng thái và tiến trình

- Cải thiện tích hợp hiển thị liều 3D:
  - Thay thế placeholder bằng DoseVisualization3D đầy đủ chức năng
  - Tối ưu hóa hiển thị isodose surface với các mức liều có thể tùy chỉnh
  - Hiển thị cấu trúc 3D với độ trong suốt và màu sắc có thể tùy chỉnh
  - Đồng bộ hóa hiển thị khi dữ liệu liều hoặc cấu trúc thay đổi

- Nâng cao độ tin cậy của hệ thống:
  - Xử lý lỗi toàn diện khi các thư viện bên ngoài không khả dụng
  - Placeholder thông minh cho các thành phần gặp lỗi
  - Chế độ dự phòng cho các tính năng khi thiếu VTK hoặc PyQt5
  - Logging chi tiết để hỗ trợ gỡ lỗi và theo dõi

## [0.8.6] - 2023-08-20

### Thêm mới
- Hoàn thiện module phân tích độ bền vững trong `quangtps/evaluation/robustness`:
  - Cập nhật constructor của `RobustnessAnalyzer` với các tham số đầy đủ:
    - `setup_uncertainty` - độ không chắc chắn vị trí bệnh nhân (mm)
    - `range_uncertainty` - độ không chắc chắn phạm vi (%)
    - `num_scenarios` - số lượng kịch bản phân tích
  - Bổ sung các phương thức cốt lõi cho `RobustnessResult`:
    - `get_structure_dvhs()` - lấy dữ liệu DVH cho từng cấu trúc
    - `get_target_coverage_data()` - lấy dữ liệu độ phủ mục tiêu
    - `get_evaluation_metrics()` - lấy chỉ số đánh giá toàn diện
    - `get_spatial_analysis_data()` - lấy dữ liệu phân tích không gian

### Cải tiến
- Thêm các tính năng xuất kết quả phân tích độ bền vững:
  - Xuất báo cáo định dạng CSV và Excel với các sheet phân tích riêng biệt
  - Tạo báo cáo PDF và HTML tương tác với biểu đồ và bảng phân tích
  - Tạo DVH bands trực quan với độ dao động giữa các kịch bản khác nhau
- Cải thiện phân tích dữ liệu cho mục tiêu và OAR:
  - Tính toán các thông số D95, Dmax và biến thiên theo kịch bản
  - Đánh giá tự động các chỉ số biến thiên và độ phù hợp lâm sàng
  - Phân tích không gian với bản đồ nhiệt sự khác biệt liều
- Tích hợp phong cách Eclipse vào module phân tích độ bền vững:
  - Thiết kế giao diện theo phong cách Eclipse hiện đại
  - Đảm bảo tương thích giữa UI và module phân tích cốt lõi
  - Kết nối các thành phần: `robust_analysis_tab.py`, `robust_analysis_widget.py` và `robustness_visualization.py`

## [0.8.5] - 2023-08-15

### Cải tiến
- Sửa lỗi indent trong module phân tích gamma (quangtps/evaluation/metrics/gamma_analysis.py)
- Tạo module phân tích độ bền vững (Robustness Analysis) phong cách Eclipse
- Cải thiện trải nghiệm phân tích độ bền vững
- Bổ sung tính năng tương thích

[0.8.11]: https://github.com/username/QuangTPS/compare/v0.8.10...v0.8.11
[0.8.10]: https://github.com/username/QuangTPS/compare/v0.8.9...v0.8.10
[0.8.9]: https://github.com/username/QuangTPS/compare/v0.8.8...v0.8.9
[0.8.8]: https://github.com/username/QuangTPS/compare/v0.8.7...v0.8.8
[0.8.7]: https://github.com/username/QuangTPS/compare/v0.8.6...v0.8.7
[0.8.6]: https://github.com/username/QuangTPS/compare/v0.8.5...v0.8.6
[0.8.5]: https://github.com/username/QuangTPS/compare/v0.8.4...v0.8.5