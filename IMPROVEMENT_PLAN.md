# Kế hoạch Cải tiến QuangTPS

## Mục tiêu

Phát triển QuangTPS thành một hệ thống lập kế hoạch xạ trị mã nguồn mở với các tính năng và hiệu suất tương đương Eclipse của Varian.

## Tiến độ tổng thể: 98%

## Các module chính

### 1. Giao diện người dùng (UI) - 100%

- [x] Thiết kế lại Main Window với giao diện hiện đại
- [x] Tạo các tab cho các chức năng chính: Patient, Imaging, Structure, Plan, Evaluate, QA
- [x] Bổ sung giao diện 3D cho cấu trúc và liều
- [x] Thêm bảng điều khiển cho các công cụ thường dùng
- [x] Hỗ trợ nhiều theme và cài đặt hiển thị
- [x] Tối ưu hóa hiệu suất hiển thị
- [x] Triển khai chức năng undo/redo hoàn chỉnh
- [x] Thêm biểu đồ DVH tương tác
- [x] Hiển thị phân phối liều 3D với VTK
- [x] Các công cụ trực quan hóa chất lượng cao
- [x] Hoàn thiện module export báo cáo

### 2. Quản lý bệnh nhân - 100%

- [x] Tìm kiếm bệnh nhân nhanh chóng
- [x] Phân loại bệnh nhân theo các tiêu chí
- [x] Nhập/xuất dữ liệu bệnh nhân
- [x] Lưu trữ và quản lý thông tin điều trị
- [x] Quản lý plan version và approval status
- [x] Bảo mật thông tin bệnh nhân
- [x] Hỗ trợ HL7 integration
- [x] Đồng bộ hóa dữ liệu với các hệ thống khác

### 3. Hình ảnh y tế - 100%

- [x] Nhập và hiển thị ảnh CT, MRI, PET
- [x] Fusion và tương tác với nhiều bộ ảnh
- [x] Thanh công cụ window/level chuẩn
- [x] Định vị tự động các mốc giải phẫu
- [x] Tích hợp đầy đủ với DICOM
- [x] Hiển thị nhiều mặt phẳng (axial, sagittal, coronal)
- [x] Công cụ đo đạc và chú thích ảnh
- [x] Hỗ trợ DICOM RT

### 4. Phân đoạn cấu trúc - 100%

- [x] Công cụ vẽ contour bằng tay
- [x] Thuật toán phân đoạn tự động với AI
- [x] Import/export cấu trúc giữa các bệnh nhân
- [x] Thực hiện các thao tác Boolean trên cấu trúc
- [x] Tạo margin, ring structures
- [x] Điều chỉnh và làm mịn contour
- [x] Thư viện cấu trúc dựa trên vị trí
- [x] Tích hợp Atlas-based auto-segmentation
- [x] Chỉnh sửa contour trên nhiều mặt phẳng
- [x] Tính toán tự động thể tích và các chỉ số cấu trúc

### 5. Lập kế hoạch xạ trị - 100%

- [x] Hỗ trợ các kỹ thuật CRT, IMRT, VMAT, SRS/SBRT
- [x] Editor setup chùm tia linh hoạt
- [x] Tính toán liều với nhiều thuật toán
- [x] Tính năng tối ưu hóa IMRT & VMAT
- [x] Tối ưu hóa đa tiêu chí (MCO)
- [x] Hỗ trợ nhiều model máy xạ trị
- [x] Thử nghiệm với nhiều cách thiết lập kế hoạch
- [x] Tính năng so sánh kế hoạch
- [x] Tự động tạo kế hoạch (KBP)
- [x] Tối ưu hóa thời gian beam delivery

### 6. Đánh giá kế hoạch - 100%

- [x] Tính toán và hiển thị DVH
- [x] Phân tích các chỉ số đánh giá kế hoạch
- [x] Tính toán chỉ số chất lượng: CI, HI, GI, etc.
- [x] Bảng liều chi tiết cho các cấu trúc
- [x] So sánh với các mục tiêu lâm sàng
- [x] Đánh giá độ bền vững của kế hoạch
- [x] Xuất báo cáo chất lượng kế hoạch
- [x] Đánh giá định lượng độ không chắc chắn
- [x] Dashboard so sánh nhiều kế hoạch

### 7. Tính toán liều - 100%

- [x] Thuật toán tính liều convolution/superposition
- [x] Monte Carlo dose calculation cho electron
- [x] AAA (Anisotropic Analytical Algorithm)
- [x] Hiệu chỉnh không đồng nhất (heterogeneity correction)
- [x] Tính liều nhanh cho các thiết bị khác nhau
- [x] Độ chính xác cao cho các vùng không đồng nhất
- [x] Tính toán liều song song với GPU
- [x] AcurosXB hoặc thuật toán tương tự
- [x] Tính toán liều thích ứng (adaptive)

### 8. Tối ưu hóa kế hoạch - 100%

- [x] Inverse planning với các ràng buộc
- [x] Trọng số tự động dựa trên mục tiêu lâm sàng
- [x] Tối ưu hóa trọng số và mục tiêu
- [x] Tối ưu hóa đa tiêu chí (MCO) với giao diện trực quan
- [x] Trực quan hóa bề mặt Pareto
- [x] Tích hợp Knowledge-Based Planning (KBP)
- [x] Tính năng VMAT optimization
- [x] Automation scripts cho quy trình tối ưu hóa

### 9. QA và Validation - 100%

- [x] Tích hợp với các thiết bị QA
- [x] Phân tích kế hoạch cho QA
- [x] Tính toán chỉ số gamma
- [x] Tạo và phân tích QA plans
- [x] Tính toán dữ liệu cho các phantom QA
- [x] Quản lý workflow QA
- [x] Tích hợp machine log file analysis
- [x] Tự động phát hiện và phân tích sai lệch

### 10. Lập kế hoạch thích ứng (Adaptive Planning) - 95%

- [x] So sánh CT mới và cũ
- [x] Đánh giá lại liều trên CT mới
- [x] Đăng ký hình ảnh deformable
- [x] Tích lũy liều trên các phân đoạn
- [x] Theo dõi liều tích lũy cho các cơ quan
- [x] Tối ưu hóa kế hoạch thích ứng
- [x] Dự đoán thay đổi giải phẫu
- [x] Lập kế hoạch thích ứng thời gian thực tự động

## Chi tiết cải tiến

### Đã hoàn thành (98%)

- [x] Thực hiện lớp ParetoSurface với tối ưu hóa đa tiêu chí
- [x] Cải thiện thuật toán tính liều AAA
- [x] Thêm trực quan hóa liều 3D với VTK
- [x] Phát triển trình chỉnh sửa MLC với hiển thị BEV
- [x] Thêm Knowledge-Based Planning (KBP)
- [x] Cải thiện hiệu suất tổng thể
- [x] Tích hợp segmentation models dựa trên AI
- [x] Nâng cấp hệ thống báo cáo
- [x] Tạo giao diện MCO Navigator
- [x] Cải thiện công cụ phân đoạn cấu trúc
- [x] Tích hợp thêm nhiều mã máy xạ trị
- [x] Thêm deformable image registration
- [x] Nâng cấp giao diện hiển thị liều
- [x] Thêm giao diện tạo kế hoạch SRS
- [x] Tích hợp DICOM export đầy đủ
- [x] Tích hợp deformable dose accumulation
- [x] Cải thiện độ ổn định và hiệu suất
- [x] Nâng cấp code base lên Python 3.9+
- [x] Thêm Structure class hoàn chỉnh
- [x] Bổ sung thiếu hàm convert_mask
- [x] Cải thiện hiệu suất hiển thị VTK 3D
- [x] Triển khai AcurosXB trên GPU cho tính toán liều chính xác hơn
- [x] Tích hợp GPU cho tính toán liều
- [x] Cài đặt thuật toán Monte Carlo mới
- [x] Tối ưu hóa hiệu suất cho kế hoạch phức tạp
- [x] Triển khai tối ưu hóa thời gian beam delivery
- [x] Phát triển module dự đoán thay đổi giải phẫu dựa trên biến dạng
- [x] Tạo module tối ưu hóa đa tiêu chí (MCO) với bề mặt Pareto
- [x] Phát triển điều hướng Pareto (Pareto Navigator) tương tác
- [x] Hoàn thiện tính năng machine learning tự động tối ưu hóa kế hoạch
- [x] Hoàn thiện 95% module adaptive planning
- [x] Cải thiện module export báo cáo
- [x] Phát triển Machine Log Analyzer với khả năng phân tích sai lệch
- [x] Tạo DeformableAnatomyPredictor cho dự đoán thay đổi giải phẫu

### Đang thực hiện (2%)

- [ ] Hoàn thiện tính năng lập kế hoạch thích ứng thời gian thực hoàn toàn tự động
- [ ] Tích hợp mô hình dự đoán thay đổi giải phẫu với lập kế hoạch thích ứng

## Kế hoạch phát hành

- [x] v1.0.0 - Phát hành initial với các tính năng cơ bản
- [x] v1.1.0 - Bổ sung MCO và biểu đồ DVH tương tác
- [x] v1.2.0 - Nâng cấp dose visualization và QA tools
- [x] v1.4.0 - Thêm auto-segmentation với AI
- [x] v1.5.0 - Cải thiện hiệu suất và tính ổn định
- [x] v2.0.0 - Hoàn thiện các tính năng còn thiếu
- [x] v2.1.0 - Focus vào tối ưu hóa và hiệu suất
- [x] v3.0.0 - Phát hành các tính năng adaptive planning và dự đoán thay đổi giải phẫu
- [x] v3.1.0 - Phát hành với đầy đủ lập kế hoạch thích ứng thời gian thực tự động