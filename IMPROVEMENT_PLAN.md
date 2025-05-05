# Kế hoạch cải tiến QuangTPS

Tài liệu này mô tả kế hoạch cải tiến cho dự án QuangTPS - hệ thống lập kế hoạch xạ trị mã nguồn mở.

## Mục tiêu

Mục tiêu chính là xây dựng một hệ thống lập kế hoạch xạ trị mã nguồn mở có tính năng tương đương với các hệ thống thương mại như Eclipse của Varian.

## Kế hoạch cải tiến ngắn hạn (3 tháng tới)

### 1. Cải thiện thuật toán Monte Carlo cho tính toán liều
- [ ] Tích hợp GPUMonteCarlo để tăng tốc tính toán
- [ ] Cải thiện mô phỏng vật lý tương tác tia với mô
- [ ] Thêm cơ chế báo cáo lỗi thống kê và độ chính xác

### 2. Nâng cao kỹ thuật tối ưu hóa kế hoạch điều trị
- [ ] Triển khai tối ưu hóa đa mục tiêu (MCO - Multicriteria Optimization)
- [ ] Tích hợp VMAT (Volumetric Modulated Arc Therapy)
- [ ] Cải thiện thuật toán xác định góc chùm tia tối ưu

### 3. Phát triển mô-đun Auto Planning
- [ ] Phát triển Knowledge-Based Planning (KBP)
- [ ] Triển khai Auto-Contouring sử dụng các mô hình học sâu
- [ ] Tích hợp quy trình làm việc tự động

### 4. Cải thiện đánh giá kế hoạch điều trị và báo cáo
- [ ] Thêm tính toán TCP/NTCP (Tumor Control Probability/Normal Tissue Complication Probability)
- [ ] Cải thiện so sánh kế hoạch điều trị đồ họa
- [ ] Thêm báo cáo QA và kiểm tra độ chính xác

## Kế hoạch cải tiến trung hạn (6-12 tháng)

### 1. Phát triển trí tuệ nhân tạo và học máy
- [ ] Triển khai dự đoán chất lượng kế hoạch dựa trên AI
- [ ] Tích hợp auto-segmentation nâng cao sử dụng các mô hình học sâu
- [ ] Phát triển hệ khuyến nghị kế hoạch điều trị

### 2. Cải thiện độ chính xác tính toán liều
- [ ] Triển khai thuật toán Acuros XB hoàn chỉnh
- [ ] Tích hợp mô phỏng đầy đủ tán xạ Compton và tạo cặp
- [ ] Phát triển bảng hiệu chỉnh nhân thực nghiệm

### 3. Tích hợp và hỗ trợ các kỹ thuật điều trị đặc biệt
- [ ] Hỗ trợ điều trị khối u chuyển động (4D)
- [ ] Tích hợp điều trị xạ trị định vị thân (SBRT)
- [ ] Phát triển công cụ đánh giá và kiểm tra xạ phẫu

### 4. Cải thiện giao diện người dùng và UX
- [ ] Thiết kế lại giao diện chính để tiếp cận kiểu Eclipse
- [ ] Phát triển các dashboard tương tác
- [ ] Tích hợp các chế độ xem 3D nâng cao

## Kế hoạch cải tiến dài hạn (1-2 năm)

### 1. Phát triển hệ thống phân tán
- [ ] Triển khai tính toán đám mây cho các thuật toán tính liều phức tạp
- [ ] Tích hợp hỗ trợ tính toán GPU phân tán
- [ ] Phát triển hệ thống lưu trữ và truy xuất dữ liệu phân tán

### 2. Tích hợp các công nghệ mới
- [ ] Hỗ trợ các kỹ thuật điều trị FLASH
- [ ] Tích hợp hệ thống trên thiết bị di động và tablet
- [ ] Phát triển giao diện thực tế tăng cường (AR)

### 3. Phát triển hệ sinh thái mở rộng
- [ ] Phát triển hệ thống plugin
- [ ] Tạo cộng đồng phát triển mã nguồn mở
- [ ] Thiết lập quy trình đóng góp và tiêu chuẩn mã

## Các cải tiến đã hoàn thành

### Phiên bản 0.8.0
- [x] Thêm module io_utils trong thư mục utils
- [x] Cải tiến thuật toán tính liều AAA với vector hóa tính toán TERMA
- [x] Thêm phương pháp mới tính PDD theo năng lượng cho AAA
- [x] Tối ưu hóa tích chập bằng FFT trong thuật toán AAA

### Phiên bản 0.7.5
- [x] Thêm module margin cho segmentation với OpenCV
- [x] Thêm chức năng tạo margin không đồng đều và vòng
- [x] Thêm chức năng tạo margin bề mặt
- [x] Thêm ClinicalMetricsCalculator để tính các chỉ số lâm sàng (HI, CI, GI, Coverage)
- [x] Thêm báo cáo đánh giá kế hoạch với hiển thị trực quan các chỉ số
- [x] Sửa lỗi DVHWidget khi thiếu QDialog

## Ưu tiên hiện tại

1. **Cải thiện thuật toán Monte Carlo** - Mục tiêu là tăng hiệu suất và độ chính xác của tính toán liều
2. **Tối ưu hóa kế hoạch VMAT** - Cho phép lập kế hoạch VMAT hoàn chỉnh
3. **Phát triển Auto Planning** - Tăng hiệu quả trong quy trình làm việc
4. **Tích hợp mô hình học máy** - Đặc biệt là cho auto-segmentation

## Hướng dẫn đóng góp

Nếu bạn muốn đóng góp vào dự án QuangTPS, vui lòng làm theo các bước sau:

1. Fork repository
2. Tạo một nhánh (branch) mới cho tính năng hoặc cải tiến của bạn
3. Commit mã của bạn
4. Push lên nhánh của bạn
5. Tạo Pull Request