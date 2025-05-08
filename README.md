# QuangTPS - Hệ thống Lập kế hoạch Xạ trị

QuangTPS là một hệ thống lập kế hoạch xạ trị toàn diện được phát triển cho mục đích nghiên cứu và giáo dục. Hệ thống cung cấp đầy đủ công cụ cho quy trình xạ trị, bao gồm hiển thị hình ảnh, vẽ cấu trúc, thiết lập chùm tia, tính toán liều, và đánh giá kế hoạch.

## Phiên bản hiện tại
**[v0.7.26]** - Đã phát triển module dự đoán thay đổi giải phẫu với biến dạng nâng cao và module tối ưu hóa đa tiêu chí (MCO) với bề mặt Pareto.

## Tính năng chính

- **Giao diện tương tự Eclipse** - Giao diện người dùng hiện đại tương tự Varian Eclipse TPS với quy trình trực quan và công cụ toàn diện
- **Tái tạo hình ảnh đa mặt phẳng (MPR)** - Xem và điều hướng hình ảnh y tế trên các mặt phẳng axial, sagittal và coronal
- **Hiển thị 3D** - Trực quan hóa giải phẫu, cấu trúc và phân bố liều trong không gian 3D sử dụng VTK
- **Công cụ vẽ cấu trúc** - Vẽ và chỉnh sửa cấu trúc với các công cụ vẽ nâng cao (brush, pencil, polygon, threshold)
- **Lập kế hoạch chùm tia ngoài** - Tạo và quản lý kế hoạch xạ trị chùm tia ngoài với đầy đủ tham số chùm tia
- **Tính toán liều nâng cao** - Tính toán phân bố liều sử dụng AAA, Acuros XB, Monte Carlo và các thuật toán đơn giản hóa
- **Tối ưu hóa VMAT** - Tối ưu hóa kế hoạch VMAT với chất lượng tương đương các hệ thống thương mại
- **Trình biên tập MLC** - Trình biên tập MLC nâng cao với chế độ Beam's Eye View để tạo hình trường chùm tia chính xác
- **Giao thức lâm sàng** - Định nghĩa, nhập và quản lý giao thức lâm sàng để đánh giá kế hoạch
- **Đánh giá chất lượng kế hoạch** - Đánh giá tự động chất lượng kế hoạch theo giao thức và mục tiêu lâm sàng
- **Các chỉ số đánh giá toàn diện** - Tính toán CI, HI, GI, TCP, NTCP và các chỉ số chất lượng khác
- **Tối ưu hóa đa tiêu chí (MCO)** - Tìm kiếm kế hoạch tối ưu với bề mặt Pareto và bộ điều hướng tương tác
- **Lập kế hoạch thích ứng** - Hỗ trợ dự đoán thay đổi giải phẫu và tối ưu hóa kế hoạch thích ứng
- **Hỗ trợ DICOM** - Nhập và xuất hình ảnh DICOM, cấu trúc, kế hoạch và liều

## Cài đặt

### Yêu cầu

- Python 3.8 trở lên
- Git

### Các bước cài đặt

1. Clone repository:
   ```
   git clone https://github.com/yourusername/QuangTPS.git
   cd QuangTPS
   ```

2. Cài đặt thư viện phụ thuộc:
   ```
   python scripts/install_all_dependencies.py
   ```

3. Xác nhận cài đặt:
   ```
   python scripts/run_quangtps.py
   ```

### Thư viện phụ thuộc

QuangTPS yêu cầu các thư viện chính sau:

- **NumPy** - Cho tính toán số học
- **PyQt5** - Cho giao diện đồ họa
- **VTK** - Cho hiển thị 3D
- **SimpleITK** - Cho xử lý hình ảnh
- **pydicom** - Cho xử lý file DICOM
- **matplotlib** - Cho vẽ biểu đồ
- **scikit-image** - Cho xử lý hình ảnh

## Cách sử dụng

### Chạy ứng dụng

```
python scripts/run_quangtps.py [options]
```

Tùy chọn:
- `--debug` - Bật ghi log debug
- `--patient-dir PATH` - Mở thư mục bệnh nhân cụ thể khi khởi động

### Quy trình cơ bản

1. **Tab Bệnh nhân**: Chọn hoặc nhập dữ liệu bệnh nhân
2. **Tab Cấu trúc**: Tạo và chỉnh sửa cấu trúc (mục tiêu và cơ quan nguy cấp)
3. **Tab Lập kế hoạch chùm tia ngoài**: Tạo và cấu hình chùm tia điều trị và tính toán liều
4. **Tab Đánh giá**: Đánh giá kế hoạch điều trị với DVH, thống kê và tuân thủ giao thức

## Đóng góp

Đóng góp vào QuangTPS được khuyến khích! Vui lòng làm theo các bước sau:

1. Fork repository
2. Tạo nhánh tính năng mới (`git checkout -b feature/amazing-feature`)
3. Commit các thay đổi của bạn (`git commit -m 'Add some amazing feature'`)
4. Push lên nhánh của bạn (`git push origin feature/amazing-feature`)
5. Mở Pull Request

## Tuyên bố từ chối trách nhiệm

QuangTPS được thiết kế cho mục đích giáo dục và nghiên cứu. Hệ thống không được FDA phê duyệt hoặc đánh dấu CE, và không nên sử dụng cho lập kế hoạch điều trị lâm sàng mà không có xác nhận và phê duyệt thích hợp.