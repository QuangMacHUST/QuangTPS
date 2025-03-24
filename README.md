# QuangTPS - Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở

QuangTPS là một hệ thống lập kế hoạch xạ trị mã nguồn mở và độc lập, được thiết kế để cung cấp các công cụ lập kế hoạch xạ trị 3D hiện đại, dễ sử dụng và miễn phí cho cộng đồng y tế toàn cầu.

![QuangTPS Logo](./quangtps/ui/icons/logo.png)

## Giới thiệu

QuangTPS là một dự án phần mềm mã nguồn mở nhằm cung cấp một giải pháp toàn diện cho việc lập kế hoạch xạ trị 3D. Nó được thiết kế để làm việc với hình ảnh y tế DICOM, cho phép bác sĩ và nhà vật lý y học tạo, tối ưu hóa và đánh giá các kế hoạch điều trị xạ trị. Tạo ra để trở thành một giải pháp thay thế miễn phí và mạnh mẽ cho các hệ thống lập kế hoạch thương mại đắt tiền.

## Tính năng

### Tính năng hiện có:

- **Quản lý bệnh nhân**: Nhập và lưu trữ thông tin bệnh nhân, bao gồm dữ liệu nhân khẩu học và lịch sử y tế.
- **Nhập/xuất DICOM**: Nhập hình ảnh DICOM, cấu trúc, kế hoạch và xuất dữ liệu DICOM cần thiết.
- **Phân đoạn hình ảnh**: Tạo và chỉnh sửa cấu trúc roi trên CT, MRI và các dữ liệu hình ảnh khác.
- **Lập kế hoạch 3D**: Tạo kế hoạch xạ trị 3D với nhiều chùm tia và điểm kiểm soát.
- **Tính toán liều**: Tính toán phân bố liều bằng thuật toán Pencil Beam, Collapsed Cone và Monte Carlo.
- **Tối ưu hóa kế hoạch**: Công cụ tối ưu hóa kế hoạch dựa trên các chỉ tiêu và ràng buộc.
- **Đánh giá kế hoạch**: Đánh giá kế hoạch sử dụng DVH (Biểu đồ thể tích liều) và các chỉ số lâm sàng.
- **Hỗ trợ mô hình máy gia tốc**: Hỗ trợ mô hình dữ liệu chùm tia từ TrueBeam.

### Tính năng đang phát triển:

- **Xạ trị điều biến cường độ (IMRT)**: Nâng cao khả năng lập kế hoạch IMRT.
- **Xạ trị điều biến thể tích (VMAT)**: Hỗ trợ xạ trị cung xoay với điều biến cường độ.
- **Lập kế hoạch thích ứng**: Cập nhật kế hoạch dựa trên hình ảnh mới trong quá trình điều trị.
- **Xạ phẫu**: Công cụ đặc biệt cho kế hoạch xạ phẫu.
- **Liên kết hình ảnh đa phương thức**: Hỗ trợ đăng ký và chồng chéo CT, MRI, PET.
- **Trí tuệ nhân tạo**: Hỗ trợ tự động phân đoạn và đề xuất kế hoạch.

## Yêu cầu hệ thống

- **Hệ điều hành**: Windows 10/11, Linux, macOS
- **RAM**: 8GB trở lên (khuyến nghị 16GB)
- **CPU**: Bộ xử lý đa nhân
- **GPU**: Hỗ trợ OpenGL 3.3 hoặc cao hơn (khuyến nghị CUDA cho tính toán Monte Carlo)
- **Dung lượng đĩa**: Tối thiểu 5GB để cài đặt, khuyến nghị 20GB cho dữ liệu bệnh nhân
- **Độ phân giải màn hình**: Tối thiểu 1920x1080

## Cài đặt

### Cài đặt từ mã nguồn

```bash
# Tải mã nguồn
git clone https://github.com/username/quangtps.git
cd quangtps

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# Chạy ứng dụng
python scripts/run_quangtps.py
```

### Yêu cầu phần mềm

QuangTPS yêu cầu các thư viện Python sau:
- NumPy
- SciPy
- pandas
- PyQt5
- pydicom
- matplotlib
- scikit-image
- dicompyler-core
- cairosvg (cho biểu tượng và đồ họa)

## Cách sử dụng

### Khởi động ứng dụng

```bash
python scripts/run_quangtps.py
```

Các tùy chọn dòng lệnh:
- `--verbose` hoặc `-v`: Hiển thị thông tin chi tiết (debug)
- `--no-splash`: Không hiển thị màn hình chào
- `--console` hoặc `-c`: Chạy ở chế độ console (không giao diện đồ họa)
- `--demo` hoặc `-d`: Chạy với dữ liệu mẫu

### Nhập dữ liệu bệnh nhân

1. Chọn "File > New Patient" hoặc nhấn Ctrl+N để tạo một hồ sơ bệnh nhân mới.
2. Để nhập dữ liệu DICOM, chọn "File > Import DICOM" hoặc nhấn Ctrl+I.
3. Chọn thư mục chứa dữ liệu DICOM và nhấp vào "Import".

### Phân đoạn cấu trúc

1. Mở tab "Structures" trong không gian làm việc.
2. Sử dụng các công cụ vẽ để tạo hoặc chỉnh sửa cấu trúc.
3. Lưu cấu trúc bằng cách nhấp vào "Save Structure Set".

### Tạo kế hoạch

1. Chọn "Planning > New Plan" để tạo một kế hoạch mới.
2. Thêm chùm tia và thiết lập các tham số cần thiết.
3. Tính toán liều bằng cách nhấp vào "Calculate Dose".
4. Đánh giá kế hoạch sử dụng DVH và các phân tích khác.

### Lưu và xuất

1. Lưu kế hoạch bằng cách chọn "File > Save Plan".
2. Xuất dữ liệu DICOM bằng cách chọn "File > Export DICOM".
3. Tạo báo cáo bằng cách chọn "Report > Generate Report".

## Cấu trúc dự án

```
quangtps/
├── adaptive/           # Mô-đun xạ trị thích nghi
├── api/                # API cho tích hợp với các hệ thống khác
├── common/             # Tiện ích chung
├── core/               # Các thành phần cốt lõi
├── database/           # Quản lý cơ sở dữ liệu
├── dicom/              # Xử lý dữ liệu DICOM
├── dose/               # Tính toán và phân tích liều
├── evaluation/         # Đánh giá kế hoạch
├── imaging/            # Xử lý hình ảnh y khoa
├── optimization/       # Tối ưu hóa kế hoạch
├── planning/           # Lập kế hoạch điều trị
├── reporting/          # Tạo báo cáo
├── scripts/            # Script hỗ trợ
├── segmentation/       # Phân đoạn cấu trúc
├── specialized/        # Tính năng đặc biệt
├── treatment/          # Mô-đun điều trị
├── ui/                 # Giao diện người dùng
└── tests/              # Kiểm thử tự động
```

## Đóng góp

Chúng tôi chào đón mọi đóng góp từ cộng đồng! Nếu bạn muốn đóng góp, vui lòng:

1. Fork dự án
2. Tạo nhánh tính năng mới (`git checkout -b feature/AmazingFeature`)
3. Commit các thay đổi của bạn (`git commit -m 'Add some AmazingFeature'`)
4. Push lên nhánh (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

Hãy đảm bảo đọc [Hướng dẫn đóng góp](CONTRIBUTING.md) trước khi bắt đầu.

## Giấy phép

Dự án này được cấp phép theo Giấy phép GNU Affero General Public License v3.0 - xem tệp [LICENSE](LICENSE) để biết chi tiết.

## Liên hệ

- Email dự án: project@example.com
- GitHub Issues: [https://github.com/username/quangtps/issues](https://github.com/username/quangtps/issues)

## Lời cảm ơn

- Các thư viện mã nguồn mở mà chúng tôi sử dụng
- Cộng đồng vật lý y học và bác sĩ xạ trị
- Tất cả những người đóng góp cho dự án

---

*QuangTPS - Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở - Phiên bản 0.1.0*