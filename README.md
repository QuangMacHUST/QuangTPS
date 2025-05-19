# QuangTPS: Hệ thống Lập kế hoạch Xạ trị Mã nguồn mở

<div align="center">
  <img src="quangtps/ui/icons/new_icons/quang_tps_logo.png" alt="QuangTPS Logo" width="200"/>
</div>

![Phiên bản](https://img.shields.io/badge/Phiên_bản-0.8.1-blue)
![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-green)
![Giấy phép](https://img.shields.io/badge/Giấy_phép-MIT-yellow)

## Tổng quan

QuangTPS là một hệ thống lập kế hoạch xạ trị mã nguồn mở cung cấp đầy đủ các công cụ cần thiết cho việc lập kế hoạch điều trị bệnh nhân trong xạ trị. Từ việc nhập dữ liệu hình ảnh DICOM, phân đoạn cấu trúc, tối ưu hóa và tính toán phân bố liều xạ trị, QuangTPS cung cấp một giải pháp toàn diện cho các chuyên gia vật lý xạ trị và các nhà nghiên cứu.

### Tính năng chính

- **Nhập/Xuất DICOM**: Nhập CT, MRI, PET và xuất RT Structure, RT Dose, RT Plan
- **Phân đoạn cấu trúc**: Công cụ phân đoạn thủ công và tự động với hỗ trợ AI
- **Lập kế hoạch xạ trị**:
  - Kỹ thuật 3D-CRT, IMRT, VMAT, SRS/SBRT
  - Tối ưu hóa dựa trên ràng buộc và mục tiêu
  - Tối ưu hóa đa tiêu chí (MCO) với giao diện Pareto Navigator hiện đại
- **Thuật toán tính liều**:
  - Pencil Beam, Collapsed Cone, Monte Carlo
  - Hỗ trợ Monte Carlo GPU với tăng tốc 50-200x
- **Lập kế hoạch thích ứng**:
  - Dự đoán thay đổi giải phẫu
  - Tạo kế hoạch thích ứng tự động
  - Đánh giá độ bền vững của kế hoạch
- **Đánh giá kế hoạch**:
  - DVH, chỉ số đánh giá lâm sàng
  - Phân tích gamma
  - So sánh kế hoạch
- **Đảm bảo chất lượng**:
  - Phân tích log file máy điều trị
  - QA cho kế hoạch xạ trị
  - Báo cáo QA tự động

### Cải tiến trong phiên bản 0.8.1

- **Tối ưu hóa đa tiêu chí (MCO) nâng cao**:
  - Biểu đồ Pareto 3D tương tác với việc tùy chọn trục và màu sắc
  - Giao diện khám phá Pareto surface hiện đại tương tự Eclipse của Varian
  - Chức năng lịch sử di chuyển trong không gian Pareto
  - Tương tác trực tiếp với bề mặt Pareto để chọn kế hoạch tối ưu

- **Nâng cao đánh giá kế hoạch**:
  - Các chỉ số đánh giá lâm sàng tự động: HI, CI, EUD, gBED
  - Đánh giá tự động theo protocol lâm sàng
  - So sánh kế hoạch theo nhiều chỉ số sinh học
  - Báo cáo đánh giá chi tiết với xuất PDF/HTML

- **Báo cáo lâm sàng chuyên nghiệp**:
  - Hệ thống mẫu báo cáo phong cách Eclipse
  - Tích hợp DVH, hình ảnh và phân phối liều
  - Nhiều định dạng xuất: PDF, HTML, DICOM SR
  - Quản lý và lưu trữ báo cáo

- **Giao diện người dùng nâng cao**:
  - Chia màn hình linh hoạt để so sánh nhiều kế hoạch
  - Widget đánh giá kế hoạch tự động
  - Tương tác trực quan với đối tượng 3D
  - Tối ưu hóa hiệu suất cho dữ liệu lớn

### Cải tiến trong phiên bản 0.8.0

- **Hiển thị liều 3D nâng cao**:
  - Hiển thị 3D phân phối liều với đồ họa VTK hiệu suất cao
  - Nhiều chế độ hiển thị: surface, volume rendering, MIP, X-Ray
  - Hiển thị cấu trúc 3D với tùy chọn độ trong suốt
  - Tích hợp với tab External Beam Planning

### Cải tiến trong phiên bản 0.7.4

- **Cải thiện tương thích và ổn định**:
  - Sửa lỗi "Module matplotlib.cm has no viridis member" trong hiển thị BEV
  - Cải thiện xử lý colormap với cơ chế dự phòng nhiều lớp
  - Nâng cao khả năng tích hợp giữa các module thích ứng
  - Thêm xử lý ngoại lệ khi import các thuật toán tính liều
  - Tạo cơ chế tự động đăng ký thuật toán với xử lý khi module không khả dụng

- **Tích hợp Monte Carlo GPU**:
  - Tạo lớp MonteCarloGPUAlgorithm kế thừa từ MonteCarloGPU
  - Tích hợp với hệ thống thuật toán tính liều chuẩn của QuangTPS
  - Chuyển đổi đúng định dạng beam_arrangement sang định dạng cấu hình
  - Thêm tính năng so sánh kết quả với lưới liều tham chiếu

- **Nâng cao hệ thống lập kế hoạch thích ứng**:
  - Tái cấu trúc hàm create_integrated_adaptive_system với tham số tùy chọn
  - Thêm cơ chế dự phòng với các lựa chọn backup_predictor, backup_planner, backup_validator
  - Cải thiện xử lý ngoại lệ khi thiết lập liên kết giữa các thành phần
  - Export lớp DeformableAnatomyPredictor với cơ chế giả mạch khi module thực không khả dụng

## Cài đặt

### Yêu cầu

- Python 3.8, 3.9 hoặc 3.10
- 8GB RAM trở lên (khuyến nghị: 16GB)
- GPU với hỗ trợ CUDA (tùy chọn, cho tính toán Monte Carlo)

### Hướng dẫn cài đặt

```bash
# Clone repository
git clone https://github.com/username/QuangTPS.git
cd QuangTPS

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # trên Windows: venv\Scripts\activate

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt

# Cài đặt gói trong chế độ phát triển
pip install -e .
```

## Bắt đầu sử dụng

```bash
# Khởi động QuangTPS
python -m quangtps
```

## Tài liệu

- [Hướng dẫn người dùng](docs/user_guide/index.md)
- [Hướng dẫn nhà phát triển](docs/developer_guide/index.md)
- [Tài liệu API](docs/api/index.md)

## Đóng góp

Chúng tôi khuyến khích và hoan nghênh các đóng góp từ cộng đồng. Vui lòng đọc [hướng dẫn đóng góp](CONTRIBUTING.md) để biết chi tiết về quy trình gửi pull request.

## Giấy phép

QuangTPS được phát hành dưới [Giấy phép MIT](LICENSE).

## Liên hệ

Nếu bạn có câu hỏi hoặc góp ý, vui lòng tạo issue hoặc liên hệ qua email: example@example.com.