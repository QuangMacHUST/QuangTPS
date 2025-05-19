# QuangTPS: Hệ thống Lập kế hoạch Xạ trị Mã nguồn mở

<div align="center">
  <img src="quangtps/ui/icons/new_icons/quang_tps_logo.png" alt="QuangTPS Logo" width="200"/>
</div>

![Phiên bản](https://img.shields.io/badge/Phiên_bản-0.7.6-blue)
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

### Cải tiến trong phiên bản 0.7.6

- **Cải thiện tương thích và ổn định**:
  - Cải thiện giao diện MCO Navigator với xử lý sự kiện tốt hơn
  - Nâng cao khả năng tích hợp giữa các module thích ứng
  - Cải thiện hệ thống dự đoán thay đổi giải phẫu với API tiêu chuẩn
  - Tăng cường độ tin cậy với cơ chế dự phòng và xử lý ngoại lệ toàn diện

- **Trải nghiệm người dùng tốt hơn**:
  - Nâng cao animation và hiệu ứng trực quan trong giao diện
  - Cải thiện hiệu suất hiển thị với cơ chế cập nhật thông minh
  - Tối ưu hóa xử lý sự kiện người dùng với debounce và thời gian phản hồi nhanh
  - Đồng bộ hóa tốt hơn giữa các thành phần UI trong tương tác

- **Sửa lỗi quan trọng**:
  - Khắc phục các lỗi lint trong module MCO Navigator
  - Cải thiện xử lý lỗi trong tích hợp hệ thống thích ứng
  - Tăng cường độ tin cậy khi module không khả dụng
  - Chuẩn hóa API để dễ dàng mở rộng trong tương lai

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