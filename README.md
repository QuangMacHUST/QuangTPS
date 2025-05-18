# QuangTPS - Hệ thống Lập Kế hoạch Xạ trị Mã nguồn Mở

<p align="center">
  <img src="docs/images/quangtps_logo.png" alt="QuangTPS Logo" width="200"/>
</p>

![Phiên bản](https://img.shields.io/badge/Phiên%20bản-0.7.8-blue)
![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-green)
![Giấy phép](https://img.shields.io/badge/Giấy%20phép-MIT-yellow)
![Trạng thái](https://img.shields.io/badge/Trạng%20thái-Đang%20phát%20triển-orange)

## Giới thiệu

QuangTPS là hệ thống lập kế hoạch xạ trị (TPS) mã nguồn mở toàn diện cung cấp các tính năng và công cụ cần thiết để lập kế hoạch điều trị ung thư bằng bức xạ. Hệ thống được phát triển với mục tiêu tạo ra một nền tảng mạnh mẽ, linh hoạt và dễ tiếp cận cho các nhà nghiên cứu, bác sĩ và kỹ sư y sinh.

<p align="center">
  <img src="docs/images/quangtps_screenshot.png" alt="QuangTPS Screenshot" width="80%"/>
</p>

## Tính năng chính

- **Xử lý hình ảnh y tế đa dạng**: Hỗ trợ đọc/ghi DICOM, phân đoạn, hiển thị 3D
- **Lập kế hoạch xạ trị nâng cao**: IMRT, VMAT, proton, brachytherapy
- **Tối ưu hóa đa tiêu chí (MCO)**: Khám phá mặt Pareto với điều hướng trực quan
- **Tính toán liều chính xác**: Thuật toán tích hợp Monte Carlo GPU
- **Lập kế hoạch thích ứng**: Cập nhật kế hoạch dựa trên thay đổi giải phẫu
- **Đánh giá kế hoạch**: Phân tích DVH, chỉ số sinh học, và đánh giá tự động
- **Báo cáo linh hoạt**: Tạo báo cáo tùy chỉnh với nhiều mẫu
- **Tích hợp và mở rộng**: API cho phát triển module mở rộng

### Tính năng mới trong phiên bản 0.7.8

- **Hiển thị 3D nâng cao**: Lớp `Image3DWidget` hỗ trợ nhiều chế độ hiển thị (surface, volume, MIP, X-ray) với các công nghệ PyVista và VTK
- **Phân tích gamma tích hợp**: Module phân tích gamma 3D/2D đầy đủ với hỗ trợ phân tích theo vùng liều và thống kê chi tiết
- **Monte Carlo GPU cải tiến**: Tự động điều chỉnh cấu hình dựa trên tài nguyên phần cứng với dự phòng CPU thông minh
- **Giao diện tương tự Eclipse**: Bảng màu isodose và giao diện người dùng tương tự hệ thống Eclipse của Varian
- **Tích hợp thông minh**: Tự động phát hiện và tận dụng các thư viện khả dụng, chuyển đổi linh hoạt giữa các backend

## Tính năng mới trong phiên bản 0.7.9

### Phân tích Gamma tốc độ cao với GPU
- **Tăng tốc GPU**: Đạt tốc độ tính toán phân tích gamma nhanh hơn 20-50 lần với GPU
- **Kernel CUDA tối ưu**: Triển khai các kernel CuPy tùy chỉnh để tính toán khoảng cách hiệu quả
- **Dự phòng tự động**: Tự động chuyển về CPU khi không có GPU khả dụng
- **Trực quan hóa kết quả**: Biểu đồ histogram và bản đồ nhiệt với colormap tùy chỉnh

### Tích hợp Monte Carlo GPU và Phân tích Gamma
- Tính toán và so sánh kế hoạch điều trị chính xác và nhanh chóng
- Phân tích theo vùng liều với tính toán song song
- Báo cáo chi tiết với biểu đồ trực quan

### Hiện đại hóa theo dõi thay đổi
- CHANGELOG.md chuẩn định dạng Keep a Changelog
- Phân loại các thay đổi theo Added, Changed, Fixed
- Tự động liên kết các phiên bản với tag GitHub

## Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.2+ (cho tính năng Monte Carlo GPU)
- 8GB RAM trở lên
- 2GB VRAM trở lên
- Windows 10/11, Linux, hoặc macOS

## Cài đặt

```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Cài đặt từ PyPI
pip install quangtps

# Hoặc cài đặt từ mã nguồn
git clone https://github.com/yourusername/QuangTPS.git
cd QuangTPS
pip install -e .
```

## Sử dụng nhanh

```python
from quangtps import QuangTPS

# Khởi tạo hệ thống
tps = QuangTPS()

# Tải dữ liệu bệnh nhân
patient = tps.load_patient('path/to/dicom/data')

# Lập kế hoạch VMAT đơn giản
plan = tps.create_vmat_plan(patient,
                            prescription=60.0,
                            fractions=30,
                            target_structure='PTV')

# Tối ưu hóa kế hoạch
optimizer = tps.get_optimizer(algorithm='gradient_descent')
result = optimizer.optimize(plan)

# Tính toán liều với Monte Carlo
dose = tps.calculate_dose(plan, algorithm='monte_carlo_gpu')

# Đánh giá kế hoạch
evaluation = tps.evaluate_plan(plan, dose)
print(evaluation.summary())

# Hiển thị DVH
tps.show_dvh(dose)

# Tạo báo cáo
tps.create_report(plan, dose, evaluation, template='clinical')
```

## Tài liệu

Tài liệu đầy đủ có sẵn tại: https://quangtps.readthedocs.io

- [Hướng dẫn người dùng](docs/user_guide/)
- [Hướng dẫn phát triển](docs/developer_guide/)
- [API Documentation](docs/api/)
- [Ví dụ](examples/)

## Đóng góp

Chúng tôi chào đón sự đóng góp từ cộng đồng! Vui lòng xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết.

## Giấy phép

QuangTPS được phát hành dưới giấy phép MIT. Xem [LICENSE](LICENSE) để biết thêm chi tiết.