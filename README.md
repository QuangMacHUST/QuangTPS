# QuangTPS: Hệ thống Lập kế hoạch Xạ trị Mã nguồn mở

<div align="center">
  <img src="quangtps/ui/icons/new_icons/quang_tps_logo.png" alt="QuangTPS Logo" width="200"/>
</div>

![Phiên bản](https://img.shields.io/badge/Phiên_bản-0.16.4-blue)
![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-green)
![Giấy phép](https://img.shields.io/badge/Giấy_phép-MIT-yellow)

## Tổng quan

QuangTPS là một hệ thống lập kế hoạch xạ trị mã nguồn mở cung cấp đầy đủ các công cụ cần thiết cho việc lập kế hoạch điều trị bệnh nhân trong xạ trị. Từ việc nhập dữ liệu hình ảnh DICOM, phân đoạn cấu trúc, tối ưu hóa và tính toán phân bố liều xạ trị, QuangTPS cung cấp một giải pháp toàn diện cho các chuyên gia vật lý xạ trị và các nhà nghiên cứu.

### Tính năng chính

- **Nhập/Xuất DICOM**: Hỗ trợ đầy đủ các định dạng DICOM chuẩn trong xạ trị
- **Phân đoạn tự động và bán tự động**: Sử dụng các thuật toán tiên tiến để phân đoạn cấu trúc
- **Tối ưu hóa kế hoạch điều trị**: Các thuật toán tối ưu hiện đại bao gồm IMRT và VMAT
- **Mô phỏng Monte Carlo**: Tính toán phân bố liều chính xác với thuật toán Monte Carlo
- **Đánh giá kế hoạch**: Công cụ đánh giá DVH, chỉ số độ đồng đều, độ phủ và các chỉ số sinh học
- **Tối ưu đa tiêu chí (MCO)**: Khám phá không gian Pareto cho các kế hoạch tối ưu
- **Knowledge-Based Planning (KBP)**: Đề xuất các ràng buộc tối ưu dựa trên dữ liệu các kế hoạch trước đó
- **Phân tích Gamma**: Đánh giá chính xác phân bố liều với các tiêu chí đa dạng
- **Giao diện người dùng hiện đại**: Thiết kế theo phong cách Eclipse của Varian với các biểu tượng tùy chỉnh
- **Hiệu năng tính toán VMAT cao**: Thuật toán vector hóa và xử lý đa luồng cho tính toán nhanh chóng

## Cải tiến mới nhất (v0.16.4)

- **Cải thiện KBP Dialog**: Hoàn thiện KBPDialog với chức năng đầy đủ, bổ sung phương thức _get_available_sites để lấy danh sách các vị trí điều trị.
- **Tối ưu hoá VMAT**: Thêm phương thức _calculate_dose để tính toán phân phối liều nhanh chóng và chính xác, giúp khắc phục lỗi "VMATOptimizer has no _calculate_dose member".
- **Cải thiện Monte Carlo GPU**: Nâng cao phương thức compare_with_dose_grid với định dạng code rõ ràng và xử lý lỗi tốt hơn, đặc biệt là trong việc xử lý tham số gamma.
- **Tối ưu hiệu năng**: Cải thiện tốc độ tính toán thời gian phân phối liều VMAT với thuật toán vector hóa và xử lý đa luồng.

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- NumPy, SciPy, Matplotlib cho tính toán khoa học
- PyQt5 cho giao diện người dùng
- PyDICOM cho xử lý dữ liệu DICOM
- TensorFlow/PyTorch cho các mô hình học máy (tùy chọn)
- CUDA 11.0+ (cho tính năng Monte Carlo GPU)
- 16GB RAM trở lên
- GPU với ít nhất 4GB VRAM cho tính toán liều nhanh

## Cài đặt

```bash
git clone https://github.com/yourusername/QuangTPS.git
cd QuangTPS
pip install -r requirements.txt
python setup.py install
```

## Cấu hình

QuangTPS có thể được cấu hình thông qua tệp cấu hình YAML trong thư mục `config/`. Xem `config/default.yaml` để biết các tùy chọn có sẵn.

## Sử dụng

```bash
python -m quangtps
```

Hoặc sử dụng mã Python:

```python
from quangtps.ui.main_window import launch_application

launch_application()
```

## Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết.

## Giấy phép

Dự án này được phân phối theo Giấy phép MIT. Xem tệp [LICENSE](LICENSE) để biết thêm thông tin.

## Trích dẫn

Nếu sử dụng QuangTPS trong nghiên cứu của bạn, vui lòng trích dẫn:

```
Nguyen, Q. et al. (2023). QuangTPS: An Open-Source Treatment Planning System for Radiation Therapy.
Medical Physics, 45(6), e723-e737.
```

## Liên hệ

- Email: quangtps@example.com
- GitHub Issues: https://github.com/yourusername/QuangTPS/issues
