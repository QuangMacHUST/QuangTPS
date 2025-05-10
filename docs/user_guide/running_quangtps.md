# Hướng dẫn chạy và sử dụng QuangTPS

## Yêu cầu hệ thống

### Cấu hình tối thiểu
- **Hệ điều hành**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **CPU**: Intel Core i5/AMD Ryzen 5 hoặc cao hơn, 4 nhân trở lên
- **RAM**: 8GB (16GB khuyến nghị cho các kế hoạch phức tạp)
- **Ổ cứng**: 50GB trống (SSD được khuyến nghị)
- **Màn hình**: Độ phân giải 1920x1080 trở lên
- **GPU**: Card đồ họa tích hợp (cho hiển thị cơ bản)

### Cấu hình khuyến nghị
- **CPU**: Intel Core i7/AMD Ryzen 7 trở lên, 8 nhân trở lên
- **RAM**: 32GB trở lên
- **Ổ cứng**: 100GB trống (SSD NVMe)
- **GPU**:
  - NVIDIA RTX 2060 6GB trở lên (cho tính toán Monte Carlo GPU)
  - AMD Radeon Pro W5500 (hoặc cao hơn)
- **Màn hình**: Nhiều màn hình với độ phân giải cao (2K trở lên)

### Yêu cầu phần mềm
- Python 3.9 hoặc cao hơn
- CUDA Toolkit 11.0+ (cho tính toán GPU với NVIDIA)
- OpenCL runtime (cho tính toán GPU với AMD/Intel)

## Cài đặt

### Cài đặt từ source (phổ biến nhất)

1. Clone repository từ GitHub:
```bash
git clone https://github.com/your-organization/quangtps.git
cd quangtps
```

2. Tạo môi trường ảo:
```bash
# Sử dụng venv
python -m venv venv

# Kích hoạt môi trường
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

4. Cài đặt gói phụ thuộc tùy chọn cho tăng tốc GPU:
```bash
# Cho NVIDIA GPU
pip install cupy-cuda11x
pip install numba

# Cho AMD/Intel GPU
pip install pyopencl
```

5. Cài đặt QuangTPS:
```bash
pip install -e .
```

### Cài đặt qua Conda (thay thế)

1. Tạo môi trường conda:
```bash
conda create -n quangtps python=3.9
conda activate quangtps
```

2. Cài đặt các gói phụ thuộc:
```bash
conda install -c conda-forge numpy scipy matplotlib pyqt vtk pillow pydicom
pip install -r requirements.txt
```

3. Cài đặt QuangTPS:
```bash
pip install -e .
```

## Khởi động QuangTPS

### Cách 1: Từ terminal/command prompt

Sau khi kích hoạt môi trường ảo hoặc conda:

```bash
python -m quangtps
```

Hoặc:

```bash
quangtps
```

### Cách 2: Sử dụng scripts chuyên dụng

#### Windows
```
run_quangtps.bat
```

#### Linux/macOS
```
./run_quangtps.sh
```

## Sử dụng QuangTPS lần đầu

Khi khởi động QuangTPS lần đầu, bạn sẽ thấy một số tùy chọn cấu hình ban đầu:

1. **Cấu hình thư mục dữ liệu**: Chọn nơi lưu trữ dữ liệu bệnh nhân và cơ sở dữ liệu
2. **Cài đặt mặc định**: Chọn các tham số tính toán mặc định
3. **Tối ưu hóa GPU**: Quét và cấu hình GPU cho tính toán nhanh
4. **Kiểm tra cơ sở dữ liệu**: Khởi tạo cơ sở dữ liệu nếu cần thiết

## Quy trình làm việc cơ bản

### 1. Tạo/Chọn Bệnh nhân
- Nhấp vào tab "Patient" để tạo bệnh nhân mới hoặc tìm kiếm bệnh nhân hiện có
- Nhập thông tin bệnh nhân (ID, tên, ngày sinh, etc.)
- Tạo nghiên cứu mới (study) hoặc chọn nghiên cứu hiện có

### 2. Nhập dữ liệu hình ảnh
- Trong tab "Imaging", nhấp "Import Images"
- Chọn thư mục chứa các file DICOM
- Xem và kiểm tra các hình ảnh đã nhập

### 3. Phân đoạn cấu trúc
- Chuyển đến tab "Structure"
- Sử dụng công cụ vẽ thủ công hoặc nhấp "Auto Segment"
- Kiểm tra và chỉnh sửa các cấu trúc đã tạo

### 4. Tạo kế hoạch điều trị
- Chuyển đến tab "Planning"
- Nhấp "New Plan" và chọn loại kế hoạch (IMRT, VMAT, etc.)
- Thêm và cấu hình các chùm tia (beams)
- Đặt mục tiêu và ràng buộc liều
- Nhấp "Optimize" để bắt đầu tối ưu hóa kế hoạch

### 5. Tính toán liều
- Trong tab "Dose", chọn thuật toán tính liều (Monte Carlo, etc.)
- Cấu hình các tham số tính toán
- Nhấp "Calculate Dose" để bắt đầu tính toán

### 6. Đánh giá kế hoạch
- Xem DVH trong tab "Evaluate"
- Kiểm tra các chỉ số chất lượng kế hoạch
- Xem phân bố liều 3D và isodose lines

### 7. Tạo báo cáo
- Chuyển đến tab "Report"
- Chọn mẫu báo cáo
- Tùy chỉnh nội dung báo cáo
- Xuất báo cáo dưới dạng PDF

## Tính năng nâng cao

### Knowledge-Based Planning (KBP)
1. Chuyển đến tab "Planning"
2. Nhấp vào "KBP" trong thanh công cụ
3. Chọn mô hình KBP phù hợp
4. Tùy chỉnh các tham số và khởi chạy tối ưu hóa

### Multicriteria Optimization (MCO)
1. Từ giao diện tối ưu hóa, nhấp "MCO Navigator"
2. Xem và điều hướng trên bề mặt Pareto
3. Chọn điểm cân bằng tối ưu
4. Áp dụng kế hoạch đã chọn

### Lập kế hoạch thích ứng
1. Nhập hình ảnh CT/CBCT mới
2. Chuyển đến "Adaptive Planning"
3. Chọn kế hoạch cần thích ứng
4. Khởi chạy phân tích sự khác biệt giải phẫu
5. Tự động hoặc thủ công điều chỉnh kế hoạch

## Khắc phục sự cố

### Ứng dụng không khởi động
- Kiểm tra Python version (yêu cầu 3.9+)
- Xác nhận tất cả các gói phụ thuộc đã được cài đặt
- Kiểm tra logs trong thư mục `logs/`

### Lỗi tính toán liều
- Kiểm tra cấu hình GPU
- Đảm bảo drivers GPU đã được cập nhật
- Thử sử dụng thuật toán tính liều khác

### Hiệu suất chậm
- Kiểm tra sử dụng RAM và GPU
- Giảm độ phân giải tính toán
- Tắt các ứng dụng khác đang chạy nền

### Lỗi đồ họa hoặc giao diện
- Cập nhật drivers đồ họa
- Kiểm tra phiên bản PyQt/VTK
- Thử giảm độ phân giải hiển thị

## Thông tin liên hệ hỗ trợ

Nếu bạn gặp vấn đề khác không được đề cập ở đây, vui lòng liên hệ:
- Email: support@quangtps.org
- GitHub Issues: https://github.com/your-organization/quangtps/issues
- Diễn đàn: https://forum.quangtps.org

---

*Cập nhật: Tháng 3, 2026*