# Hướng dẫn sử dụng Acuros XB GPU trong QuangTPS

## Giới thiệu

Acuros XB là thuật toán tính toán liều lượng tiên tiến dựa trên phương pháp xác định giải phương trình vận chuyển Boltzmann tuyến tính (LBTE). Bản cải tiến Acuros XB GPU trong QuangTPS tận dụng sức mạnh của GPU để tăng tốc đáng kể quá trình tính toán liều lượng trong kế hoạch xạ trị, cung cấp độ chính xác tương đương Monte Carlo nhưng nhanh hơn gấp nhiều lần.

## Ưu điểm của Acuros XB GPU

- **Độ chính xác cao**: Cung cấp độ chính xác gần tương đương với Monte Carlo trong môi trường không đồng nhất.
- **Tốc độ vượt trội**: Nhanh hơn 50-100 lần so với Monte Carlo truyền thống nhờ tận dụng sức mạnh của GPU.
- **Tính toán ổn định**: Ít nhiễu thống kê hơn so với phương pháp Monte Carlo.
- **Mô phỏng chính xác**: Mô phỏng tốt các hiệu ứng vật lý như tán xạ, hấp thụ trong các môi trường không đồng nhất.
- **Hỗ trợ nhiều loại chùm tia**: Xử lý chính xác các loại chùm tia Photon, từ 6MV đến 15MV.

## Yêu cầu hệ thống

- GPU NVIDIA hỗ trợ CUDA với bộ nhớ tối thiểu 4GB (khuyến nghị 8GB hoặc cao hơn)
- Driver NVIDIA phiên bản 460.x trở lên
- CUDA Toolkit 11.0 trở lên
- Hệ điều hành: Windows 10/11 64-bit hoặc Linux 64-bit

## Cách sử dụng

### Bước 1: Kích hoạt tính năng Acuros XB GPU

1. Mở QuangTPS và tải bệnh nhân cần lập kế hoạch
2. Trong menu chính, chọn **Kế hoạch điều trị > Tính toán liều lượng**
3. Trong hộp thoại **Tính toán liều lượng**, chọn thuật toán **Acuros XB**
4. Đánh dấu vào hộp kiểm **Sử dụng GPU** (nếu hệ thống phát hiện GPU tương thích)

### Bước 2: Cấu hình tham số tính toán

Acuros XB GPU có các tham số cấu hình sau:

| Tham số | Mô tả | Giá trị đề xuất |
|---------|-------|----------------|
| Kích thước voxel | Kích thước voxel cho tính toán liều | 2.5mm (cân bằng giữa tốc độ và độ chính xác) |
| Năng lượng cắt | Ngưỡng năng lượng electron cắt | 500 keV (mặc định) |
| Số lượng góc rời rạc | Số góc trong không gian góc | 16 (tiêu chuẩn), 32 (cao cấp) |
| Dung sai hội tụ | Dung sai cho điều kiện dừng | 1e-4 (mặc định) |
| Kích thước batch | Số lượng voxel xử lý đồng thời | Tự động dựa vào bộ nhớ GPU |

### Bước 3: Bắt đầu tính toán

1. Chọn vùng quan tâm (ROI) cho việc tính toán liều
2. Kiểm tra các tham số tính toán và điều chỉnh nếu cần
3. Nhấn nút **Bắt đầu tính toán**

Quá trình tính toán sẽ hiển thị thanh tiến trình cho các giai đoạn:
- Chuẩn bị dữ liệu bệnh nhân
- Xây dựng hạt nhân vận chuyển (transport kernel)
- Tính toán trên GPU
- Chuyển đổi fluence thành liều

### Bước 4: Xem và phân tích kết quả

Sau khi hoàn tất, hệ thống sẽ hiển thị:
- Phân bố liều lượng 3D
- DVH (Biểu đồ thể tích-liều lượng)
- Thống kê liều lượng cho các cấu trúc

## Mẹo để đạt hiệu suất tối ưu

1. **Tối ưu kích thước voxel**: Cân nhắc giữa độ chính xác và tốc độ tính toán
   - 1mm: Độ chính xác cao nhưng chậm hơn
   - 2.5mm: Cân bằng giữa tốc độ và độ chính xác
   - 5mm: Tính toán nhanh nhưng độ chính xác thấp hơn

2. **Tận dụng tính toán song song cho nhiều trường chiếu**: Hệ thống sẽ tự động phân chia công việc tính toán cho các trường chiếu để tận dụng tối đa sức mạnh của GPU.

3. **Kiểm tra bộ nhớ GPU**: Đảm bảo GPU có đủ bộ nhớ cho kích thước tập dữ liệu. Nếu bộ nhớ không đủ, hệ thống sẽ tự động chuyển sang chế độ phân đoạn (chunking) nhưng sẽ chậm hơn.

## So sánh với các thuật toán khác

| Thuật toán | Độ chính xác | Tốc độ | Ứng dụng phù hợp |
|------------|--------------|--------|-----------------|
| Pencil Beam | Thấp | Rất nhanh | Vùng đồng nhất, QA nhanh |
| AAA | Trung bình | Nhanh | Hầu hết các trường hợp lâm sàng |
| Acuros XB (CPU) | Cao | Trung bình | Trường hợp không đồng nhất phức tạp |
| **Acuros XB (GPU)** | Cao | Nhanh | Trường hợp không đồng nhất phức tạp, định dạng VMAT, SRS |
| Monte Carlo | Rất cao | Rất chậm | Nghiên cứu, thẩm định |

## Xử lý sự cố

### Lỗi khởi tạo GPU

Nếu gặp lỗi "Không thể khởi tạo GPU" hoặc "CUDA initialization failed":

1. Kiểm tra driver GPU đã được cài đặt đúng cách
2. Đảm bảo GPU hỗ trợ CUDA và có compute capability 3.5 trở lên
3. Thử khởi động lại phần mềm và máy tính

### Lỗi bộ nhớ không đủ

Nếu gặp lỗi "GPU out of memory":

1. Giảm độ phân giải voxel (tăng kích thước voxel)
2. Giảm kích thước vùng tính toán
3. Sử dụng GPU có bộ nhớ lớn hơn

### Kết quả tính toán không chính xác

Nếu kết quả tính toán không như mong đợi:

1. Kiểm tra dữ liệu đầu vào (CT, cấu trúc, thông số chùm tia)
2. Đảm bảo rằng mật độ vật liệu được ấn định chính xác
3. Thử tăng số lượng góc rời rạc và giảm dung sai hội tụ

## Tham khảo

1. Vassiliev ON, et al. Validation of a new grid-based Boltzmann equation solver for dose calculation in radiotherapy with photon beams. Phys Med Biol. 2010;55(3):581-598.
2. Bush K, et al. Dosimetric validation of Acuros XB with Monte Carlo methods for photon dose calculations. Med Phys. 2011;38(4):2208-2221.
3. Karbalaee M, et al. An approach in radiation therapy treatment planning: A fast, GPU-based Monte Carlo method. J Med Signals Sens. 2017;7(2):108-113.