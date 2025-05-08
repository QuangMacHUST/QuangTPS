# Hướng dẫn sử dụng Monte Carlo trong QuangTPS

## Giới thiệu

Monte Carlo là thuật toán tính toán liều lượng chính xác nhất hiện có trong lĩnh vực xạ trị. QuangTPS triển khai thuật toán Monte Carlo hiệu năng cao với khả năng tận dụng GPU để tăng tốc đáng kể thời gian tính toán, giúp đưa phương pháp này vào sử dụng trong thực hành lâm sàng hàng ngày.

Thuật toán Monte Carlo của QuangTPS cung cấp độ chính xác cao cho tất cả các loại mô (đồng nhất và không đồng nhất), các kế hoạch điều trị phức tạp (IMRT, VMAT, SRS), và có thể xử lý được các implant kim loại, cũng như các điều kiện biên phức tạp.

## Nguyên lý hoạt động

Monte Carlo mô phỏng vật lý tương tác bức xạ một cách trực tiếp bằng cách theo dõi hàng triệu "lịch sử hạt" (particle histories). Mỗi hạt được mô phỏng riêng biệt từ khi phát ra từ nguồn cho đến khi mất năng lượng hoặc thoát khỏi không gian tính toán.

Quá trình tính toán gồm các bước chính:
1. Mô phỏng nguồn bức xạ (Linear accelerator simulation)
2. Vận chuyển hạt qua các vật liệu (Particle transport)
3. Tương tác vật lý: tán xạ Compton, hấp thụ quang điện, sản sinh cặp, v.v.
4. Tính toán liều tích lũy qua tất cả các lịch sử hạt

## Tính năng chính của Monte Carlo trong QuangTPS

- **Tăng tốc GPU đa thiết bị**: Hỗ trợ tính toán song song trên nhiều GPU, giảm thời gian tính toán đến 50-100 lần so với CPU
- **Tối ưu hóa Variance Reduction**: Sử dụng nhiều kỹ thuật giảm phương sai để cải thiện hiệu quả thống kê
- **Chia batch thích ứng**: Tự động chia nhỏ vấn đề lớn để tránh lỗi hết bộ nhớ GPU
- **Hỗ trợ nhiều loại hạt**: Mô phỏng photon, electron và các hạt thứ cấp
- **Denoising thông minh**: Công nghệ giảm nhiễu thống kê mà không làm mất thông tin không gian
- **Tương thích đầy đủ**: Hoạt động với mọi loại vật liệu, mật độ, và cấu hình chùm tia
- **Tính toán liều theo thời gian thực**: Hiển thị kết quả sơ bộ trong khi đang tính toán
- **Phân tích độ không đảm bảo thống kê**: Báo cáo chi tiết về độ không đảm bảo thống kê của liều

## Yêu cầu hệ thống

### Phiên bản CPU:
- CPU đa nhân (khuyến nghị tối thiểu 4 nhân)
- RAM tối thiểu 8GB (khuyến nghị 16GB hoặc cao hơn)
- Không cần thêm phần cứng đặc biệt

### Phiên bản GPU (khuyến nghị):
- GPU NVIDIA hỗ trợ CUDA Compute Capability 3.5 hoặc cao hơn
- Bộ nhớ GPU tối thiểu 4GB (khuyến nghị 8GB trở lên)
- Driver NVIDIA phiên bản 460 trở lên
- Tối thiểu CUDA Toolkit 10.0 (khuyến nghị 11.0 trở lên)

## Cách sử dụng

### Bước 1: Chọn thuật toán Monte Carlo
1. Trong giao diện QuangTPS, tạo hoặc mở kế hoạch điều trị cần tính toán
2. Vào menu **Tính toán liều lượng** (Dose Calculation)
3. Chọn thuật toán **Monte Carlo** từ danh sách thuật toán

### Bước 2: Cấu hình các tham số tính toán
Cấu hình các tham số quan trọng của Monte Carlo:

#### Tham số cơ bản:
- **Số lịch sử hạt (Histories)**: Số lượng hạt để mô phỏng
  - 1,000,000 → Kết quả nhanh, nhiễu cao
  - 10,000,000 → Cân bằng giữa tốc độ và độ chính xác
  - 100,000,000 → Độ chính xác cao, thời gian tính toán dài
- **Độ không đảm bảo thống kê mục tiêu**: Giá trị phần trăm tối đa của độ không đảm bảo thống kê
  - 2% → Đủ cho kế hoạch lâm sàng thông thường
  - 1% → Cho các kế hoạch cần độ chính xác cao
  - <1% → Cho nghiên cứu hoặc SRS
- **Sử dụng GPU**: Bật/tắt tính toán GPU

#### Tham số nâng cao:
- **Năng lượng cắt (Energy Cutoff)**: Ngưỡng năng lượng tối thiểu để tiếp tục theo dõi hạt
  - Photon: 0.01 MeV (mặc định)
  - Electron: 0.2 MeV (mặc định)
- **Kích thước voxel**: Kích thước voxel cho tính toán
  - 2mm → Độ chính xác cao
  - 3mm → Cân bằng giữa tốc độ và độ chính xác
  - 5mm → Tính toán nhanh
- **Phương pháp giảm nhiễu**: Chọn thuật toán giảm nhiễu
  - Gaussian → Đơn giản, hiệu quả cho các trường tĩnh
  - SVD → Tốt cho các kế hoạch IMRT phức tạp
  - Adaptive → Thông minh nhất, điều chỉnh theo mức độ nhiễu cục bộ
- **Chế độ song song**: Tùy chỉnh mức độ song song
  - Số luồng (CPU)
  - GPU IDs nếu có nhiều GPU

### Bước 3: Chạy tính toán
1. Nhấn nút **Tính toán** (Calculate) để bắt đầu
2. Thanh tiến trình sẽ hiển thị trạng thái của quá trình tính toán:
   - Khởi tạo dữ liệu (vật liệu, tiết diện)
   - Mô phỏng vận chuyển hạt
   - Tích lũy liều
   - Xử lý kết quả và giảm nhiễu

### Bước 4: Phân tích kết quả
Sau khi tính toán hoàn thành, bạn có thể:
1. Xem phân bố liều 3D trên các mặt cắt CT
2. Kiểm tra DVH (Biểu đồ thể tích-liều lượng)
3. Phân tích liều tại các cấu trúc quan tâm
4. Quan sát bản đồ độ không đảm bảo thống kê

## Các phương pháp giảm phương sai (Variance Reduction Techniques)

Để tối ưu hóa hiệu suất, Monte Carlo trong QuangTPS triển khai nhiều kỹ thuật giảm phương sai:

1. **Photon Splitting**: Tách mỗi photon thành nhiều photon với trọng số thấp hơn, giúp cải thiện thống kê mà không tăng thời gian tính toán
2. **Russian Roulette**: Loại bỏ có chọn lọc các hạt năng lượng thấp để tăng hiệu quả
3. **Importance Sampling**: Tập trung mô phỏng vào các vùng có đóng góp lớn đến kết quả cuối cùng
4. **Woodcock Tracking**: Tăng tốc vận chuyển hạt trong môi trường không đồng nhất
5. **Track-Length Estimator**: Ghi nhận liều theo chiều dài đường đi thay vì điểm tương tác

## Mẹo tối ưu hóa hiệu suất

1. **Tận dụng GPU**: Luôn bật tính năng GPU nếu có phần cứng hỗ trợ. Hiệu suất cải thiện đáng kể
2. **Điều chỉnh số lịch sử hạt**: Bắt đầu với số thấp (1-10M) cho tính toán nhanh, sau đó tăng dần nếu cần
3. **Cân nhắc kích thước voxel**: Sử dụng voxel lớn hơn cho tính toán sơ bộ
4. **Sử dụng ROI**: Giới hạn vùng tính toán bằng cách tạo ROI bao quanh PTV + lề
5. **Điều chỉnh mức độ giảm nhiễu**: Tăng độ mạnh của denoising nếu cần cải thiện thị giác
6. **Nhiều GPU**: Nếu hệ thống có nhiều GPU, bật tùy chọn đa GPU để tăng tốc
7. **Ưu tiên đúng mục đích**: Sử dụng thuật toán nhanh hơn (như Acuros XB) cho tính toán sơ bộ, và Monte Carlo cho xác minh cuối cùng

## So sánh với các thuật toán khác

| Thuật toán    | Thời gian tính toán | Độ chính xác | Xử lý bất đồng nhất | Trường hợp phù hợp |
|---------------|---------------------|--------------|---------------------|-------------------|
| Pencil Beam   | Rất nhanh           | Thấp-trung bình | Kém               | QA nhanh, kế hoạch sơ bộ |
| Convolution   | Nhanh               | Trung bình      | Khá               | Các kế hoạch thường quy |
| Acuros XB     | Trung bình          | Cao          | Tốt                 | Đa số kế hoạch lâm sàng |
| Monte Carlo   | Chậm (CPU)/Nhanh (GPU) | Rất cao    | Xuất sắc           | KH phức tạp, nghiên cứu, SRS |

## Xử lý sự cố

### Vấn đề phổ biến và cách khắc phục

#### 1. Thời gian tính toán quá lâu
- **Nguyên nhân**: Số lịch sử hạt quá lớn hoặc không sử dụng GPU
- **Giải pháp**:
  - Giảm số lịch sử hạt
  - Bật tính năng GPU
  - Tăng kích thước voxel
  - Giới hạn ROI tính toán

#### 2. Lỗi hết bộ nhớ GPU
- **Nguyên nhân**: Vùng tính toán quá lớn hoặc bộ nhớ GPU không đủ
- **Giải pháp**:
  - Giảm kích thước vùng tính toán
  - Tăng kích thước voxel
  - Điều chỉnh giảm số lịch sử hạt
  - Nâng cấp phần cứng GPU

#### 3. Kết quả nhiễu thống kê
- **Nguyên nhân**: Không đủ số lịch sử hạt
- **Giải pháp**:
  - Tăng số lịch sử hạt
  - Sử dụng thuật toán denoising mạnh hơn
  - Kiểm tra xem có đang sử dụng đúng các kỹ thuật giảm phương sai

#### 4. Sai lệch liều lượng ở các vùng không đồng nhất
- **Nguyên nhân**: Chuyển đổi CT không chính xác hoặc tiết diện không đúng
- **Giải pháp**:
  - Kiểm tra lại đường cong chuyển đổi HU sang vật liệu
  - Điều chỉnh tham số mô phỏng vật lý

## Tham khảo

1. Kawrakow I, Rogers DWO. The EGSnrc Code System: Monte Carlo simulation of electron and photon transport. NRCC Report PIRS-701, 2000.
2. Karbalaee M, et al. An approach in radiation therapy treatment planning: A fast, GPU-based Monte Carlo method. J Med Signals Sens. 2017;7(2):108-113.
3. Jia X, et al. GPU-based fast Monte Carlo simulation for radiotherapy dose calculation. Phys Med Biol. 2011;56(22):7017-31.

## Phụ lục: Các tham số nâng cao khác

### Tham số vật lý
- **Phổ năng lượng nguồn**: Mặc định sử dụng dữ liệu từ mô hình chùm tia
- **Dữ liệu tiết diện**: NIST (mặc định), ICRP, hoặc tùy chỉnh
- **Xử lý electron thứ cấp**: Đầy đủ hoặc cục bộ (local deposit)

### Tham số kỹ thuật
- **Precision**: Đơn (single) hoặc kép (double)
- **Batch size**: Kích thước lô xử lý (tự động hoặc tùy chỉnh)
- **Seed**: Hạt giống cho bộ tạo số ngẫu nhiên
- **Tần suất báo cáo**: Tần suất cập nhật trạng thái tính toán