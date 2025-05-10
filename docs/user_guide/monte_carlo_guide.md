# Hướng dẫn sử dụng Monte Carlo trong QuangTPS

## Giới thiệu

Monte Carlo là thuật toán tính toán liều lượng chính xác nhất hiện có trong lĩnh vực xạ trị. QuangTPS triển khai thuật toán Monte Carlo hiệu năng cao với khả năng tận dụng GPU để tăng tốc đáng kể thời gian tính toán, giúp đưa phương pháp này vào sử dụng trong thực hành lâm sàng hàng ngày.

Thuật toán Monte Carlo của QuangTPS cung cấp độ chính xác cao cho tất cả các loại mô (đồng nhất và không đồng nhất), các kế hoạch điều trị phức tạp (IMRT, VMAT, SRS), và có thể xử lý được các implant kim loại, cũng như các điều kiện biên phức tạp.

### Cập nhật mới nhất (phiên bản 0.7.2)

Với phiên bản 0.7.2, chúng tôi đã thực hiện các cải tiến quan trọng cho thuật toán Monte Carlo:

- **Thống nhất API**: Các triển khai Monte Carlo khác nhau đã được hợp nhất thành một API thống nhất
- **Tăng cường hỗ trợ GPU**: Hỗ trợ nhiều framework GPU (CUDA thông qua CuPy, Numba CUDA, OpenCL)
- **Cải thiện khả năng chuyển đổi**: Tương thích tốt hơn giữa các phiên bản QuangTPS
- **Điều chỉnh các tham số**: Thêm nhiều tùy chọn nâng cao để tối ưu hiệu suất và độ chính xác
- **Xử lý ngoại lệ mạnh mẽ hơn**: Tự động chuyển đổi giữa GPU và CPU khi gặp lỗi

## Nguyên lý hoạt động

Monte Carlo mô phỏng vật lý tương tác bức xạ một cách trực tiếp bằng cách theo dõi hàng triệu "lịch sử hạt" (particle histories). Mỗi hạt được mô phỏng riêng biệt từ khi phát ra từ nguồn cho đến khi mất năng lượng hoặc thoát khỏi không gian tính toán.

Quá trình tính toán gồm các bước chính:
1. Mô phỏng nguồn bức xạ (Linear accelerator simulation)
2. Vận chuyển hạt qua các vật liệu (Particle transport)
3. Tương tác vật lý: tán xạ Compton, hấp thụ quang điện, sản sinh cặp, v.v.
4. Tính toán liều tích lũy qua tất cả các lịch sử hạt

## Tính năng chính của Monte Carlo trong QuangTPS

- **Tăng tốc GPU đa nền tảng**: Hỗ trợ CUDA (CuPy/Numba) và OpenCL, tự động lựa chọn tối ưu
- **Hỗ trợ đa GPU**: Tự động phân phối tính toán trên nhiều GPU để tăng tốc đáng kể
- **Tối ưu hóa Variance Reduction**: Sử dụng nhiều kỹ thuật giảm phương sai để cải thiện hiệu quả thống kê
- **Chia batch thích ứng**: Tự động chia nhỏ vấn đề lớn để tránh lỗi hết bộ nhớ GPU
- **Hỗ trợ nhiều loại hạt**: Mô phỏng photon, electron và các hạt thứ cấp
- **Denoising thông minh**: Nhiều tùy chọn giảm nhiễu thống kê mà không làm mất thông tin không gian
- **Tương thích đầy đủ**: Hoạt động với mọi loại vật liệu, mật độ, và cấu hình chùm tia
- **Tính toán liều theo thời gian thực**: Hiển thị kết quả sơ bộ trong khi đang tính toán
- **Phân tích độ không đảm bảo thống kê**: Báo cáo chi tiết về độ không đảm bảo thống kê của liều
- **Tự động phục hồi khi lỗi**: Nếu GPU gặp lỗi, tự động chuyển sang CPU hoặc GPU khác

## Yêu cầu hệ thống

### Phiên bản CPU:
- CPU đa nhân (khuyến nghị tối thiểu 4 nhân, tối ưu với 8+ nhân)
- RAM tối thiểu 8GB (khuyến nghị 16GB hoặc cao hơn)
- Hỗ trợ AVX2 để tối ưu hóa hiệu suất (tùy chọn)

### Phiên bản GPU (khuyến nghị):
- **NVIDIA GPU**: Compute Capability 3.5+ (Kepler, Maxwell, Pascal, Volta, Turing, Ampere hoặc mới hơn)
  - Bộ nhớ GPU tối thiểu 4GB (khuyến nghị 8GB+ cho các trường hợp phức tạp)
  - Driver NVIDIA phiên bản 460 trở lên
  - CUDA Toolkit 10.0+ (khuyến nghị 11.0+)
- **AMD GPU** (qua OpenCL):
  - Kiến trúc GCN 2.0 trở lên (Radeon R9 series hoặc mới hơn)
  - Driver AMD phiên bản mới nhất
- **Intel GPU** (qua OpenCL, hiệu suất hạn chế hơn):
  - Integrated GPU Gen9 trở lên (Skylake hoặc mới hơn)
  - Driver Intel mới nhất

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
- **Framework GPU**: (Mới trong phiên bản 0.7.2)
  - Auto → Tự động chọn framework tốt nhất
  - CUDA (CuPy) → Hiệu suất cao nhất cho GPU NVIDIA
  - CUDA (Numba) → Tùy chọn thay thế cho NVIDIA
  - OpenCL → Hỗ trợ GPU AMD, Intel

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
6. **Fast Multipole Method** (Mới): Tăng tốc tính toán tương tác đa hạt

## Mẹo tối ưu hóa hiệu suất

1. **Tận dụng GPU đúng cách**:
   - Đối với GPU NVIDIA: Chọn CUDA (CuPy) hoặc CUDA (Numba)
   - Đối với GPU AMD: Chọn OpenCL
   - Đối với hệ thống không có GPU mạnh: Sử dụng chế độ CPU với nhiều luồng

2. **Tối ưu hóa cấu hình khi sử dụng nhiều GPU**:
   - Bật chế độ Multi-GPU trong tab nâng cao
   - Phân bổ công việc phù hợp bằng cách chỉ định GPU IDs cụ thể
   - Tăng GPU Batch Size khi sử dụng GPU có dung lượng bộ nhớ lớn

3. **Điều chỉnh số lịch sử hạt**: Bắt đầu với số thấp (1-10M) cho tính toán nhanh, sau đó tăng dần nếu cần

4. **Cân nhắc kích thước voxel**: Sử dụng voxel lớn hơn cho tính toán sơ bộ

5. **Sử dụng ROI**: Giới hạn vùng tính toán bằng cách tạo ROI bao quanh PTV + lề

6. **Điều chỉnh mức độ giảm nhiễu**: Tăng độ mạnh của denoising nếu cần cải thiện thị giác

7. **Theo dõi bảng điều khiển**: Quan sát thông tin trong bảng điều khiển để phát hiện vấn đề tiềm ẩn

8. **Đối với các kế hoạch phức tạp**:
   - Sử dụng adaptive batch processing để tránh lỗi hết bộ nhớ
   - Bật chế độ multilevel parallelism để tận dụng cả CPU và GPU
   - Cân nhắc sử dụng chế độ kết hợp CPU+GPU nếu có nhiều lõi CPU mạnh

## Tính năng nâng cao trong phiên bản 0.7.2

### 1. Hệ thống phát hiện và phục hồi tự động
QuangTPS 0.7.2 giờ đây có thể tự động phát hiện và phục hồi từ các lỗi GPU, chuyển sang CPU hoặc GPU khác khi cần thiết. Điều này đảm bảo tính toán không bị gián đoạn ngay cả khi gặp sự cố phần cứng.

### 2. Tùy chọn framework GPU
Người dùng có thể lựa chọn giữa các framework khác nhau để tận dụng tối đa phần cứng:
- **CuPy**: Hiệu suất tốt nhất cho GPU NVIDIA mới
- **Numba CUDA**: Tính ổn định cao cho NVIDIA
- **OpenCL**: Hỗ trợ đa dạng GPU (NVIDIA, AMD, Intel)

### 3. Fast Multipole Method (FMM)
Thuật toán tăng tốc mới sử dụng FMM để xử lý nhanh các tương tác đa hạt, giảm thời gian tính toán lên tới 30% trong các trường hợp phức tạp.

### 4. Dự đoán thời gian
Hệ thống ước tính thời gian hoàn thành dựa trên hiệu suất thực tế và tiến độ hiện tại, giúp người dùng lập kế hoạch công việc hiệu quả hơn.

## So sánh với các thuật toán khác

| Thuật toán    | Thời gian tính toán | Độ chính xác | Xử lý bất đồng nhất | Trường hợp phù hợp |
|---------------|---------------------|--------------|---------------------|-------------------|
| Pencil Beam   | Rất nhanh (s)      | Thấp-trung bình | Kém               | QA nhanh, kế hoạch sơ bộ |
| Convolution   | Nhanh (s-min)       | Trung bình      | Khá               | Các kế hoạch thường quy |
| AAA           | Trung bình (min)    | Trung bình-cao  | Tốt               | Kế hoạch lâm sàng thông thường |
| Acuros XB     | Trung bình (min)    | Cao             | Rất tốt           | Đa số kế hoạch lâm sàng |
| Monte Carlo CPU| Chậm (min-hr)      | Rất cao         | Xuất sắc          | Nghiên cứu, phức tạp |
| Monte Carlo GPU| Nhanh (min)        | Rất cao         | Xuất sắc          | KH phức tạp, nghiên cứu, SRS |

## Xử lý sự cố

### Vấn đề phổ biến và cách khắc phục

#### 1. Thời gian tính toán quá lâu
- **Nguyên nhân**: Số lịch sử hạt quá lớn hoặc không sử dụng GPU
- **Giải pháp**:
  - Giảm số lịch sử hạt
  - Bật tính năng GPU
  - Tăng kích thước voxel
  - Giới hạn ROI tính toán
  - Chọn framework GPU phù hợp

#### 2. Lỗi hết bộ nhớ GPU
- **Nguyên nhân**: Vùng tính toán quá lớn hoặc bộ nhớ GPU không đủ
- **Giải pháp**:
  - Bật chế độ "adaptive_histories" trong tham số nâng cao
  - Giảm kích thước vùng tính toán
  - Tăng kích thước voxel
  - Sử dụng chế độ "batched processing"

#### 3. Kết quả nhiễu thống kê
- **Nguyên nhân**: Không đủ số lịch sử hạt
- **Giải pháp**:
  - Tăng số lịch sử hạt
  - Sử dụng thuật toán denoising mạnh hơn
  - Kiểm tra xem có đang sử dụng đúng các kỹ thuật giảm phương sai

#### 4. GPU không được nhận diện
- **Nguyên nhân**: Vấn đề với driver hoặc phần mềm CUDA/OpenCL
- **Giải pháp**:
  - Cập nhật driver GPU
  - Thử chuyển sang framework khác (CuPy ↔ Numba ↔ OpenCL)
  - Kiểm tra xem GPU có được cài đặt đúng cách

## Phụ lục: Các tham số nâng cao bổ sung trong v0.7.2

### Tham số kỹ thuật mới
- **use_fmm_acceleration**: Sử dụng Fast Multipole Method để tăng tốc (mặc định: true)
- **use_avx_vectorization**: Sử dụng AVX cho tính toán CPU (mặc định: true)
- **multi_gpu**: Sử dụng nhiều GPU nếu có (mặc định: true)
- **use_multilevel_parallelism**: Kết hợp CPU và GPU (mặc định: true)
- **use_opencl_fallback**: Tự động chuyển sang OpenCL nếu CUDA không khả dụng (mặc định: true)
- **adaptive_histories**: Tự động điều chỉnh số lịch sử hạt dựa trên độ không đảm bảo (mặc định: true)

### Trạng thái tương thích API
Với phiên bản 0.7.2, các API cũ từ phiên bản trước vẫn được hỗ trợ thông qua lớp tương thích, nhưng sẽ hiển thị cảnh báo không dùng nữa (deprecation warnings). Người dùng và nhà phát triển nên chuyển sang sử dụng API mới.

## Tham khảo

1. Kawrakow I, Rogers DWO. The EGSnrc Code System: Monte Carlo simulation of electron and photon transport. NRCC Report PIRS-701, 2000.
2. Karbalaee M, et al. An approach in radiation therapy treatment planning: A fast, GPU-based Monte Carlo method. J Med Signals Sens. 2017;7(2):108-113.
3. Jia X, et al. GPU-based fast Monte Carlo simulation for radiotherapy dose calculation. Phys Med Biol. 2011;56(22):7017-31.