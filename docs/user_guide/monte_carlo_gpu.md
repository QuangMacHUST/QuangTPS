# Hướng dẫn sử dụng Monte Carlo GPU

## Giới thiệu

Module tính toán Monte Carlo GPU là tính năng mới trong QuangTPS, cho phép tăng tốc tính toán liều lượng từ 50-200 lần so với phiên bản CPU truyền thống. Tính năng này đặc biệt hữu ích khi làm việc với kế hoạch điều trị phức tạp hoặc khi cần tính toán liều chi tiết trong môi trường không đồng nhất.

## Yêu cầu hệ thống

- GPU hỗ trợ CUDA hoặc OpenCL
- Trình điều khiển NVIDIA (cho CUDA) hoặc trình điều khiển GPU tương thích với OpenCL
- RAM GPU tối thiểu 4GB (khuyến nghị 8GB trở lên)
- Một trong các thư viện sau:
  - CuPy (ưu tiên cho GPU NVIDIA)
  - PyOpenCL (cho các GPU khác)
  - Numba CUDA (thay thế cho CuPy)

## Cài đặt các thư viện cần thiết

Để sử dụng tính năng Monte Carlo GPU, bạn cần cài đặt các thư viện Python bổ sung:

```bash
# Cho GPU NVIDIA (CUDA)
pip install cupy-cuda11x  # Thay x bằng phiên bản CUDA của bạn, ví dụ: cupy-cuda116

# HOẶC cho các GPU khác (OpenCL)
pip install pyopencl

# Tùy chọn - cho tăng tốc bổ sung
pip install numba
```

## Cách sử dụng

### Sử dụng trong giao diện đồ họa

1. Mở kế hoạch điều trị của bạn trong QuangTPS
2. Chọn tab "Tính toán liều"
3. Chọn thuật toán "Monte Carlo GPU" từ danh sách thuật toán
4. Cấu hình các tham số (số lượng histories, số lượng batches...)
5. Nhấn "Tính toán liều" để bắt đầu tính toán

### Sử dụng trong Python

```python
from quangtps.dose.algorithms.improvements.monte_carlo_gpu import MonteCarloGPU

# Khởi tạo thuật toán
mc_gpu = MonteCarloGPU(
    num_histories=1_000_000,  # Số lượng histories
    num_batches=10,           # Số lượng batches
    use_variance_reduction=True,  # Sử dụng kỹ thuật giảm phương sai
    gpu_id=0,                 # ID của GPU (0 cho GPU đầu tiên, -1 để tự động chọn)
    max_gpu_memory=0.8        # Tỷ lệ tối đa của bộ nhớ GPU sẽ sử dụng (0-1)
)

# Tính toán liều
dose_matrix = mc_gpu.calculate_dose(ct_data, structures, beams)

# Tính toán liều và độ không chắc chắn
result = mc_gpu.calculate_dose_with_uncertainty(ct_data, structures, beams)
dose_matrix = result.dose_matrix
uncertainty_matrix = result.uncertainty_matrix

# Lấy thông tin về hiệu suất tính toán
performance_stats = result.summary()
print(f"Thời gian tính toán: {performance_stats['computation_time']:.2f} giây")
print(f"Số lượng histories: {performance_stats['num_histories']}")
print(f"Hiệu suất: {performance_stats['performance']:.2f} histories/giây")
print(f"Độ không chắc chắn trung bình: {performance_stats['mean_uncertainty']:.4f}")
```

### Sử dụng qua API

```python
import requests
import json

url = "http://localhost:5000/api/dose/calculate"
data = {
    "algorithm": "MONTE_CARLO_GPU",
    "parameters": {
        "num_histories": 1000000,
        "num_batches": 10,
        "gpu_id": 0
    },
    "plan_id": "your_plan_id"
}

response = requests.post(url, json=data)
result = response.json()
```

## Kiểm tra GPU khả dụng

Để kiểm tra các GPU khả dụng trên hệ thống của bạn:

```python
from quangtps.dose.algorithms.improvements.monte_carlo_gpu import get_available_devices

# Liệt kê các GPU khả dụng
devices = get_available_devices()
for device in devices:
    print(device)
```

## So sánh hiệu suất

Để so sánh hiệu suất giữa Monte Carlo GPU và CPU:

```python
from quangtps.dose.algorithms.improvements.monte_carlo_gpu import MonteCarloGPU

mc_gpu = MonteCarloGPU(num_histories=1_000_000)
comparison = mc_gpu.compare_with_cpu(ct_data, structures, beams)

print(f"Thời gian GPU: {comparison['gpu_time']:.2f} giây")
print(f"Thời gian CPU: {comparison['cpu_time']:.2f} giây")
print(f"Tăng tốc: {comparison['speedup']:.2f}x")
print(f"Sai số trung bình: {comparison['mean_rel_diff']*100:.2f}%")
```

## Cấu hình nâng cao

### Phân tán tính toán trên nhiều GPU

Nếu hệ thống của bạn có nhiều GPU, bạn có thể phân tán tính toán trên tất cả các GPU để tăng tốc độ hơn nữa. Để sử dụng tính năng này, bạn cần cài đặt thư viện `dask`:

```bash
pip install dask distributed
```

Sau đó, bạn có thể sử dụng như sau:

```python
from quangtps.dose.algorithms.improvements.monte_carlo_gpu import MonteCarloGPU

# Tự động sử dụng tất cả GPU có sẵn
mc_gpu = MonteCarloGPU(use_all_gpus=True)

# Tính toán liều
dose_matrix = mc_gpu.calculate_dose(ct_data, structures, beams)
```

### Các tham số thuật toán nâng cao

- `use_variance_reduction`: Sử dụng kỹ thuật giảm phương sai (mặc định: True)
- `save_intermediate_results`: Lưu kết quả trung gian sau mỗi batch (mặc định: False)
- `use_beam_splitting`: Chia tính toán theo chùm tia (mặc định: True)
- `precision`: Độ chính xác của tính toán ("single" hoặc "double", mặc định: "single")

## Xử lý sự cố

### GPU không được phát hiện

- Kiểm tra trình điều khiển GPU của bạn đã được cài đặt chính xác
- Kiểm tra thư viện CUDA/OpenCL đã được cài đặt
- Thử chạy `nvidia-smi` (NVIDIA) hoặc `clinfo` (OpenCL) để xác nhận GPU hoạt động

### Lỗi bộ nhớ GPU

- Giảm giá trị `max_gpu_memory` xuống 0.5 hoặc thấp hơn
- Tăng số lượng batches để giảm lượng bộ nhớ cần thiết
- Giảm kích thước ma trận liều bằng cách giảm độ phân giải

### Không tương thích với phiên bản CUDA

- Kiểm tra phiên bản CUDA trên hệ thống của bạn: `nvcc --version`
- Cài đặt phiên bản CuPy tương thích: `pip install cupy-cudaXXX`

## Kết luận

Tính năng Monte Carlo GPU cung cấp hiệu suất tính toán liều vượt trội so với phương pháp CPU truyền thống, cho phép tính toán chính xác hơn và nhanh hơn. Điều này giúp cải thiện chất lượng kế hoạch điều trị và giảm thời gian lập kế hoạch, đặc biệt cho các trường hợp phức tạp.