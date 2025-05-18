# Phân tích Gamma trong QuangTPS

Phân tích gamma là một phương pháp định lượng để so sánh hai phân phối liều, thường được sử dụng để đánh giá chất lượng kế hoạch điều trị và kiểm soát chất lượng trong xạ trị. Module `gamma_analysis` cung cấp các công cụ để thực hiện phân tích gamma trong không gian 2D và 3D.

## API Tổng quan

Module `gamma_analysis` cung cấp các hàm chính sau:

- `calculate_gamma_3d`: Tính toán chỉ số gamma giữa hai phân phối liều 3D
- `calculate_gamma_2d`: Tính toán chỉ số gamma giữa hai phân phối liều 2D
- `gamma_pass_rate`: Tính tỷ lệ điểm vượt qua tiêu chí gamma
- `get_gamma_statistics`: Lấy các thống kê từ phân phối gamma
- `analyze_gamma_by_dose_regions`: Phân tích gamma theo vùng liều

## Cách sử dụng

### Phân tích Gamma 3D

```python
from quangtps.evaluation.metrics import calculate_gamma_3d, gamma_pass_rate

# Hai phân phối liều 3D (reference và evaluation)
reference_dose = dose_grid_1.get_array()
evaluation_dose = dose_grid_2.get_array()
voxel_size = dose_grid_1.get_spacing()  # [dx, dy, dz] mm

# Tính toán gamma 3%/3mm
gamma = calculate_gamma_3d(
    reference_dose=reference_dose,
    evaluation_dose=evaluation_dose,
    voxel_size=voxel_size,
    dose_threshold=3.0,  # 3% tiêu chí liều
    distance_threshold=3.0,  # 3mm tiêu chí khoảng cách
    lower_dose_cutoff=10.0,  # Chỉ tính trong vùng > 10% liều tối đa
    local_gamma=False  # Sử dụng tiêu chí toàn cục
)

# Tính tỷ lệ vượt qua
pass_rate = gamma_pass_rate(gamma, threshold=1.0)
print(f"Tỷ lệ vượt qua gamma 3%/3mm: {pass_rate:.2f}%")
```

### Phân tích Gamma 2D

```python
from quangtps.evaluation.metrics import calculate_gamma_2d, gamma_pass_rate

# Hai phân phối liều 2D (ví dụ: lát cắt từ phân phối liều 3D)
reference_slice = reference_dose[50, :, :]  # Lát cắt thứ 50
evaluation_slice = evaluation_dose[50, :, :]
pixel_size = [voxel_size[1], voxel_size[2]]  # [dy, dz] mm

# Tính toán gamma 2%/2mm
gamma_2d = calculate_gamma_2d(
    reference_dose=reference_slice,
    evaluation_dose=evaluation_slice,
    pixel_size=pixel_size,
    dose_threshold=2.0,
    distance_threshold=2.0
)

# Tính tỷ lệ vượt qua
pass_rate_2d = gamma_pass_rate(gamma_2d)
print(f"Tỷ lệ vượt qua gamma 2D 2%/2mm: {pass_rate_2d:.2f}%")
```

### Lấy thống kê chi tiết

```python
from quangtps.evaluation.metrics import get_gamma_statistics

# Tính toán các thống kê từ phân phối gamma
stats = get_gamma_statistics(gamma)
print(f"Chỉ số gamma trung bình: {stats['mean']:.3f}")
print(f"Chỉ số gamma trung vị: {stats['median']:.3f}")
print(f"Chỉ số gamma tối đa: {stats['max']:.3f}")
print(f"Độ lệch chuẩn gamma: {stats['std']:.3f}")
print(f"Tỷ lệ đạt: {stats['pass_rate']:.2f}%")
```

### Phân tích theo vùng liều

```python
from quangtps.evaluation.metrics import analyze_gamma_by_dose_regions
import matplotlib.pyplot as plt
import numpy as np

# Tạo mảng phần trăm liều tương đối (% so với liều tối đa)
max_dose = np.max(reference_dose)
relative_dose = reference_dose / max_dose * 100.0

# Phân tích theo vùng liều
regions = [(0, 10), (10, 20), (20, 50), (50, 80), (80, 100)]
region_results = analyze_gamma_by_dose_regions(
    gamma=gamma,
    dose=relative_dose,
    dose_regions=regions
)

# Hiển thị kết quả theo vùng
for region_name, stats in region_results.items():
    print(f"Vùng liều {region_name}:")
    print(f"  - Tỷ lệ đạt: {stats['pass_rate']:.2f}%")
    print(f"  - Gamma trung bình: {stats['mean']:.3f}")
    print(f"  - Số voxel: {stats['volume']}")

# Tạo biểu đồ
regions = list(region_results.keys())
pass_rates = [region_results[r]['pass_rate'] for r in regions]

plt.figure(figsize=(10, 6))
plt.bar(regions, pass_rates)
plt.xlabel('Vùng liều (%)')
plt.ylabel('Tỷ lệ đạt gamma (%)')
plt.title('Tỷ lệ đạt gamma theo vùng liều')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.savefig('gamma_region_analysis.png')
plt.close()
```

## API Chi tiết

### `calculate_gamma_3d`

```python
def calculate_gamma_3d(
    reference_dose: np.ndarray,
    evaluation_dose: np.ndarray,
    voxel_size: List[float] = [1.0, 1.0, 1.0],
    dose_threshold: float = 3.0,
    distance_threshold: float = 3.0,
    max_gamma: float = 2.0,
    mask: Optional[np.ndarray] = None,
    lower_dose_cutoff: float = 10.0,
    local_gamma: bool = False,
    interp_method: str = "linear",
    num_threads: int = 1,
) -> np.ndarray:
    """
    Tính toán chỉ số gamma 3D giữa hai phân phối liều.

    Parameters
    ----------
    reference_dose : np.ndarray
        Phân phối liều tham chiếu, 3D numpy array
    evaluation_dose : np.ndarray
        Phân phối liều cần đánh giá, 3D numpy array
    voxel_size : List[float], optional
        Kích thước voxel theo mm cho mỗi chiều [x, y, z], mặc định [1.0, 1.0, 1.0]
    dose_threshold : float, optional
        Tiêu chí sai khác liều (%), mặc định 3.0%
    distance_threshold : float, optional
        Tiêu chí khoảng cách (mm), mặc định 3.0mm
    max_gamma : float, optional
        Giá trị gamma tối đa được tính, các vùng vượt quá sẽ bị cắt ở giá trị này, mặc định 2.0
    mask : np.ndarray, optional
        Mặt nạ nhị phân xác định vùng cần tính gamma, mặc định None (tất cả)
    lower_dose_cutoff : float, optional
        Ngưỡng liều dưới (% so với max) để loại trừ vùng liều thấp, mặc định 10.0%
    local_gamma : bool, optional
        Sử dụng phân tích gamma chuẩn hóa cục bộ thay vì toàn cục, mặc định False
    interp_method : str, optional
        Phương pháp nội suy ('linear', 'nearest'), mặc định 'linear'
    num_threads : int, optional
        Số luồng sử dụng cho tính toán song song, mặc định 1

    Returns
    -------
    np.ndarray
        Mảng 3D chứa giá trị gamma tại mỗi voxel
    """
```

## Tích hợp với Monte Carlo

Phân tích gamma có thể được sử dụng để so sánh kết quả tính toán Monte Carlo với kết quả tính toán từ thuật toán khác hoặc với đo đạc thực tế:

```python
from quangtps.dose.algorithms.improvements.monte_carlo_gpu import MonteCarloGPU

# Khởi tạo Monte Carlo
mc = MonteCarloGPU()

# Tính toán liều với Monte Carlo
mc_result = mc.calculate_dose(beam_arrangement)

# So sánh với phân phối liều khác (ví dụ: PencilBeam)
comparison = mc.compare_with_dose_grid(
    other_dose_grid=pencil_beam_dose,
    dose_threshold_percent=3.0,
    distance_threshold_mm=3.0
)

# Hiển thị kết quả
print(f"Tỷ lệ đạt gamma: {comparison['gamma_pass_rate']:.2f}%")
print(f"Sai khác liều trung bình: {comparison['comparison_metrics']['mean_dose_diff_percent']:.2f}%")

# Phân tích theo vùng liều
for region, stats in comparison['region_analysis'].items():
    print(f"Vùng {region}: {stats['pass_rate']:.2f}% đạt tiêu chí gamma")
```

## Ví dụ thực tế

### So sánh hai kế hoạch điều trị

```python
from quangtps import QuangTPS
from quangtps.evaluation.metrics import calculate_gamma_3d, gamma_pass_rate

# Khởi tạo hệ thống
tps = QuangTPS()

# Tải các kế hoạch
plan1 = tps.load_plan("plan1.dcm")
plan2 = tps.load_plan("plan2.dcm")

# Tính toán liều
dose1 = tps.calculate_dose(plan1)
dose2 = tps.calculate_dose(plan2)

# So sánh phân phối liều với phân tích gamma
gamma = calculate_gamma_3d(
    reference_dose=dose1.get_grid(),
    evaluation_dose=dose2.get_grid(),
    voxel_size=dose1.get_spacing(),
    dose_threshold=3.0,
    distance_threshold=3.0
)

# Tạo báo cáo so sánh
pass_rate = gamma_pass_rate(gamma)
tps.generate_comparison_report(
    plan1=plan1,
    plan2=plan2,
    dose1=dose1,
    dose2=dose2,
    gamma=gamma,
    gamma_pass_rate=pass_rate,
    output_file="plan_comparison.pdf"
)
```

## Ghi chú

- Phân tích gamma sẽ tự động sử dụng SciPy để nội suy nếu thư viện này khả dụng, ngược lại sẽ sử dụng phương pháp đơn giản hơn.
- Đối với dữ liệu lớn (nhiều voxel), khuyến nghị sử dụng nhiều luồng (`num_threads` > 1) để tăng tốc độ tính toán.
- Tiêu chí cục bộ (`local_gamma=True`) thường nghiêm ngặt hơn và phù hợp với vùng có gradient liều cao.