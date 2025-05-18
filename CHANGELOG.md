# Changelog

Tất cả những thay đổi đáng chú ý của dự án QuangTPS sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án này tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.13] - 2026-05-20

### Thay đổi

- Nâng cấp module tối ưu hóa đa tiêu chí (MCO) với hiển thị bề mặt Pareto nâng cao
- Cải thiện giao diện người dùng điều hướng Pareto tương tự Eclipse của Varian
- Thêm tính năng đồ thị màu nhiệt theo giá trị mục tiêu
- Tối ưu hóa tương tác người dùng với thông tin phản hồi trực quan
- Tăng cường xử lý lỗi và đồng bộ hóa giữa trọng số và các giải pháp

### Sửa lỗi

- Khắc phục vấn đề hiển thị khi không có giải pháp Pareto
- Sửa lỗi đồng bộ hóa trọng số khi chọn giải pháp mới
- Cải thiện xử lý ngoại lệ khi tìm kiếm giải pháp theo trọng số

## [0.7.12] - 2026-05-15

### Thay đổi

- Cải thiện phân tích độ bền vững với tính toán song song và xử lý lỗi toàn diện
- Nâng cấp hiển thị 3D phân phối liều với chuyển đổi dữ liệu tối ưu hơn
- Tăng cường lập kế hoạch thích ứng với đánh giá cấu trúc chuẩn xác hơn

### Sửa lỗi

- Khắc phục vấn đề trong phân tích độ bền vững khi xử lý nhiều kịch bản
- Sửa lỗi định dạng dữ liệu khi chuyển đổi phân phối liều sang VTK
- Cải thiện tính toán hệ số Dice để đánh giá sự khác biệt giữa các cấu trúc

## [Chưa phát hành]

### Thêm mới
- Cải thiện tài liệu hướng dẫn người dùng và API

## [0.7.9] - 2026-04-25

### Thêm mới
- Tạo file `CHANGELOG.md` chuẩn format Keep a Changelog:
  - Cải tiến quản lý nhật ký thay đổi với định dạng chuẩn
  - Tách biệt các thay đổi theo loại (Added, Changed, Fixed)
  - Liên kết các phiên bản với tag GitHub
  - Hỗ trợ tiếng Việt đầy đủ với khả năng hiển thị trên GitHub

- Cập nhật module phân tích gamma trong `quangtps/evaluation/metrics/gamma_analysis.py`:
  - Thêm phương thức `calculate_gamma_3d_gpu` sử dụng CUDA thông qua CuPy
  - Tăng tốc độ tính toán phân tích gamma lên 20-50 lần trên GPU
  - Hỗ trợ dự phòng CPU tự động khi GPU không khả dụng
  - Tối ưu hóa thuật toán để giảm sử dụng bộ nhớ GPU
  - Triển khai các kernel CuPy tùy chỉnh để tăng hiệu suất tính toán khoảng cách
  - Thêm phương thức `plot_gamma_results` để hiển thị kết quả

- Tích hợp phân tích gamma và Monte Carlo GPU:
  - Cải thiện phương thức `compare_with_dose_grid` trong `MonteCarloGPU` để sử dụng GPU
  - Bổ sung khả năng phân tích gamma theo vùng liều với tính toán song song
  - Thêm tính năng xuất báo cáo so sánh chi tiết với biểu đồ

### Cải tiến
- Nâng cao hiệu suất tính toán gamma:
  - Tối ưu hóa thuật toán tìm kiếm khoảng cách tối thiểu
  - Giảm yêu cầu bộ nhớ cho tính toán gamma 3D quy mô lớn
  - Hỗ trợ tính toán tối ưu trên CPU đa nhân khi không có GPU

- Cải thiện trực quan hóa kết quả gamma:
  - Hỗ trợ hiển thị bản đồ nhiệt với các mức đạt/không đạt
  - Tạo biểu đồ histograms tương tác cho phân bố giá trị gamma
  - Hỗ trợ xuất kết quả phân tích dưới dạng PDF và HTML

- Cập nhật requirements.txt với các thư viện mới:
  - Thêm CuPy và PyCUDA với cấu hình theo nền tảng
  - Bổ sung các thư viện hiển thị và phân tích dữ liệu
  - Cải thiện quản lý phụ thuộc cho các môi trường khác nhau

### Sửa lỗi
- Khắc phục vấn đề tràn bộ nhớ GPU khi tính toán gamma với kích thước lớn
- Sửa lỗi index trong phân tích gamma ở vùng biên
- Đảm bảo hoạt động ổn định trên các phiên bản CuPy và CUDA khác nhau
- Cải thiện khả năng tương thích với nhiều loại thiết bị GPU khác nhau
- Cải thiện xử lý ngoại lệ và ghi log chi tiết

## [0.7.8] - 2026-04-20

### Thêm mới
- Tạo lớp `Image3DWidget` trong `quangtps/ui/image_3d_widget.py`:
  - Hỗ trợ hiển thị 3D với nhiều công nghệ (PyVista, VTK) và tự động chuyển đổi
  - Cung cấp nhiều chế độ hiển thị: bề mặt (surface), thể tích (volume), MIP, X-ray
  - Giao diện người dùng hiện đại với thanh công cụ tích hợp và tùy chọn tùy chỉnh

- Cải thiện `MonteCarloGPU` trong `quangtps/dose/algorithms/improvements/monte_carlo_gpu.py`:
  - Thêm phương thức `_setup_cpu_fallback` xử lý khi không có GPU khả dụng
  - Tối ưu hóa tự động dựa trên tài nguyên hệ thống (bộ nhớ, số lõi CPU)
  - Triển khai `_setup_gpu_cupy` và `_setup_gpu_pycuda` để hỗ trợ nhiều backend khác nhau
  - Thêm phương thức `compare_with_dose_grid` sử dụng phân tích gamma

- Module phân tích gamma đầy đủ trong `quangtps/evaluation/metrics/gamma_analysis.py`:
  - Triển khai phân tích gamma 3D và 2D với nhiều tùy chọn linh hoạt
  - Hỗ trợ nội suy SciPy cho kết quả chính xác cao
  - Phương pháp dự phòng không phụ thuộc SciPy cho môi trường thiếu thư viện
  - Hỗ trợ phân tích theo vùng liều và thống kê chi tiết

### Cải tiến
- Tích hợp `quangtps/ui/image_viewer.py` với lớp `Image3DWidget`:
  - Thêm lớp `ImageSliceWidget` để hiển thị lát cắt hình ảnh với nhiều tính năng
  - Hỗ trợ nhiều chế độ hiển thị 3D (surface, volume, MIP, X-ray)
  - Cải thiện bố cục và tính năng tương tác với hình ảnh

- Đăng ký module gamma_analysis.py trong `quangtps/evaluation/metrics/__init__.py`
- Tạo tài liệu API cho gamma_analysis trong `docs/api/gamma_analysis.md`
- Tối ưu hóa hiệu suất hiển thị 3D bằng cách giảm số lượng mesh được hiển thị
- Nâng cao tính mô-đun hóa của mã nguồn để dễ dàng mở rộng và bảo trì

### Sửa lỗi
- Sửa lỗi khi truy cập thuộc tính GPU không tồn tại trong môi trường không có CUDA
- Khắc phục sự cố khi hiển thị cấu trúc 3D với các mask không hợp lệ
- Sửa lỗi import không tìm thấy trong các module phụ thuộc
- Cải thiện độ ổn định khi tính toán liều trong môi trường có tài nguyên hạn chế

## [0.7.7] - 2026-04-15

### Thêm mới
- Cải thiện tính năng thông qua script `isolated_plan_comparison_demo.py`:
  - Demo ứng dụng phân tích liều độc lập sử dụng QuangTPS
  - Hỗ trợ các trường hợp sử dụng quan trọng như so sánh kế hoạch và phân tích cấu trúc
  - Kêt nối nhiều thành phần riêng lẻ của hệ thống

- Thêm cải tiến cho tính năng hiển thị ROI/VOI 3D:
  - Cho phép nhấp chuột vào ROI trong cây cấu trúc để hiển thị/ẩn
  - Tùy chỉnh độ mờ đục cho từng cấu trúc riêng lẻ
  - Thêm tùy chọn hiển thị tất cả cấu trúc đồng thời

### Cải tiến
- Cập nhật hàm hiển thị trong `quangtps/ui/widgets/dose_display.py`:
  - Hỗ trợ colormap trong suốt cho hiển thị liều tốt hơn
  - Thêm kiểm soát chất lượng hiển thị
  - Hỗ trợ điều chỉnh wuadow-level tự động dựa trên histogram

- Cải thiện quy trình lập kế hoạch thích ứng:
  - Nâng cấp module registration.py với các thuật toán mới
  - Thêm hỗ trợ cho cấu trúc tự động và deformation fields
  - Tối ưu hóa độ chính xác và tốc độ của thuật toán

- Tái cấu trúc nhiều thành phần chính:
  - Nâng cấp cấu trúc code để tuân theo PEP 8
  - Cải thiện ghi log và xử lý ngoại lệ
  - Đảm bảo khả năng hoạt động trên các hệ thống không có GPU

[0.7.9]: https://github.com/username/QuangTPS/compare/v0.7.8...v0.7.9
[0.7.8]: https://github.com/username/QuangTPS/compare/v0.7.7...v0.7.8
[0.7.7]: https://github.com/username/QuangTPS/compare/v0.7.6...v0.7.7

## [0.7.11] - 2026-05-10

### Thay đổi

- Nâng cấp hiển thị 3D với colormap VTK tùy chỉnh và Eclipse-style
- Cải thiện xử lý colormap trong BEV với cơ chế dự phòng nhiều lớp
- Tăng cường module thuật toán tính liều với hệ thống kiểm tra GPU tích hợp
- Nâng cao hệ thống thích ứng với xử lý lỗi toàn diện

### Sửa lỗi

- Khắc phục vấn đề "không tìm thấy colormap" trong hiển thị 3D và BEV
- Cải thiện xử lý tình huống không có GPU với dự phòng CPU tự động
- Sửa lỗi khi các thành phần trong hệ thống thích ứng không thể kết nối
- Đảm bảo tính ổn định khi GPU không đủ bộ nhớ cho tính toán Monte Carlo

## [0.7.10] - 2026-05-01

### Thay đổi

- Cải thiện đáng kể module hiển thị 3D với xử lý lỗi tốt hơn
- Tăng cường hiệu suất hiển thị 3D và tối ưu hóa tạo mesh
- Nâng cao trải nghiệm người dùng với nhiều chế độ hiển thị 3D
- Tích hợp tốt hơn giữa `StructureViewer3D` và `structure_tab.py`