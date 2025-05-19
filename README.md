# QuangTPS: Hệ thống Lập kế hoạch Xạ trị Mã nguồn mở

<div align="center">
  <img src="quangtps/ui/icons/new_icons/quang_tps_logo.png" alt="QuangTPS Logo" width="200"/>
</div>

![Phiên bản](https://img.shields.io/badge/Phiên_bản-0.8.10-blue)
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

### Cải tiến trong phiên bản 0.8.10

- **Hoàn thiện tính năng phân đoạn tự động (Auto Segmentation)**:
  - Tích hợp AutoSegmentationEngine vào Structure tab với giao diện đầy đủ
  - Phân đoạn tự động các cấu trúc thông qua mô hình học máy tiên tiến
  - Giao diện người dùng trực quan với các tùy chọn và thông báo tiến trình
  - Xử lý kết quả phân đoạn với khả năng tích hợp vào quy trình lập kế hoạch

- **Cải thiện tính ổn định module MCO**:
  - Khắc phục lỗi liên quan đến hiển thị bề mặt Pareto 3D
  - Tăng cường khả năng phục hồi với cơ chế fallback cho các thành phần không khả dụng
  - Đảm bảo hiển thị đồ thị Pareto màu sắc hoạt động trong mọi môi trường

- **Nâng cao khả năng xử lý ngoại lệ toàn hệ thống**:
  - Cải thiện xử lý lỗi khi các thư viện phụ thuộc không khả dụng
  - Cơ chế phục hồi thông minh với khả năng thông báo rõ ràng
  - Trải nghiệm người dùng mượt mà ngay cả khi thiếu thành phần không thiết yếu

### Cải tiến trong phiên bản 0.8.9

- **Hoàn thiện module đánh giá kế hoạch tích hợp phong cách Eclipse**:
  - Tạo module báo cáo kế hoạch chuyên nghiệp với nhiều định dạng xuất (PDF, HTML, CSV)
  - Định nghĩa chuẩn cho hơn 20 loại mục tiêu lâm sàng đầy đủ phong cách Eclipse
  - Hệ thống điểm đánh giá kế hoạch với thang điểm trực quan từ "Kém" đến "Xuất sắc"

- **Bộ đánh giá kế hoạch xạ trị toàn diện**:
  - Đánh giá tự động kế hoạch theo protocol lâm sàng với tính điểm riêng cho PTV và OAR
  - Tính toán chỉ số đồng nhất (HI), chỉ số phù hợp (CI) và các chỉ số nâng cao khác
  - Phân tích kết quả với hiển thị trực quan bằng màu sắc và biểu đồ

- **Nâng cao độ tin cậy của hệ thống**:
  - Xử lý ngoại lệ và lỗi import thông minh với cơ chế fallback tự động
  - Dễ dàng mở rộng với các module mới nhờ kiến trúc mô-đun hóa
  - Tăng cường khả năng phục hồi khi thiếu thư viện bên ngoài

### Cải tiến trong phiên bản 0.8.8

- **Hệ thống đánh giá kế hoạch xạ trị toàn diện theo phong cách Eclipse**:
  - Giao diện hiện đại tích hợp DVH, đánh giá mục tiêu lâm sàng và thống kê liều
  - Hiển thị điểm đánh giá kế hoạch với thang màu từ đỏ đến xanh theo chất lượng
  - Đánh giá tự động các mục tiêu lâm sàng với hiển thị màu sắc trực quan
  - Khả năng tùy chỉnh và chỉnh sửa protocol lâm sàng trực tiếp trong giao diện

- **Tính năng báo cáo đánh giá chuyên nghiệp**:
  - Báo cáo PDF với đầy đủ thông tin kế hoạch, DVH và kết quả đánh giá
  - Báo cáo HTML tương tác với khả năng phóng to/thu nhỏ biểu đồ
  - Xuất dữ liệu đánh giá ra CSV để phân tích nâng cao
  - Tích hợp so sánh kế hoạch để phân tích nhiều phương án điều trị

- **Tích hợp hoàn chỉnh với các thành phần hiện có**:
  - Kết nối liền mạch với module DVH và hiển thị liều 3D
  - Tích hợp với protocol manager để quản lý các tiêu chí lâm sàng
  - Xử lý ngoại lệ thông minh khi thiếu các thành phần phụ thuộc
  - Giao diện đồng nhất theo phong cách Eclipse trong toàn hệ thống

### Cải tiến trong phiên bản 0.8.7

- **Tích hợp hoàn chỉnh hiển thị liều 3D vào External Beam Planning**:
  - Tích hợp liền mạch DoseVisualization3D vào tab External Beam Planning
  - Hiển thị đồng thời phân bố liều 3D, isodose và cấu trúc giải phẫu
  - Đồng bộ hóa tự động giữa hiển thị 3D và DVH khi dữ liệu thay đổi
  - Cập nhật trực quan khi cấu trúc hoặc chùm tia thay đổi

- **Giao diện Eclipse-style cho External Beam Planning tab**:
  - Thiết kế lại layout với panel trái quản lý kế hoạch và panel phải hiển thị kết quả
  - Chọn chế độ lập kế hoạch (Forward, Inverse, MCO) bằng radio button trực quan
  - Tab chùm tia và mục tiêu tối ưu hóa được tổ chức hợp lý và dễ sử dụng
  - Status bar hiển thị thông tin trạng thái và tiến trình

- **Xử lý lỗi toàn diện**:
  - Placeholder thông minh cho các thành phần không khả dụng
  - Chế độ dự phòng khi VTK hoặc PyQt5 không hoạt động
  - Chuyển đổi tự động giữa các định dạng dữ liệu liều
  - Logging chi tiết để dễ dàng gỡ lỗi và theo dõi

### Cải tiến trong phiên bản 0.8.6

- **Hoàn thiện module phân tích độ bền vững**:
  - Module toàn diện mô phỏng Eclipse Robustness Analyzer với giao diện hiện đại
  - Đánh giá độ bền vững kế hoạch đối với độ không chắc chắn về vị trí và phạm vi
  - Hỗ trợ phân tích nhiều kịch bản với số lượng tùy chỉnh
  - Phân tích chỉ số đầy đủ cho cả mục tiêu và cơ quan nguy cấp

- **Trực quan hóa kết quả phân tích độ bền vững nâng cao**:
  - Hiển thị DVH bands cho từng cấu trúc với dải biến thiên theo từng kịch bản
  - Biểu đồ độ phủ mục tiêu với phân tích min-max range
  - Bản đồ nhiệt phân tích không gian hiển thị các điểm nóng của sự thay đổi liều
  - Bảng chỉ số đánh giá với màu sắc trực quan theo mức độ biến thiên

- **Tạo báo cáo phân tích độ bền vững chuyên nghiệp**:
  - Xuất báo cáo CSV và Excel đa sheet với phân tích chi tiết
  - Tạo báo cáo PDF với bảng, biểu đồ và đánh giá lâm sàng
  - Tạo báo cáo HTML tương tác với khả năng lọc và hiển thị tùy chỉnh
  - Tích hợp đánh giá tự động dựa trên ngưỡng lâm sàng

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