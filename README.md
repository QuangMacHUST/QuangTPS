# QuangTPS: Hệ thống Lập kế hoạch Xạ trị Mã nguồn mở

<div align="center">
  <img src="quangtps/ui/icons/new_icons/quang_tps_logo.png" alt="QuangTPS Logo" width="200"/>
</div>

![Phiên bản](https://img.shields.io/badge/Phiên_bản-0.9.2-blue)
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

### Cải tiến trong phiên bản 0.9.1

- **Knowledge-Based Planning (KBP) phong cách RapidPlan**:
  - Giao diện KBP hiện đại với thiết kế tương tự RapidPlan của Eclipse
  - Dự đoán tự động các tham số tối ưu từ dữ liệu kế hoạch trước đó
  - Phân tích thông minh đặc trưng hình học và liều lượng
  - Tích hợp liền mạch vào quy trình lập kế hoạch ngược

- **Nút KBP trong thanh công cụ External Beam Planning**:
  - Truy cập nhanh chức năng KBP từ giao diện chính
  - Hiển thị thông tin mô hình và các đề xuất trực quan
  - Áp dụng tự động đề xuất vào kế hoạch hiện tại
  - Hỗ trợ tối ưu hóa tự động sau khi áp dụng đề xuất

- **Phân tích các đặc trưng quan trọng**:
  - Hiển thị đóng góp của các đặc trưng trong mô hình KBP
  - Hỗ trợ phân tích khoảng cách từ PTV đến các OAR
  - Dự đoán các tham số tối ưu cho các cấu trúc cụ thể
  - Điều chỉnh trọng số mục tiêu tối ưu theo kinh nghiệm lâm sàng

- **Cải thiện trải nghiệm người dùng**:
  - Biểu tượng chuyên nghiệp trong thanh công cụ
  - Thông báo trực quan với hướng dẫn rõ ràng
  - Xử lý ngoại lệ toàn diện cho tất cả tính năng
  - Giao diện nhất quán theo phong cách Eclipse hiện đại

### Cải tiến trong phiên bản 0.9.2

- **Nâng cấp độ ổn định của module MCO (Multi-Criteria Optimization)**:
  - Khắc phục lỗi indentation gây trở ngại trong việc import module
  - Cải thiện phương thức lựa chọn giải pháp Pareto tối ưu
  - Tăng cường xử lý ngoại lệ khi các thành phần không khả dụng
  - Tích hợp liền mạch với quy trình lập kế hoạch ngược

- **Tối ưu hóa tích hợp KBP (Knowledge-Based Planning)**:
  - Chuẩn hóa hệ thống biểu tượng với hàm create_eclipse_icon
  - Khắc phục lỗi import trong kbp_dialog.py cho tăng tính tương thích
  - Cải thiện quy trình xử lý đề xuất KBP vào kế hoạch hiện tại
  - Đồng bộ hóa giữa KBP và hệ thống tối ưu hóa ngược

- **Tăng cường tính ổn định toàn hệ thống**:
  - Cập nhật cơ chế import trong hệ thống dialog với cơ chế dự phòng
  - Cải thiện hệ thống thông báo lỗi với chi tiết và hướng dẫn khắc phục
  - Đảm bảo khả năng hoạt động khi thiếu các thành phần không thiết yếu
  - Tăng cường truy vết lỗi và logging để dễ dàng gỡ lỗi

### Cải tiến trong phiên bản 0.9.0

- **Tối ưu hiệu năng tính toán liều**:
  - Cải thiện thuật toán Monte Carlo CPU và GPU cho hiệu suất cao hơn 30%
  - Tính toán song song đa luồng với hỗ trợ tối ưu cho nhiều nền tảng
  - Quản lý bộ nhớ thông minh giảm 40% lượng RAM sử dụng cho kế hoạch lớn
  - Hiển thị 3D hiệu suất cao cho kế hoạch phức tạp với nhiều cấu trúc

- **Hệ thống dialog toàn diện phong cách Eclipse**:
  - Import thông minh với cơ chế dự phòng tự động cho tất cả dialog
  - Thông báo lỗi hữu ích với hướng dẫn chi tiết khi thiếu thành phần
  - Hiển thị tự động PlanPropertiesDialog khi chỉnh sửa kế hoạch
  - ColorButton trong StructurePropertiesDialog với bảng màu trực quan

- **Tích hợp MCO nâng cao**:
  - Cải thiện giao diện Pareto Navigator với hiển thị 3D đẹp mắt
  - Điều hướng trực quan trên bề mặt Pareto với trải nghiệm mượt mà
  - Thêm cân bằng và phân tích độ nhạy cho các tiêu chí tối ưu
  - Hệ thống lưu trữ và truy xuất giải pháp Pareto hiệu quả

- **Tích hợp quy trình làm việc liền mạch**:
  - Kết nối thông minh giữa Object Explorer và các dialog thuộc tính
  - Đồng bộ hóa tự động giữa các thành phần khi dữ liệu thay đổi
  - Hệ thống thông báo và cảnh báo toàn diện với hướng dẫn khắc phục
  - Giao diện nhất quán theo phong cách Eclipse trong toàn bộ hệ thống

### Cải tiến trong phiên bản 0.8.16

- **Hoàn thiện hệ thống dialog phong cách Eclipse**:
  - Dialog so sánh kế hoạch với biểu đồ DVH chồng và bảng so sánh chỉ số
  - Dialog chỉnh sửa thuộc tính kế hoạch với giao diện trực quan
  - Dialog chỉnh sửa thuộc tính cấu trúc với ColorButton tùy chỉnh
  - Tích hợp đầy đủ các dialog vào quy trình làm việc chính
  - Giao diện nhất quán theo phong cách Eclipse hiện đại

- **Nâng cao khả năng phục hồi khi thiếu thành phần**:
  - Xử lý ngoại lệ toàn diện khi import các module không khả dụng
  - Lớp giả mạch cho tất cả các thành phần có thể thiếu
  - Thông báo lỗi chi tiết với hướng dẫn khắc phục
  - Đảm bảo hệ thống vẫn hoạt động với chức năng cơ bản

- **Cải thiện tích hợp giữa các thành phần**:
  - Đồng bộ hóa giữa Object Explorer và các dialog thuộc tính
  - Kết nối liền mạch giữa PlanComparisonDialog và DVH Widget
  - Nhất quán dữ liệu khi chỉnh sửa thuộc tính đối tượng
  - Cập nhật tự động các thành phần UI khi dữ liệu thay đổi

### Cải tiến trong phiên bản 0.8.15

- **Hoàn thiện tích hợp Object Explorer phong cách Eclipse**:
  - Kết nối liền mạch giữa Object Explorer và các thành phần chính của hệ thống
  - Xử lý ngoại lệ thông minh khi kết nối signal/slot
  - Cải thiện giao diện với splitter điều chỉnh kích thước
  - Đồng bộ hóa tự động giữa Object Explorer và các tab chức năng

- **Tái cấu trúc hệ thống tab hiện đại**:
  - Thiết kế lại bố cục tab đúng phong cách Eclipse
  - Tích hợp các tab chuyên biệt: Plan Evaluation Report, Robust Analysis
  - Tab MCO Navigator thông minh (hiển thị khi cần thiết)
  - Chuyển đổi mượt mà giữa các tab chức năng

- **Nâng cao độ tin cậy của hệ thống**:
  - Khắc phục nhiều lỗi linter trong các file UI chính
  - Xử lý ngoại lệ toàn diện cho quá trình tính toán liều
  - Cải thiện hệ thống import với cơ chế try-except và fallback
  - Đảm bảo hoạt động cơ bản ngay cả khi thiếu thành phần không thiết yếu

- **Cải thiện đồng bộ hóa dữ liệu**:
  - Cập nhật tự động Object Explorer khi tải kế hoạch hoặc cấu trúc mới
  - Đồng bộ đa chiều giữa Object Explorer và các module chính
  - Logging chi tiết và báo cáo lỗi trong quá trình đồng bộ
  - Thông báo trạng thái trực quan trong status bar

### Cải tiến trong phiên bản 0.8.14

- **Object Explorer Panel hoàn chỉnh theo phong cách Eclipse**:
  - Hiển thị và quản lý bệnh nhân, cấu trúc giải phẫu và kế hoạch xạ trị trong một giao diện hợp nhất
  - Hỗ trợ tìm kiếm đối tượng với bộ lọc thời gian thực
  - Hiển thị cấu trúc với màu sắc trực quan tương ứng với màu thực tế
  - Menu ngữ cảnh đầy đủ cho các thao tác phổ biến (tạo mới, sửa, xóa)
  - Checkbox hiển thị/ẩn cấu trúc trực tiếp trên panel

- **Tích hợp Object Explorer Panel vào hệ thống**:
  - Đồng bộ hóa tự động giữa Object Explorer và các tab khác
  - Tương tác hai chiều giữa Object Explorer và các module chức năng
  - Quản lý toàn diện cho bệnh nhân, cấu trúc, và kế hoạch xạ trị
  - Tạo cấu trúc và kế hoạch mới trực tiếp từ Object Explorer

- **Nâng cao độ tin cậy của hệ thống**:
  - Cải thiện xử lý ngoại lệ cho các thành phần không khả dụng
  - Bổ sung cơ chế dự phòng khi các phương thức quan trọng không tồn tại
  - Sửa nhiều lỗi linter và khắc phục các vấn đề tiềm ẩn
  - Xử lý các trường hợp ngoại lệ toàn diện

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