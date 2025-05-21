# QuangTPS: Hệ thống Lập kế hoạch Xạ trị Mã nguồn mở

<div align="center">
  <img src="quangtps/ui/icons/new_icons/quang_tps_logo.png" alt="QuangTPS Logo" width="200"/>
</div>

![Phiên bản](https://img.shields.io/badge/Phiên_bản-0.10.2-blue)
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

- **Phân tích sinh học nâng cao**:
  - Đánh giá kế hoạch sử dụng các mô hình sinh học tiên tiến như TCP, NTCP và EUD
  - Tính toán xác suất kiểm soát khối u và biến chứng mô lành với dữ liệu tham số đặc trưng theo cơ quan
  - Trực quan hóa các chỉ số sinh học với biểu đồ màu sắc và bảng so sánh
  - Đánh giá liều tương đương sinh học khi thay đổi phương pháp phân liều

- **Đánh giá độ bền vững của kế hoạch xạ trị**:
  - Phân tích tác động của sai số thiết lập trên phân bố liều
  - Đánh giá độ bền vững với sự không chắc chắn về phạm vi trong xạ trị proton/ion
  - Mô phỏng ảnh hưởng của biến thiên ngẫu nhiên và hệ thống
  - Trực quan hóa dải DVH cho các kịch bản bền vững khác nhau

- **Cải thiện module MCO và KBP**:
  - Khắc phục lỗi indentation trong mco_engine.py gây lỗi import
  - Tích hợp liền mạch KBP với quy trình tối ưu hóa lập kế hoạch
  - Cải thiện chức năng đánh giá và lựa chọn giải pháp tối ưu
  - Nâng cao độ ổn định của các module đặc biệt trong điều kiện hạn chế

- **Nâng cấp hệ thống và hiệu suất**:
  - Tối ưu hóa hiệu suất các thuật toán tính toán chỉ số chất lượng kế hoạch
  - Cải thiện khả năng phục hồi lỗi và xử lý ngoại lệ
  - Bổ sung tài liệu và ghi chú mã nguồn cho việc phát triển và bảo trì dễ dàng
  - Nâng cao tính tương thích với nhiều định dạng dữ liệu y tế khác nhau

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

### Cải tiến trong phiên bản 0.10.2

- **Phân tích độ bền vững (Robustness Analysis) nâng cao**:
  - Đánh giá chi tiết tác động của sai số thiết lập và bất định phạm vi với giao diện trực quan
  - Hiển thị dải DVH (DVH bands) với thiết kế màu sắc thẩm mỹ và thông tin biên độ dao động
  - Đánh dấu trực quan các cấu trúc đã phân tích với dấu "*" và màu nổi bật
  - Tooltip phong phú hiển thị thông tin biến động chi tiết cho từng chỉ số đánh giá
  - Mã màu thông minh đánh giá độ ổn định (xanh lá: rất ổn định, xanh dương: ổn định, vàng: chấp nhận, đỏ: không ổn định)

- **Cải thiện quy trình phân tích độ bền vững**:
  - Xử lý toàn diện các trường hợp đặc biệt và dữ liệu đầu vào không đầy đủ
  - Báo cáo tiến trình chi tiết trong quá trình phân tích với thông báo trạng thái rõ ràng
  - Tự động chuyển đổi dữ liệu thành numpy array để xử lý hiệu quả hơn
  - Tính toán tự động các chỉ số thống kê về độ biến động (biên độ trung bình, biên độ lớn nhất)
  - Tích hợp liền mạch kết quả phân tích với các thành phần khác của hệ thống

- **Giao diện người dùng trực quan**:
  - Hiển thị đánh giá độ ổn định của kế hoạch với mã màu trực quan
  - Đánh dấu và làm nổi bật các cấu trúc đã được phân tích độ bền vững
  - Hiển thị thông tin chi tiết về phạm vi biến động trong tooltip của các chỉ số
  - Tương tác mượt mà giữa dialog phân tích và các thành phần hiển thị kết quả

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

## Cải tiến trong phiên bản 0.9.3

Phiên bản 0.9.3 mang đến nhiều cải tiến đáng kể trong đánh giá kế hoạch xạ trị:

### Phân tích sinh học (Biological Analysis)
- Tính toán các chỉ số sinh học quan trọng: TCP, NTCP, EUD và BED
- Cơ sở dữ liệu tham số sinh học cho nhiều loại cơ quan
- Giao diện trực quan hiển thị các chỉ số sinh học với màu sắc trực quan
- Biểu đồ so sánh TCP/NTCP cho các cấu trúc

### Đánh giá độ bền vững (Robustness)
- Phân tích ảnh hưởng của sai số thiết lập (setup error)
- Đánh giá tác động của độ không chắc chắn phạm vi (range uncertainty) cho xạ trị proton
- Tạo báo cáo tóm tắt với các chỉ số đánh giá độ bền vững
- Biểu đồ trực quan cho kết quả phân tích độ bền vững

### Giao diện người dùng
- Tích hợp các công cụ phân tích sinh học vào tab đánh giá kế hoạch
- Cải thiện hiển thị DVH với các chỉ số lâm sàng
- Sửa lỗi và cải thiện hiệu suất trong nhiều module

## Cải tiến trong phiên bản 0.9.2

- Nâng cấp module MCO (Multi-Criteria Optimization) cho hoạt động ổn định hơn
- Cải thiện giao diện External Beam Planning tab theo phong cách Eclipse
- Tích hợp hoàn chỉnh hiển thị liều 3D vào External Beam Planning tab
- Nâng cấp module DVH với các chỉ số đánh giá kế hoạch tự động