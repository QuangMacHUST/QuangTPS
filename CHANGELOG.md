# Changelog

Tất cả những thay đổi đáng chú ý của dự án QuangTPS sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án này tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.2] - 2023-11-01

### Cải tiến
- Nâng cấp module MCO (Multi-Criteria Optimization) cho hoạt động ổn định hơn:
  - Khắc phục lỗi indentation trong mco_engine.py gây lỗi import
  - Hoàn thiện phương thức select_solution_by_objectives để cải thiện kết quả chọn giải pháp
  - Cải thiện cơ chế xử lý lỗi khi các thành phần không khả dụng
  - Đảm bảo tích hợp liền mạch với External Beam Planning tab

- Cải thiện tích hợp KBP (Knowledge-Based Planning):
  - Thêm hàm create_eclipse_icon trong utils/ui_utils.py để tạo biểu tượng nhất quán
  - Cập nhật import trong kbp_dialog.py để đảm bảo tương thích
  - Tối ưu hóa xử lý ngoại lệ trong quá trình áp dụng đề xuất KBP
  - Cải thiện tương thích giữa KBP và quy trình tối ưu hóa

- Ổn định hệ thống:
  - Cập nhật cơ chế import trong module dialogs để đảm bảo tính nhất quán
  - Cải thiện khả năng phục hồi khi thiếu các module không bắt buộc
  - Đảm bảo các tác vụ chính vẫn hoạt động ngay cả khi thiếu một số thành phần phụ
  - Tăng cường kiểm soát lỗi với thông báo chi tiết hơn

### Sửa lỗi
- Khắc phục lỗi "expected an indented block after 'if' statement" trong mco_engine.py
- Sửa lỗi import của create_eclipse_icon trong kbp_dialog.py
- Cải thiện xử lý lỗi khi một số module KBP hoặc MCO không khả dụng
- Đảm bảo Dialog KBP hoạt động đúng với các loại import khác nhau

## [0.9.1] - 2023-10-25

### Thêm mới
- Module Knowledge-Based Planning (KBP) theo phong cách Eclipse:
  - Dialog KBPDialog với giao diện tương tự RapidPlan của Eclipse
  - Nút KBP trong thanh công cụ External Beam Planning
  - Tích hợp dự đoán tham số tối ưu từ dữ liệu kế hoạch trước đó
  - Áp dụng tự động đề xuất vào kế hoạch hiện tại

### Cải tiến
- Nâng cấp module KBP cốt lõi:
  - Cấu trúc lớp KBPModel với mô hình Gradient Boosting
  - KBPPredictor cho dự đoán tham số tối ưu
  - KBPFeatureExtractor cho trích xuất đặc trưng từ dữ liệu
  - Đánh giá mô hình với hiển thị đặc trưng quan trọng

- Tăng cường tích hợp với quy trình làm việc:
  - Tích hợp liền mạch với quy trình lập kế hoạch ngược
  - Áp dụng tự động đề xuất vào danh sách mục tiêu và ràng buộc
  - Chuyển đổi chế độ lập kế hoạch thông minh
  - Tối ưu hóa tự động sau khi áp dụng đề xuất KBP

- Cải thiện trải nghiệm người dùng:
  - Biểu tượng chuyên nghiệp trong thanh công cụ
  - Hàm create_eclipse_icon cho giao diện nhất quán
  - Xử lý ngoại lệ toàn diện cho tính năng KBP
  - Thông báo trực quan với hướng dẫn rõ ràng

### Sửa lỗi
- Vấn đề tương thích với các phiên bản Qt khác nhau trong KBPDialog
- Lỗi hiển thị trọng số mục tiêu trong bảng đề xuất KBP
- Quá trình xử lý khi thiếu dữ liệu cấu trúc
- Đảm bảo tính khả dụng biểu tượng cho tất cả các nút

## [0.9.0] - 2023-10-20

### Thêm mới
- Hoàn thiện hệ thống import an toàn cho module dialogs:
  - Cơ chế import thông minh với try-except và lớp giả mạch
  - Thông báo lỗi hữu ích với hướng dẫn khắc phục
  - Khả năng hoạt động khi thiếu các thành phần không thiết yếu

### Cải tiến
- Tích hợp toàn diện các dialog phong cách Eclipse vào quy trình làm việc:
  - Kết nối PlanComparisonDialog với Object Explorer
  - Tự động hiển thị PlanPropertiesDialog khi đúp chuột vào kế hoạch
  - Tích hợp StructurePropertiesDialog với ColorButton tùy chỉnh
  - Giao diện nhất quán giữa các dialog và hệ thống chính

- Tối ưu hiệu năng cho các tính toán liều lớn:
  - Cải thiện thuật toán Monte Carlo cho hiệu suất cao hơn
  - Tính toán song song với hỗ trợ đa luồng và GPU
  - Quản lý bộ nhớ thông minh cho phân phối liều lớn
  - Hiển thị 3D hiệu suất cao cho kế hoạch phức tạp

- Tăng cường tích hợp MCO nâng cao:
  - Giao diện Pareto Navigator với hiển thị 3D hiệu quả
  - Điều hướng trực quan trên bề mặt Pareto
  - Cân bằng và phân tích độ nhạy cho các tiêu chí tối ưu
  - Lưu trữ và truy xuất giải pháp Pareto hiệu quả

### Sửa lỗi
- Khắc phục vấn đề tương thích khi import các dialog
- Sửa lỗi hiển thị ColorButton trên các hệ điều hành khác nhau
- Cải thiện xử lý ngoại lệ toàn diện trong hệ thống
- Đảm bảo tính nhất quán giữa các module
- Sửa lỗi tràn bộ nhớ khi tính toán liều cho kế hoạch lớn
- Khắc phục hiển thị DVH trong PlanComparisonDialog

## [0.8.16] - 2023-10-15

### Thêm mới
- Hoàn thiện hệ thống dialog phong cách Eclipse:
  - PlanComparisonDialog để so sánh nhiều kế hoạch với giao diện trực quan
  - PlanPropertiesDialog để chỉnh sửa thuộc tính kế hoạch xạ trị
  - StructurePropertiesDialog với ColorButton tùy chỉnh cho cấu trúc
  - Tích hợp đầy đủ các dialog vào quy trình làm việc

### Cải tiến
- Nâng cao khả năng phục hồi khi thiếu thành phần:
  - Thêm lớp giả mạch cho tất cả các thành phần có thể thiếu
  - Xử lý ngoại lệ toàn diện khi import các module không khả dụng
  - Cải thiện thông báo lỗi với hướng dẫn khắc phục chi tiết
  - Đảm bảo hoạt động cơ bản ngay cả khi thiếu thành phần nâng cao

- Cải thiện tích hợp giữa các thành phần:
  - Đồng bộ hóa giữa Object Explorer và các dialog thuộc tính
  - Kết nối liền mạch giữa PlanComparisonDialog và DVH Widget
  - Tích hợp các dialog thuộc tính vào quy trình làm việc chính
  - Đảm bảo nhất quán dữ liệu khi chỉnh sửa thuộc tính đối tượng

### Sửa lỗi
- Khắc phục lỗi import các dialog trong object_explorer_panel.py
- Sửa lỗi khi hiển thị dialog thuộc tính với dữ liệu không hợp lệ
- Cải thiện xử lý ngoại lệ khi tạo và sử dụng các dialog
- Đảm bảo tính tương thích giữa các phiên bản PyQt khác nhau
- Khắc phục lỗi hiển thị màu sắc trong StructurePropertiesDialog

## [0.8.15] - 2023-10-10

### Thêm mới
- Tái cấu trúc hệ thống tab phong cách Eclipse:
  - Thiết kế lại bố cục tab thống nhất với Eclipse
  - Tích hợp tab Plan Evaluation Report và Robust Analysis
  - Thêm tab MCO Navigator ẩn (hiển thị khi cần)
  - Cải thiện cơ chế chuyển đổi giữa các tab

### Cải tiến
- Hoàn thiện tích hợp ObjectExplorerPanel vào MainWindow:
  - Cải thiện kết nối giữa ObjectExplorerPanel và các thành phần chính
  - Thêm xử lý ngoại lệ cho kết nối signal/slot
  - Thiết kế lại LeftPanel với splitter điều chỉnh kích thước
  - Thêm tiêu đề và giao diện trực quan cho Object Explorer

- Nâng cao phương thức calculate_dose:
  - Cải thiện xử lý ngoại lệ toàn diện cho quá trình tính toán liều
  - Kết nối thông minh với signal progress_updated của DoseCalculator
  - Thêm thông báo tiến trình và kết quả trong status bar
  - Đảm bảo giải phóng tài nguyên sau khi tính toán hoàn tất

- Cải thiện đồng bộ hóa dữ liệu:
  - Tự động cập nhật Object Explorer khi tải kế hoạch và cấu trúc mới
  - Đồng bộ giữa các tab và Object Explorer khi dữ liệu thay đổi
  - Cải thiện logging và báo cáo lỗi trong quá trình đồng bộ
  - Bổ sung thông báo trạng thái rõ ràng trong status bar

### Sửa lỗi
- Khắc phục lỗi "No name 'QApplication' in module 'PyQt5.QtWidgets'" trong object_explorer_panel.py
- Sửa nhiều lỗi import trong main_window.py với hệ thống import thông minh
- Khắc phục lỗi "Access to member 'current_beam_set' before its definition"
- Sửa lỗi import module dialogs với cơ chế fallback tự động
- Đảm bảo kết nối signal/slot được bao bọc bởi try-except

## [0.8.14] - 2023-10-01

### Thêm mới
- Tạo Object Explorer Panel hoàn chỉnh theo phong cách Eclipse:
  - Hiển thị và quản lý bệnh nhân, cấu trúc và kế hoạch xạ trị trong một giao diện hợp nhất
  - Hỗ trợ tìm kiếm đối tượng với bộ lọc thời gian thực
  - Menu ngữ cảnh đầy đủ cho các thao tác phổ biến
  - Chức năng thêm/xóa/sửa đối tượng trực tiếp từ panel

### Cải tiến
- Tích hợp Object Explorer Panel vào hệ thống chính:
  - Đồng bộ hóa giữa Object Explorer Panel và các tab khác
  - Tương tác hai chiều với các module
  - Hiển thị/ẩn cấu trúc trực tiếp từ panel
  - Quản lý bệnh nhân và kế hoạch toàn diện

- Nâng cao độ tin cậy của hệ thống:
  - Sửa nhiều lỗi linter trong main_window.py
  - Cải thiện xử lý ngoại lệ cho các thành phần không khả dụng
  - Bổ sung cơ chế dự phòng khi phương thức không tồn tại
  - Xử lý các trường hợp ngoại lệ đầy đủ

- Nâng cao trải nghiệm người dùng:
  - Panel trái với thanh phân chia điều chỉnh kích thước
  - Đồng bộ tự động khi kế hoạch, cấu trúc hoặc liều cập nhật
  - Giao diện người dùng nhất quán với phong cách Eclipse
  - Thanh trạng thái cải tiến với hiển thị tiến trình

### Sửa lỗi
- Khắc phục lỗi "Instance of 'Plan' has no 'add_beam_set' member"
- Khắc phục lỗi "Instance of 'DoseCalculator' has no 'progress_updated' member"
- Sửa lỗi "Too many positional arguments for constructor call" trong PlanComparisonDialog
- Sửa lỗi current_beam_set không tồn tại trong calculate_dose()
- Cải thiện xử lý MPR Viewer không khả dụng
- Đảm bảo tính nhất quán khi chuyển đổi giữa các tab và đối tượng

## [0.8.13] - 2023-09-25

### Thêm mới
- Tích hợp đầy đủ phân tích độ bền vững (Robustness Analysis) vào hệ thống:
  - Thêm tab phân tích độ bền vững vào giao diện chính
  - Cải thiện DVHWidget để hỗ trợ hiển thị dải DVH (DVH bands) từ kết quả phân tích

### Cải tiến
- Nâng cao trải nghiệm người dùng trong phân tích độ bền vững:
  - Đồng bộ hóa hiển thị kết quả giữa các tab
  - Hiển thị trực quan kết quả phân tích với màu sắc và độ trong suốt
  - Xử lý ngoại lệ toàn diện khi phân tích độ bền vững

- Cải thiện giao diện DVH widget:
  - Thiết kế lại giao diện với bố cục rõ ràng và dễ sử dụng hơn
  - Tổ chức danh sách cấu trúc trong một scroll area duy nhất
  - Xử lý ngoại lệ khi các thư viện bên ngoài không khả dụng

### Sửa lỗi
- Khắc phục lỗi khi tích hợp tab phân tích độ bền vững vào MainWindow
- Sửa lỗi trong DVHWidget khi hiển thị dữ liệu từ nhiều kế hoạch
- Cải thiện xử lý ngoại lệ khi các thư viện bên ngoài không khả dụng
- Đảm bảo tính nhất quán khi chuyển đổi giữa các tab

## [0.8.12] - 2023-09-20

### Thêm mới
- Nâng cấp tích hợp MPR viewer trong Structure tab:
  - Khắc phục các lỗi linter trong structure_tab.py với forward declaration
  - Cải thiện phương thức `setup_slice_view()` với xử lý ngoại lệ toàn diện
  - Tối ưu hiển thị cấu trúc với phương thức update_structure_overlay() được cải tiến

### Cải tiến
- Nâng cao xử lý sự kiện chuột để vẽ contour trực tiếp trên MPR viewer:
  - Thêm phương thức `_get_active_drawing_tool()` để chọn công cụ vẽ phù hợp
  - Đồng bộ hiển thị cấu trúc giữa danh sách và MPR viewer
  - Cải thiện hoạt động của các công cụ vẽ, xóa, cọ và smart brush

- Cải thiện quản lý contour:
  - Cập nhật `ContourManager` với phương thức undo() và redo() hoạt động đúng
  - Lưu trữ hoạt động chỉnh sửa contour cho khả năng hoàn tác/làm lại
  - Cải thiện hiệu suất khi cập nhật nhiều contour cùng lúc

### Sửa lỗi
- Khắc phục các lỗi linter trong structure_tab.py:
  - Sửa lỗi "Using variable X before assignment" cho các lớp cơ bản
  - Sửa lỗi "Instance of 'ContourManager' has no 'undo/redo' member"
- Sửa lỗi không cập nhật hiển thị MPR khi toggle_structure_visibility()
- Cải thiện xử lý ngoại lệ khi làm việc với các cấu trúc không có dữ liệu contour

## [0.8.11] - 2023-09-15

### Thêm mới
- Nâng cấp MPR viewer trong Structure tab với hiển thị dữ liệu hình ảnh thực:
  - Tích hợp đầy đủ MPRViewer với hiển thị đa mặt phẳng (axial, sagittal, coronal)
  - Thanh công cụ vẽ contour với các tùy chọn: vẽ, xóa, cọ và smart brush
  - Hỗ trợ hoàn tác/làm lại các thao tác vẽ contour
  - Hiển thị và tương tác với cấu trúc giải phẫu trên các mặt phẳng MPR

### Cải tiến
- Nâng cao trải nghiệm người dùng khi phân đoạn cấu trúc:
  - Xử lý sự kiện chuột để vẽ contour trực tiếp trên MPR viewer
  - Đồng bộ hóa hiển thị giữa danh sách cấu trúc và MPR viewer
  - Cập nhật tự động hiển thị khi chỉnh sửa hoặc thay đổi lựa chọn cấu trúc
  - Hiển thị trực quan trạng thái "đã chọn" cho cấu trúc trong MPR viewer

### Sửa lỗi
- Khắc phục lỗi không hiển thị cấu trúc trong MPR viewer khi thay đổi lựa chọn
- Xử lý các trường hợp ngoại lệ khi dữ liệu hình ảnh không hợp lệ
- Cải thiện xử lý ngoại lệ khi các module phụ thuộc không khả dụng
- Đảm bảo làm sạch tài nguyên khi đóng Structure tab

## [0.8.10] - 2023-09-10

### Thêm mới
- Hoàn thiện tính năng phân đoạn tự động trong Structure tab:
  - Tích hợp đầy đủ AutoSegmentationEngine vào phương thức `auto_segment()`
  - Giao diện người dùng cho lựa chọn và thiết lập tham số phân đoạn
  - Hiển thị tiến trình thực hiện với ProgressDialog và ProgressBar
  - Tích hợp kết quả phân đoạn vào quy trình lập kế hoạch

### Cải tiến
- Tăng cường tính ổn định cho module MCO:
  - Khắc phục lỗi trong hàm plot_pareto_front_3d của mco_module.py
  - Thêm xử lý ngoại lệ cho colormap khi matplotlib.cm.viridis không khả dụng
  - Giảm thiểu rủi ro lỗi runtime khi hiển thị bề mặt Pareto 3D
  - Đảm bảo khả năng sử dụng trên nhiều môi trường khác nhau

### Sửa lỗi
- Sửa lỗi trong mco_module.py khi biến cmap và norm được sử dụng trước khi khởi tạo
- Cải thiện xử lý ngoại lệ khi thiếu thư viện PyQt5 hoặc VTK
- Khắc phục vấn đề cập nhật giao diện người dùng sau quá trình phân đoạn tự động

## [0.8.9] - 2023-09-05

### Thêm mới
- Tạo module báo cáo đánh giá kế hoạch trong `quangtps/reporting/plan_report_generator.py`:
  - Hỗ trợ tạo báo cáo đánh giá kế hoạch xạ trị với nhiều định dạng xuất (PDF, HTML, CSV)
  - Báo cáo chứa thông tin kế hoạch, DVH, kết quả đánh giá mục tiêu và chỉ số nâng cao
  - Hỗ trợ so sánh nhiều kế hoạch trong cùng báo cáo

- Định nghĩa chuẩn cho các loại mục tiêu lâm sàng trong `quangtps/evaluation/protocols/goal_type.py`:
  - Các loại mục tiêu cơ bản như DOSE_VOLUME, VOLUME_DOSE
  - Chỉ số thống kê liều như MEAN_DOSE, MAX_DOSE, MIN_DOSE
  - Chỉ số đánh giá nâng cao như HOMOGENEITY_INDEX, CONFORMITY_INDEX, GRADIENT_INDEX
  - Chỉ số đánh giá sinh học như TCP, NTCP, EUD

- Tạo hệ thống điểm đánh giá kế hoạch trong `quangtps/evaluation/plan_quality/plan_quality_score.py`:
  - Các mức đánh giá EXCELLENT, GOOD, ACCEPTABLE, MARGINAL, POOR
  - Phương thức chuyển đổi từ phần trăm đạt mục tiêu sang mức đánh giá
  - Ánh xạ màu sắc trực quan cho từng mức đánh giá

### Cải tiến
- Phát triển bộ đánh giá kế hoạch xạ trị toàn diện:
  - Lớp PlanQualityEvaluator đánh giá kế hoạch theo protocol lâm sàng
  - Tính toán các chỉ số đánh giá cho mục tiêu và cơ quan nguy cấp riêng biệt
  - Hỗ trợ xuất kết quả đánh giá sang nhiều định dạng để phân tích
  - Tính toán chỉ số đồng nhất và phù hợp cho đánh giá nâng cao

- Tăng cường xử lý ngoại lệ và tính ổn định:
  - Cơ chế dự phòng khi các module không khả dụng
  - Xử lý lỗi import thông minh với cơ chế fallback
  - Báo cáo lỗi chi tiết để dễ dàng gỡ lỗi
  - Tăng cường khả năng phục hồi khi thiếu các thư viện bên ngoài

### Sửa lỗi
- Khắc phục các lỗi trong module đánh giá kế hoạch:
  - Sửa lỗi tham chiếu đến các thành viên enum không tồn tại
  - Sửa lỗi import module báo cáo kế hoạch
  - Cải thiện xử lý khi không có DVHAnalyzer

## [0.8.8] - 2023-08-30

### Thêm mới
- Tạo module đánh giá kế hoạch xạ trị toàn diện trong `quangtps/ui/plan_evaluation_report_tab.py`:
  - Giao diện phong cách Eclipse cho đánh giá kế hoạch xạ trị chuyên nghiệp
  - Báo cáo đánh giá kế hoạch với khả năng xuất PDF, HTML và CSV
  - Hiển thị tích hợp của DVH, đánh giá mục tiêu lâm sàng và thống kê liều
  - Tính năng so sánh kế hoạch để phân tích nhiều kế hoạch cùng lúc

### Cải tiến
- Nâng cao trải nghiệm đánh giá kế hoạch:
  - Hiển thị điểm đánh giá tổng thể, PTV và OAR với mã màu trực quan
  - Quản lý và chỉnh sửa protocol lâm sàng ngay trong giao diện
  - Đánh giá kế hoạch tự động với kết quả màu sắc trực quan
  - Tích hợp chọn cấu trúc từ bảng mục tiêu với hiển thị DVH
  - Bố cục thông minh với splitter có thể tùy chỉnh giữa các panel

- Tích hợp với các module hiện có:
  - Kết nối liền mạch với DVHWidget, ProtocolManager và Clinical Goals
  - Tận dụng chức năng của PlanQualityWidget với giao diện mới
  - Đảm bảo tương thích với cả hệ thống Eclipse-style theme
  - Xử lý ngoại lệ toàn diện khi các thành phần không khả dụng

## [0.8.7] - 2023-08-25

### Thêm mới
- Tích hợp hoàn chỉnh hiển thị liều 3D vào External Beam Planning tab:
  - Kết nối tự động giữa dữ liệu liều, cấu trúc giải phẫu và hiển thị 3D
  - Cơ chế tự động chuyển đổi dữ liệu liều từ nhiều định dạng khác nhau
  - Hiển thị đồng bộ liều 3D và cấu trúc giải phẫu trong một khung nhìn thống nhất

### Cải tiến
- Thiết kế lại giao diện External Beam Planning theo phong cách Eclipse:
  - Layout tối ưu với splitter có thể điều chỉnh giữa các panel
  - Chuyển đổi từ combo box sang radio button cho chọn chế độ lập kế hoạch
  - Tổ chức các tính năng thành các tab con logic và trực quan hơn
  - Thêm status bar cung cấp thông tin trạng thái và tiến trình

- Cải thiện tích hợp hiển thị liều 3D:
  - Thay thế placeholder bằng DoseVisualization3D đầy đủ chức năng
  - Tối ưu hóa hiển thị isodose surface với các mức liều có thể tùy chỉnh
  - Hiển thị cấu trúc 3D với độ trong suốt và màu sắc có thể tùy chỉnh
  - Đồng bộ hóa hiển thị khi dữ liệu liều hoặc cấu trúc thay đổi

- Nâng cao độ tin cậy của hệ thống:
  - Xử lý lỗi toàn diện khi các thư viện bên ngoài không khả dụng
  - Placeholder thông minh cho các thành phần gặp lỗi
  - Chế độ dự phòng cho các tính năng khi thiếu VTK hoặc PyQt5
  - Logging chi tiết để hỗ trợ gỡ lỗi và theo dõi

## [0.8.6] - 2023-08-20

### Thêm mới
- Hoàn thiện module phân tích độ bền vững trong `quangtps/evaluation/robustness`:
  - Cập nhật constructor của `RobustnessAnalyzer` với các tham số đầy đủ:
    - `setup_uncertainty` - độ không chắc chắn vị trí bệnh nhân (mm)
    - `range_uncertainty` - độ không chắc chắn phạm vi (%)
    - `num_scenarios` - số lượng kịch bản phân tích
  - Bổ sung các phương thức cốt lõi cho `RobustnessResult`:
    - `get_structure_dvhs()` - lấy dữ liệu DVH cho từng cấu trúc
    - `get_target_coverage_data()` - lấy dữ liệu độ phủ mục tiêu
    - `get_evaluation_metrics()` - lấy chỉ số đánh giá toàn diện
    - `get_spatial_analysis_data()` - lấy dữ liệu phân tích không gian

### Cải tiến
- Thêm các tính năng xuất kết quả phân tích độ bền vững:
  - Xuất báo cáo định dạng CSV và Excel với các sheet phân tích riêng biệt
  - Tạo báo cáo PDF và HTML tương tác với biểu đồ và bảng phân tích
  - Tạo DVH bands trực quan với độ dao động giữa các kịch bản khác nhau
- Cải thiện phân tích dữ liệu cho mục tiêu và OAR:
  - Tính toán các thông số D95, Dmax và biến thiên theo kịch bản
  - Đánh giá tự động các chỉ số biến thiên và độ phù hợp lâm sàng
  - Phân tích không gian với bản đồ nhiệt sự khác biệt liều
- Tích hợp phong cách Eclipse vào module phân tích độ bền vững:
  - Thiết kế giao diện theo phong cách Eclipse hiện đại
  - Đảm bảo tương thích giữa UI và module phân tích cốt lõi
  - Kết nối các thành phần: `robust_analysis_tab.py`, `robust_analysis_widget.py` và `robustness_visualization.py`

## [0.8.5] - 2023-08-15

### Cải tiến
- Sửa lỗi indent trong module phân tích gamma (quangtps/evaluation/metrics/gamma_analysis.py)
- Tạo module phân tích độ bền vững (Robustness Analysis) phong cách Eclipse
- Cải thiện trải nghiệm phân tích độ bền vững
- Bổ sung tính năng tương thích

[0.8.11]: https://github.com/username/QuangTPS/compare/v0.8.10...v0.8.11
[0.8.10]: https://github.com/username/QuangTPS/compare/v0.8.9...v0.8.10
[0.8.9]: https://github.com/username/QuangTPS/compare/v0.8.8...v0.8.9
[0.8.8]: https://github.com/username/QuangTPS/compare/v0.8.7...v0.8.8
[0.8.7]: https://github.com/username/QuangTPS/compare/v0.8.6...v0.8.7
[0.8.6]: https://github.com/username/QuangTPS/compare/v0.8.5...v0.8.6
[0.8.5]: https://github.com/username/QuangTPS/compare/v0.8.4...v0.8.5