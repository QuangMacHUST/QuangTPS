# QuangTPS: Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở

## 🏥 Giới thiệu
QuangTPS là một hệ thống lập kế hoạch xạ trị mã nguồn mở toàn diện, được phát triển để cung cấp cho các chuyên gia y tế một công cụ mạnh mẽ, linh hoạt và sáng tạo trong điều trị ung thư. Lấy cảm hứng từ các hệ thống thương mại hàng đầu như RayStation, QuangTPS tích hợp các thuật toán tiên tiến và giao diện thân thiện với người dùng để tối ưu hóa quá trình lập kế hoạch xạ trị.

## ✨ Tính năng Cơ bản

### 📋 Quản lý Dữ liệu DICOM
- **Nhập/xuất và quản lý DICOM toàn diện**
  - Hỗ trợ đầy đủ các định dạng DICOM: CT, MRI, PET, CBCT, 4D-CT
  - Nhập/xuất chuẩn hóa RT Structure, RT Dose, RT Plan, RT Image
  - Chuyển đổi tự động giữa các định dạng khác nhau
  - Tích hợp với các hệ thống PACS bệnh viện
  - Lưu trữ dữ liệu DICOM theo cách có tổ chức với khả năng tìm kiếm mạnh mẽ

- **Quản lý bệnh nhân và kế hoạch**
  - Lưu trữ thông tin bệnh nhân toàn diện 
  - Quản lý nhiều kế hoạch cho mỗi bệnh nhân với khả năng phân loại
  - So sánh song song nhiều kế hoạch để lựa chọn tối ưu
  - Lưu trữ lịch sử điều trị và thông tin liên quan
  - Hệ thống khôi phục phiên làm việc tự động và sao lưu dữ liệu

### 🖼️ Hiển thị và Xử lý Hình ảnh
- **Hiển thị hình ảnh đa chiều và đa phương thức**
  - Hiển thị đa mặt phẳng (MPR): Axial, Sagittal, Coronal, Oblique
  - Rendering 3D với khả năng xoay, phóng to/thu nhỏ và cắt lớp
  - Chồng hình và tổng hợp dữ liệu đa phương thức (CT-MRI-PET fusion)
  - Điều chỉnh độ sáng/tương phản và cài đặt cửa sổ tùy chỉnh
  - Tái tạo cong (curved reconstruction) và mặt phẳng tùy chỉnh

- **Công cụ đo lường và phân tích hình ảnh**
  - Đo khoảng cách, góc, diện tích và thể tích trên hình ảnh
  - Phân tích và hiển thị giá trị HU (Hounsfield Unit)
  - Đo lường mật độ xương và mô
  - Tạo lập profile và histogram
  - Thay đổi độ trong suốt, màu sắc và cài đặt hiển thị

### 🔍 Phân đoạn và Contour
- **Công cụ vẽ và chỉnh sửa Contour chuyên nghiệp**
  - Đa dạng công cụ vẽ contour (brush, polygon, freehand, point)
  - Chỉnh sửa contour nâng cao (push/pull, smooth, interpolate, erase)
  - Tạo và quản lý nhiều contour sets cho cùng một bệnh nhân
  - Tạo cấu trúc từ phép toán Boolean (union, intersection, subtraction, exclusive OR)
  - Tạo margin tự động với khả năng kiểm soát chi tiết (expansion, contraction, ring)
  - Chức năng hỗ trợ độ ưu tiên và phân loại cấu trúc

- **Phân đoạn tự động với AI và thuật toán hiện đại**
  - Phân đoạn tự động các cơ quan nguy cấp (OAR) sử dụng mạng U-Net, Cycle-GANs
  - Phân đoạn tự động GTV/CTV/PTV dựa trên mô hình học sâu
  - Atlas-based segmentation với thư viện mẫu phong phú
  - Phân đoạn bán tự động với thuật toán threshold và region growing
  - Hỗ trợ tải và sử dụng nhiều mô hình đã huấn luyện
  - Tự động điều chỉnh và tinh chỉnh kết quả phân đoạn

## 🚀 Tính năng Nâng cao

### 📊 Tính toán và Mô phỏng Liều
- **Đa dạng thuật toán tính toán liều xạ trị**
  - Collapsed Cone Convolution (CCC)
  - Pencil Beam
  - Analytical Anisotropic Algorithm (AAA)
  - Acuros XB
  - Convolution Superposition
  - Monte Carlo
  - Grid-Based Boltzmann Solver (GBBS)
  - Tích hợp các thuật toán mở rộng thông qua plugin

- **Mô hình vật lý nâng cao**
  - Tính toán TERMA (Total Energy Released per unit MAss)
  - Mô hình hóa đầy đủ tương tác chùm tia-vật chất
  - Hiệu chỉnh không đồng nhất (heterogeneity correction)
  - Mô phỏng các hiệu ứng tán xạ và hấp thụ
  - Tính toán và hiển thị LET (Linear Energy Transfer) cho proton và ion
  - Điều chỉnh RBE (Relative Biological Effectiveness) cho liệu pháp ion

### 🎛️ Kỹ thuật Xạ trị Tiên tiến
- **Hỗ trợ đầy đủ các kỹ thuật xạ trị hiện đại**
  - 3D-CRT (Conformal Radiation Therapy)
  - IMRT (Intensity Modulated Radiation Therapy)
  - VMAT (Volumetric Modulated Arc Therapy)
  - DCAT (Dynamic Conformal Arc Therapy)
  - SRS/SBRT (Stereotactic Radiosurgery/Stereotactic Body Radiation Therapy)
  - TBI/TSI (Total Body/Total Skin Irradiation)
  - Proton Therapy (PBS - Pencil Beam Scanning)
  - Carbon Ion Therapy
  - Adaptive Radiation Therapy
  - Electron Therapy
  - Image-Guided Radiotherapy
  - FLASH Radiotherapy
  - Boron Neutron Capture Therapy (BNCT)

- **Quản lý MLC (Multi-Leaf Collimator) chi tiết**
  - Hiển thị và điều chỉnh MLC trong Beam's Eye View (BEV)
  - Tự động điều chỉnh lá MLC dựa trên hình dạng cấu trúc
  - Mô phỏng chuyển động của MLC trong kỹ thuật VMAT và IMRT
  - Hỗ trợ nhiều loại MLC của các nhà sản xuất khác nhau
  - Tối ưu hóa tuần tự lá MLC

### 🧩 Tối ưu hóa Kế hoạch
- **Tối ưu hóa liều xạ trị đa dạng**
  - Tối ưu hóa đa tiêu chí (Multi-Criteria Optimization - MCO)
  - Tối ưu hóa dựa trên ràng buộc (Constraint-based optimization)
  - Tối ưu hóa dựa trên mục tiêu (Objective-based optimization)
  - Tối ưu hóa mạnh mẽ (Robust optimization) cho proton và ion
  - Tối ưu hóa liều sinh học (Biological optimization)
  - Tự động điều chỉnh trọng số tối ưu
  - Tối ưu hóa dự trữ liều (Fall-off)

- **Lập kế hoạch dựa trên kiến thức (KBP - Knowledge-Based Planning)**
  - Dự đoán DVH tối ưu dựa trên dữ liệu lịch sử
  - Tạo kế hoạch mới dựa trên kế hoạch tương tự
  - Học máy để tự động cải thiện các mục tiêu tối ưu
  - Áp dụng các kinh nghiệm từ kế hoạch trước đó
  - Tối ưu hóa dựa trên mô hình dự đoán

### 📈 Đánh giá Kế hoạch
- **Công cụ phân tích kế hoạch toàn diện**
  - Biểu đồ Liều-Thể tích (DVH) tương tác với nhiều tùy chọn
  - Phân tích thống kê liều với chỉ số Dmin, Dmax, Dmean, V95, D98, D50, D2
  - Hiển thị phân bố liều 3D với khả năng tùy chỉnh
  - Phân tích lát cắt liều (dose slicing) theo mặt phẳng tùy chỉnh
  - So sánh trực quan nhiều kế hoạch cùng lúc

- **Chỉ số đánh giá kế hoạch đầy đủ**
  - Chỉ số đồng dạng (Conformity Index - CI)
  - Chỉ số đồng nhất (Homogeneity Index - HI)
  - Chỉ số gradient (Gradient Index - GI)
  - Chỉ số Paddick
  - Phân tích hotspot và coldspot
  - Tính toán integral dose
  - Mô hình TCP (Tumor Control Probability)
  - Mô hình NTCP (Normal Tissue Complication Probability)
  - Tính toán BED (Biologically Effective Dose) và EQD2

### 🔄 Thích ứng và Phân tích Thời gian
- **Liệu pháp thích ứng (Adaptive Therapy)**
  - Biến dạng hình ảnh linh hoạt (Deformable Image Registration)
  - Tổng và ánh xạ liều giữa các phân số (dose accumulation)
  - Đánh giá và điều chỉnh kế hoạch dựa trên hình ảnh mới
  - Mô phỏng phân phối liều trên hình ảnh cone-beam CT
  - Phân tích sai số thiết lập và đánh giá độ bám sát kế hoạch

- **Phân tích thời gian và liều 4D**
  - Quản lý và xử lý dữ liệu 4D-CT
  - Theo dõi chuyển động nội tạng
  - Tính toán liều tích lũy theo thời gian
  - Tối ưu hóa có tính đến chuyển động (4D optimization)
  - Trình diễn động (dynamic display) của phân phối liều

### 🛡️ Đảm bảo Chất lượng và An toàn
- **Công cụ QA (Quality Assurance) tích hợp**
  - Tạo kế hoạch QA tự động
  - So sánh liều tính toán và đo đạc
  - Phân tích Gamma Index
  - Kiểm tra va chạm ảo (Virtual collision check)
  - Mô phỏng các thiết bị phantom QA

- **Tính năng an toàn và kiểm tra**
  - Kiểm tra tính đúng đắn của dữ liệu bệnh nhân
  - Cảnh báo và ngăn chặn các thông số không an toàn
  - Kiểm tra giới hạn liều cho các cơ quan nguy cấp
  - Hệ thống xác minh trước điều trị
  - Ghi nhật ký đầy đủ các thao tác và thay đổi

### 📑 Báo cáo và Xuất dữ liệu
- **Hệ thống báo cáo linh hoạt**
  - Tạo báo cáo kế hoạch điều trị tự động với thông tin đầy đủ
  - Tùy chỉnh mẫu báo cáo theo yêu cầu của cơ sở y tế
  - Báo cáo QA chi tiết
  - Báo cáo so sánh kế hoạch
  - Báo cáo kiểm tra va chạm
  - Xuất báo cáo sang nhiều định dạng (PDF, DOCX, HTML)

- **Xuất dữ liệu đa dạng**
  - Xuất DICOM RT theo chuẩn
  - Xuất dữ liệu DVH dạng CSV, Excel
  - Xuất hình ảnh và đồ thị dạng PNG, JPEG, SVG
  - Xuất dữ liệu phân tích cho xử lý nâng cao
  - Tích hợp với các hệ thống R&V (Record and Verify)

## 🧠 Module Chuyên biệt

### 👨‍⚕️ Micro-TPS
- **Lập kế hoạch tiền lâm sàng**
  - Hỗ trợ dữ liệu vi mô và động vật nhỏ
  - Tích hợp với thiết bị xạ trị tiền lâm sàng
  - Mô phỏng liều chính xác cho nghiên cứu
  - Hỗ trợ nhiều mô hình liều sinh học

### 💉 Brachytherapy
- **Lập kế hoạch xạ trị áp sát**
  - Hỗ trợ HDR (High Dose Rate) và LDR (Low Dose Rate)
  - Tối ưu hóa vị trí và thời gian dừng nguồn
  - Tính toán liều theo TG-43 và mô hình tiên tiến
  - Kết nối với các hệ thống brachytherapy phổ biến

### 🎯 BNCT (Boron Neutron Capture Therapy)
- **Mô phỏng và lập kế hoạch BNCT**
  - Mô hình hóa phản ứng bắt neutron
  - Tối ưu hóa phân bố boron
  - Phân tích RBE phức tạp
  - Tích hợp với các trung tâm BNCT

### 🔬 Nghiên cứu và Phát triển
- **Nền tảng mở rộng cho nghiên cứu**
  - API mở cho Python và C++
  - Môi trường phát triển plugin
  - Framework phát triển mô hình AI
  - Công cụ phân tích và đánh giá thuật toán mới

## 🖥️ Yêu cầu Hệ thống
- **Hệ điều hành**:
  - Windows 10/11 64-bit
  - macOS 11.0+
  - Linux (Ubuntu 20.04+, CentOS 8+)

- **Phần cứng tối thiểu**:
  - CPU: 4 nhân, 2.5GHz trở lên
  - RAM: 8GB (16GB+ khuyến nghị cho dữ liệu lớn)
  - GPU: NVIDIA GPU với CUDA hỗ trợ (GTX 1060 6GB trở lên)
  - Ổ cứng: 20GB không gian trống (SSD khuyến nghị)
  - Màn hình: Độ phân giải 1920x1080 trở lên

- **Phần mềm**:
  - Python 3.8 trở lên
  - CUDA Toolkit 11.0+ (cho tính năng AI và Monte Carlo)
  - Thư viện khoa học (NumPy, SciPy, PyTorch/TensorFlow)
  - OpenGL 4.0+

## 📝 Giấy phép và Đóng góp
- **Giấy phép**: GPL v3
- **Đóng góp**: Xem hướng dẫn đóng góp trong tài liệu phát triển
- **Báo cáo lỗi**: Sử dụng hệ thống issue tracker trên GitHub

## 👥 Liên hệ và Hỗ trợ
- GitHub: [https://github.com/yourusername/QuangTPS](https://github.com/yourusername/QuangTPS)
- Tài liệu: [https://quangtps.readthedocs.io](https://quangtps.readthedocs.io)
- Diễn đàn: [https://forum.quangtps.org](https://forum.quangtps.org)
- Email: support@quangtps.org

---

*QuangTPS được phát triển với mục tiêu cung cấp một giải pháp mã nguồn mở chất lượng cao cho cộng đồng xạ trị ung thư. Mọi đóng góp và phản hồi đều được đánh giá cao và sẽ giúp cải thiện hệ thống này.*

## Cấu trúc của dự án
QuangTPS/
├── data/ - Data storage directory
│   ├── beam_data/ - Beam data from linear accelerators
│   ├── clinical_protocols/ - Clinical protocol templates
│   ├── database/ - Local database files
│   ├── dicom/ - DICOM file storage
│   ├── images/ - Image resources
│   ├── machine_data/ - Treatment machine configurations
│   ├── models/ - AI/ML models for segmentation, etc.
│   └── templates/ - Planning templates
├── quangtps/ - Main application code
│   ├── adaptive/ - Adaptive planning modules
│   │   ├── deformation/ - Image deformation tools
│   │   ├── dose_accumulation.py - Dose accumulation algorithms
│   │   ├── four_d.py - 4D planning support
│   │   ├── setup_error.py - Setup error analysis
│   │   └── temporal_analysis.py - Temporal analysis tools
│   ├── api/ - API interfaces
│   │   ├── cpp_interface.py - C++ bindings for performance-critical operations
│   │   ├── plugin_interface.py - Plugin system for extensions
│   │   ├── python_api.py - Python API for scripting
│   │   └── rest_api.py - REST API for external systems
│   ├── common/ - Common utilities
│   │   ├── paths.py - File path management
│   │   └── widgets.py - Common UI widgets
│   ├── core/ - Core system components
│   │   ├── beam_types.py - Beam type definitions
│   │   ├── config.py - System configuration
│   │   ├── constants.py - System constants
│   │   ├── exceptions.py - Custom exceptions
│   │   ├── logging.py - Logging system
│   │   ├── patient/ - Patient data models
│   │   ├── types.py - Type definitions
│   │   └── utils.py - Utility functions
│   ├── database/ - Database management
│   │   ├── beam_db.py - Beam database operations
│   │   ├── db_connector.py - Database connection management
│   │   ├── dose_db.py - Dose database operations
│   │   ├── image_db.py - Image database operations
│   │   ├── patient_db.py - Patient database operations
│   │   ├── plan_db.py - Plan database operations
│   │   ├── prescription_db.py - Prescription database operations
│   │   ├── query.py - Database query utilities
│   │   ├── series_db.py - Series database operations
│   │   ├── structure_db.py - Structure database operations
│   │   └── study_db.py - Study database operations
│   ├── dicom/ - DICOM handling
│   │   ├── cbct_processor.py - CBCT image processing
│   │   ├── ct4d_manager.py - 4D-CT management
│   │   ├── dicom_anonymizer.py - DICOM data anonymization
│   │   ├── dicom_converter.py - Format conversion
│   │   ├── dicom_dataset_manager.py - DICOM dataset handling
│   │   ├── dicom_exporter.py - DICOM export functions
│   │   ├── dicom_factory.py - DICOM object creation
│   │   ├── dicom_fusion.py - Image fusion tools
│   │   ├── dicom_importer.py - DICOM import functions
│   │   ├── dicom_reader.py - DICOM file reading
│   │   ├── dicom_sequence_manager.py - DICOM sequence handling
│   │   ├── dicom_utils.py - DICOM utilities
│   │   ├── dicom_validator.py - DICOM validation
│   │   ├── dicom_writer.py - DICOM file writing
│   │   ├── four_d_ct_processor.py - 4D-CT processing
│   │   ├── pacs_client.py - PACS connectivity
│   │   ├── pet_processor.py - PET image processing
│   │   ├── rt_dose.py - RT-Dose handling
│   │   ├── rt_image.py - RT-Image handling
│   │   ├── rt_plan.py - RT-Plan handling
│   │   └── rt_structure.py - RT-Structure handling
│   ├── dose/ - Dose calculation
│   │   ├── algorithms/ - Dose calculation algorithms
│   │   ├── base.py - Base classes for dose calculation
│   │   ├── beam_data_processor.py - Beam data processing
│   │   ├── dose_calculation.py - Dose calculation framework
│   │   ├── dose_calculator.py - Dose calculation manager
│   │   ├── dose_engine.py - Dose calculation engine
│   │   ├── dose_grid.py - Dose grid management
│   │   ├── dose_visualization.py - Dose visualization tools
│   │   └── physics/ - Physics models
│   ├── evaluation/ - Plan evaluation
│   │   ├── biological/ - Biological evaluation metrics
│   │   ├── dose_analysis.py - Dose analysis tools
│   │   ├── dvh/ - DVH calculation and analysis
│   │   ├── evaluation_report.py - Evaluation reporting
│   │   ├── metrics/ - Evaluation metrics
│   │   ├── plan_comparison.py - Plan comparison tools
│   │   └── qa/ - Quality assurance tools
│   ├── imaging/ - Image handling
│   │   ├── contour.py - Contour tools
│   │   ├── fusion.py - Image fusion
│   │   ├── image.py - Image processing
│   │   ├── image_processing.py - Image processing functions
│   │   ├── image_viewer.py - Image viewer core
│   │   ├── integrated_viewer.py - Integrated image viewing
│   │   ├── measurement.py - Image measurement tools
│   │   ├── mpr_viewer.py - Multi-planar reconstruction
│   │   ├── structures.py - Structure handling
│   │   ├── visualization.py - Visualization utilities
│   │   └── volume_renderer.py - 3D volume rendering
│   ├── optimization/ - Plan optimization
│   │   ├── constraints.py - Optimization constraints
│   │   ├── kbp/ - Knowledge-based planning
│   │   ├── methods/ - Optimization algorithms
│   │   ├── objectives.py - Optimization objectives
│   │   ├── optimization_engine.py - Optimization engine
│   │   └── solver.py - Optimization solver
│   ├── planning/ - Treatment planning
│   │   ├── beam.py - Beam definition and handling
│   │   ├── beam_configurator.py - Beam configuration tools
│   │   ├── beam_data_config.py - Beam data configuration
│   │   ├── comparison.py - Plan comparison
│   │   ├── dose_visualization.py - Dose visualization
│   │   ├── evaluation.py - Plan evaluation
│   │   ├── mlc.py - MLC handling
│   │   ├── optimization.py - Plan optimization
│   │   ├── plan.py - Plan definition
│   │   ├── prescription.py - Prescription management
│   │   ├── template_manager.py - Template management
│   │   ├── templates.py - Planning templates
│   │   └── treatment_planner.py - Treatment planning tools
│   ├── reporting/ - Reporting and export
│   │   ├── data_export.py - Data export utilities
│   │   ├── dicom_export.py - DICOM export utilities
│   │   ├── excel_export.py - Excel export utilities
│   │   ├── integration.py - Third-party integration
│   │   ├── report_generator.py - Report generation
│   │   └── report_templates.py - Report templates
│   ├── scripts/ - System scripts
│   │   ├── batch_processing.py - Batch processing tools
│   │   ├── data_conversion.py - Data conversion tools
│   │   └── system_check.py - System check utilities
│   ├── segmentation/ - Image segmentation
│   │   ├── auto/ - Auto-segmentation
│   │   ├── auto_segmentation/ - Auto-segmentation tools
│   │   ├── bridges/ - Segmentation integration bridges
│   │   ├── contour/ - Contour tools
│   │   ├── manual_segmentation/ - Manual segmentation tools
│   │   ├── structures/ - Structure handling
│   │   └── validation/ - Segmentation validation
│   ├── treatment/ - Treatment delivery
│   │   ├── beams/ - Beam delivery
│   │   ├── fractionation.py - Fractionation schemes
│   │   ├── machine/ - Treatment machine models
│   │   ├── mlc/ - MLC delivery models
│   │   ├── plan.py - Treatment plan implementation
│   │   ├── scheduler.py - Treatment scheduling
│   │   ├── techniques/ - Treatment techniques
│   │   ├── treatment_delivery.py - Treatment delivery simulation
│   │   ├── treatment_manager.py - Treatment management
│   │   └── treatment_technique_selector.py - Technique selection
│   └── ui/ - User interface
│       ├── auto_segmentation_tool.py - Auto-segmentation interface
│       ├── base_contour_tool.py - Base contour tools
│       ├── bnct_widget.py - BNCT planning widget
│       ├── dialogs/ - UI dialogs
│       ├── dicom_loader.py - DICOM loading interface
│       ├── dose_calculation_dialog.py - Dose calculation interface
│       ├── dose_tab.py - Dose calculation tab
│       ├── freehand_contour_tool.py - Freehand contouring
│       ├── geometric_contour_tool.py - Geometric contouring
│       ├── icons/ - UI icons
│       ├── image_display.py - Image display
│       ├── image_viewer.py - Image viewer interface
│       ├── imaging_tab.py - Imaging tab
│       ├── main_window.py - Main application window
│       ├── patient_browser.py - Patient browser
│       ├── patient_tab.py - Patient management tab
│       ├── plan_evaluation.py - Plan evaluation interface
│       ├── planning_tab.py - Planning tab
│       ├── qa_tab.py - Quality assurance tab
│       ├── reporting_tab.py - Reporting tab
│       ├── structure_editor.py - Structure editing interface
│       ├── structure_view.py - Structure viewing interface
│       ├── styles/ - UI styles
│       ├── threshold_contour_tool.py - Threshold-based contouring
│       ├── treatment_planning_tab.py - Treatment planning tab
│       ├── treatment_tab.py - Treatment delivery tab
│       └── workflow_panel.py - Workflow panel
└── scripts/ - Development and installation scripts
    ├── generate_docs.sh - Documentation generation
    ├── install.sh - Installation script
    └── setup_dev.sh - Development environment setup

## System Architecture
The QuangTPS system follows a modular architecture with clear separation of concerns:
Data Layer: Database and file management (database/, core/patient/)
Business Logic Layer: Core algorithms and processing (dose/, planning/, optimization/, evaluation/)
Presentation Layer: User interface components (ui/)
Integration Layer: APIs and external system integration (api/, reporting/)
Key architectural relationships:
The Patient model is central, connecting to Studies, Series, Images, Structures, Plans, and Doses
Treatment Plans reference Beams, which reference MLCs and Dose calculations
The Evaluation module depends on Dose, Plans, and Structures
The UI components communicate with business logic through controller classes
The system follows these design principles:
Modular design with clear component boundaries
Separation of UI and business logic
Comprehensive database for persistent storage
Support for standard medical imaging formats (DICOM)
Extensibility through plugins and APIs