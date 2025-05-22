# QuangTPS Development Tasks

## Completed Tasks
- [x] Basic dose calculation engine
- [x] DICOM RT import/export
- [x] Structure contouring tools
- [x] DVH calculation and visualization
- [x] Forward planning tools
- [x] Inverse planning optimization
- [x] Plan evaluation tools
- [x] Basic reporting functionality
- [x] Implement Knowledge-Based Planning (KBP) similar to RapidPlan
- [x] Implement MCO with Eclipse-like interface
- [x] Fix stability issues with MCO and KBP modules
- [x] Enhanced VMAT optimization implementation
- [x] Added test file for dose calculation algorithms
- [x] Enhanced beam visualization with 3D capabilities
- [x] Implemented system check utility script
- [x] Created comprehensive user guide
- [x] Created main entry point script with command-line options
- [x] Created dose calculation testing script with phantom
- [x] Added README file in the scripts directory
- [x] Created RT plan template system for common treatment sites
- [x] Created template manager for applying templates to treatment plans
- [x] Added DICOM-RT plan template converter
- [x] Added template selection dialog UI
- [x] Created unit tests for the template system
- [x] Implemented auto-segmentation module with AI models
- [x] Added support for multiple dose calculation algorithms
- [x] Improved structure visualization in 3D view
- [x] Added plan evaluation metrics
- [x] Created clinical protocol system for plan evaluation
- [x] Implemented plan comparison functionality
- [x] Added patient database integration
- [x] Created robust analysis module
- [x] Enhanced UI with Eclipse-like interface
- [x] Added Pareto surface navigation for MCO
- [x] Improved object explorer panel with Eclipse-like functionality
- [x] Implemented better integration between Structure tab and External Beam Planning tab
- [x] Add multi-criteria optimization to Eclipse-like interface
- [x] Create comprehensive dialog system for plan and structure properties
- [x] Implement complete dialog system with Eclipse-like styling
- [x] Enhance error handling and fallback mechanisms for missing components
- [x] Improve import error handling with more robust fallback mechanisms
- [x] Implement comprehensive dose calculation algorithm selection UI
- [x] Add knowledge-based planning features:
  - [x] Created KBPDialog with RapidPlan-style interface
  - [x] Implemented KBP button in External Beam Planning toolbar
  - [x] Developed model information display with feature importance visualization
  - [x] Added automatic objectives and constraints application from KBP recommendations
  - [x] Integrated KBP seamlessly with inverse planning workflow
- [x] Enhance synchronization between ObjectExplorerPanel and all tabs
- [x] Implement inverse planning algorithms for IMRT/VMAT optimization
- [x] Create comprehensive documentation system
- [x] Implement radiobiology modeling for TCP/NTCP in plan evaluation
- [x] Implement robust analysis tools for treatment plan evaluation
- [x] Fix KBP Dialog missing method for site selection
- [x] Fix VMATOptimizer missing dose calculation method
- [x] Enhance Monte Carlo GPU algorithm with improved error handling
- [x] Fix imports in KBP Dialog

## In Progress
- [ ] Implement machine learning dose prediction module
- [ ] Create adaptive planning workflow
- [ ] Add support for proton therapy planning
- [x] Enhance Monte Carlo dose calculation with GPU acceleration
- [ ] Implement automated quality assurance tools
- [ ] Add export functionality for 3D visualizations and BEV snapshots
- [ ] Implement radiobiology modeling for TCP/NTCP in plan evaluation
- [ ] Implement MCO with Eclipse-like interface
- [x] Optimize VMAT delivery time calculation

## Planned Tasks
- [ ] Create advanced reporting system with customizable templates
- [ ] Implement deformable image registration for adaptive planning
- [ ] Add support for brachytherapy planning
- [ ] Create machine log file analysis tools
- [ ] Implement collision detection system
- [ ] Add support for stereotactic radiosurgery planning
- [ ] Create treatment delivery simulation
- [ ] Implement beam data modeling tools
- [ ] Add support for multiple treatment machines
- [ ] Create patient-specific QA tools
- [ ] Implement advanced biological models for treatment evaluation
- [ ] Add support for 4D planning
- [ ] Create workflow for MR-guided radiotherapy
- [ ] Implement advanced optimization techniques (FMO, DAO)
- [ ] Add support for scripting and automation
- [ ] Create plugin system for extending functionality
- [ ] Implement DICOM-RT Ion support
- [ ] Add support for online adaptive planning
- [ ] Implement multi-language support
- [ ] Add treatment scheduling and fractionation tools
- [ ] Implement comprehensive dose comparison tools with gamma analysis
- [ ] Add advanced visualization for Knowledge-Based Planning models
- [ ] Enhance machine learning integration for treatment response prediction

## Current Tasks
- [ ] Extend robust optimization based on robustness analysis results
- [ ] Add comprehensive GPU memory management for large Monte Carlo calculations
- [ ] Implement real-time adaptive planning with anatomy prediction
- [ ] Enhance robustness analysis with machine-specific uncertainty models
- [ ] Implement probabilistic robustness analysis framework
- [ ] Create scenario-based planning optimization derived from robustness results
- [ ] Develop advanced biological metrics for treatment plan evaluation
- [ ] Implement automated protocol selection based on treatment site
- [ ] Create knowledge-based quality metrics using historical plan data
- [ ] Add auto-recovery and auto-save for patient and plan data
- [ ] Enhance VMAT control point optimization with improved algorithms
- [ ] Implement efficient machine learning prediction for dose calculation acceleration
- [ ] Create comprehensive documentation for the Monte Carlo GPU implementation

## Future Tasks
- [ ] Further enhance Eclipse-like UI with ribbon interface
- [ ] Add scripting support within External Beam Planning tab
- [ ] Implement plan template library similar to Eclipse
- [ ] Create clinical protocol system for automated plan evaluation
- [ ] Implement adaptive planning workflow
- [ ] Create machine learning module for dose prediction
- [ ] Add automated plan quality evaluation
- [ ] Enhance reporting capabilities with custom templates
- [ ] Add support for 4D planning with motion management
- [ ] Implement automated segmentation using deep learning
- [ ] Create an API for integration with external systems
- [ ] Add support for proton therapy planning
- [ ] Implement Monte Carlo dose calculation for electrons
- [ ] Add biologically effective dose (BED) calculation tools
- [ ] Create QA tools for IMRT/VMAT plan verification
- [ ] Implement patient-specific QA workflow
- [ ] Add support for deformable image registration
- [ ] Create pre-treatment validation workflow
- [ ] Develop uncertainty analysis tools for multi-institutional studies
- [ ] Add distributed Monte Carlo computation across multiple workstations
- [ ] Implement anatomically constrained dose painting for biologically guided RT
- [ ] Develop comprehensive auto-segmentation module
- [ ] Integrate with external treatment planning systems
- [ ] Implement real-time plan quality checks
- [ ] Create automation scripts for routine planning tasks
- [ ] Develop AI assistant for treatment planning
- [ ] Add support for multi-modality image registration
- [ ] Create interface for external beam calibration data
- [ ] Implement brachytherapy planning module
- [ ] Add radiation biology models for treatment optimization
- [ ] Develop 4D dose calculation and optimization

## High Priority

- [ ] Improve contouring tools with auto-segmentation integration
- [ ] Enhance real-time dose calculation and display in planning interface
- [ ] Add patient-specific QA module integrated with External Beam Planning tab
- [ ] Implement comprehensive clinical protocol system for plan validation
- [ ] Add DICOM-RT export functionality for treatment plans
- [ ] Create better integration with record-and-verify systems
- [ ] Expand GPU acceleration to support multiple vendors (NVIDIA, AMD, Intel)
- [ ] Improve error handling for all critical algorithms in production mode
- [ ] Extend robustness analysis to support respiratory motion and anatomical deformations

## Medium Priority

- [ ] Enhance the DVH analysis tools with additional metrics
- [ ] Improve user interface for contouring tools
- [ ] Add multi-threading support for dose calculation
- [ ] Create comprehensive documentation for system components
- [ ] Implement database migration tools
- [ ] Add pre-configured clinical protocols for common treatment sites
- [ ] Complete implementation of clinical protocols
- [ ] Add more automated tests for critical components
- [ ] Implement constraint-based planning workflow
- [ ] Add more visualization options for 3D dose distribution
- [ ] Improve the GUI performance for large datasets
- [ ] Implement auto-contouring for common structures using ML models
- [ ] Add comprehensive user preference system
- [ ] Enhance robustness analysis by incorporating auto-segmentation uncertainty

## Low Priority

- [ ] Add machine learning component for dose prediction
- [ ] Create visualization tools for treatment delivery
- [ ] Enhance 3D visualization with PyVista integration
- [ ] Add plugin system for custom extensions
- [ ] Create mobile viewing application for plan review
- [ ] Add more documentation and comments to the codebase
- [ ] Implement database backup and restore functionality
- [ ] Optimize database queries for better performance
- [ ] Add more treatment planning templates
- [ ] Implement multi-threaded optimization
- [ ] Add support for additional treatment delivery systems
- [ ] Implement integration with commercial PACS systems
- [ ] Create web-based robustness analysis viewer for remote plan evaluation
- [ ] Add machine-specific robustness models based on QA measurement data

## Đánh giá kế hoạch

- [x] Cải thiện module DVH để tính toán các chỉ số đánh giá kế hoạch tự động
- [x] Triển khai module phân tích sinh học (TCP, NTCP, EUD)
- [x] Tạo giao diện hiển thị các chỉ số sinh học trong tab đánh giá kế hoạch
- [x] Phát triển module đánh giá độ bền vững (robustness) của kế hoạch
- [ ] Cải thiện giao diện so sánh kế hoạch với khả năng so sánh nhiều kế hoạch cùng lúc
- [ ] Thêm khả năng xuất báo cáo đánh giá kế hoạch theo mẫu có thể tùy chỉnh

## Tối ưu hóa kế hoạch

- [x] Sửa lỗi trong module MCO (Multi-Criteria Optimization)
- [x] Cải thiện giao diện điều hướng Pareto trong MCO
- [x] Cải thiện hiệu suất tính toán trong quá trình tối ưu hóa
- [ ] Thêm các thuật toán tối ưu hóa mới (ví dụ: IPOPT, SLSQP)
- [ ] Thêm khả năng tối ưu hóa dựa trên các chỉ số sinh học (TCP, NTCP)
- [x] Tối ưu hóa thời gian phân phối liều VMAT với thuật toán vector hóa

## Tính toán liều

- [x] Cải thiện thuật toán tính toán liều để tăng tốc độ và độ chính xác
- [x] Cải thiện thuật toán Monte Carlo trên GPU
- [x] Triển khai xử lý đa luồng cho tính toán liều
- [ ] Thêm hỗ trợ cho model photon năng lượng cao (FFF)
- [ ] Nâng cao thuật toán tính toán liều với các hiệu ứng không đồng nhất mô
- [ ] Tích hợp tính toán liều dựa trên GPU trong quy trình tối ưu hóa

## Giao diện người dùng

- [x] Cải thiện giao diện External Beam Planning tab theo phong cách Eclipse
- [x] Tích hợp hiển thị liều 3D vào External Beam Planning tab
- [ ] Cải thiện hiệu suất hiển thị 3D cho các bộ dữ liệu lớn
- [ ] Thêm khả năng tùy chỉnh giao diện người dùng theo sở thích
- [ ] Cải thiện trải nghiệm người dùng với các hướng dẫn và tooltips

## Khác

- [ ] Cải thiện hệ thống quản lý dữ liệu bệnh nhân
- [ ] Thêm khả năng nhập/xuất dữ liệu từ các hệ thống khác (ARIA, RayStation, etc.)
- [ ] Phát triển module đảm bảo chất lượng (QA) cho kế hoạch xạ trị
- [ ] Cải thiện tài liệu và hướng dẫn sử dụng
- [ ] Tối ưu hóa hiệu suất tổng thể của hệ thống

## Phiên bản 0.9.x

### Sửa lỗi
- [x] Sửa lỗi hiển thị DVH không chính xác
- [x] Khắc phục lỗi khi tải DICOM từ một số nhà sản xuất
- [x] Xử lý lỗi indentation trong mco_engine.py
- [x] Sửa lỗi trùng lặp hàm create_eclipse_icon() trong kbp_dialog.py
- [x] Khắc phục lỗi cú pháp trong monte_carlo_gpu_algorithm.py
- [ ] Khắc phục lỗi không hiển thị đúng MLC trong BEV view
- [ ] Sửa lỗi chậm khi tải dữ liệu bệnh nhân lớn

### Tính năng đang triển khai
- [x] Triển khai Knowledge-Based Planning tương tự RapidPlan
- [x] Nâng cao khả năng tùy chỉnh biểu tượng UI theo phong cách Eclipse
- [x] Cải thiện phân tích Gamma để hỗ trợ đa tham số
- [x] Tối ưu hiệu năng tính toán VMAT
- [ ] Hoàn thiện module Robust Optimization
- [ ] Tích hợp mô hình thích ứng với phân đoạn AI
- [ ] Tích hợp thuật toán dự đoán biến dạng cơ quan
- [ ] Thêm hỗ trợ xử lý kế hoạch thích ứng dựa trên CBCT

### Giao diện người dùng
- [x] Cải thiện Object Explorer Panel
- [x] Thêm KBP Dialog theo phong cách Eclipse
- [x] Nâng cấp hàm create_eclipse_icon() để hỗ trợ biểu tượng tùy chỉnh
- [ ] Thiết kế lại màn hình tính liều xạ
- [ ] Cải thiện giao diện BEV với MLC
- [ ] Cải thiện hiệu suất 3D Viewer cho mô hình lớn

### Tối ưu hiệu năng
- [x] Cải thiện hiệu năng thuật toán tính toán VMAT
- [ ] Tối ưu hóa các thuật toán Monte Carlo
- [ ] Tối ưu quá trình tính toán phân tích DVH cho các cấu trúc lớn
- [ ] Triển khai tính toán song song cho các phép biến đổi hình ảnh
- [ ] Giảm bộ nhớ sử dụng khi hiển thị các hình ảnh và cấu trúc lớn

## Phiên bản 0.10

### Tính năng mới
- [ ] Hoàn thiện thuật toán tính liều Collapsed-Cone Convolution
- [ ] Triển khai tính năng xử lý nhiều kế hoạch điều trị cùng lúc
- [ ] Thêm tính năng so sánh kế hoạch điều trị toàn diện
- [ ] Hỗ trợ điều trị theo phân đoạn liều (fractionation)
- [ ] Triển khai module kiểm tra chất lượng (QA) toàn diện

### Tích hợp hệ thống
- [ ] Kết nối với hệ thống quản lý bệnh nhân (OIS)
- [ ] Hỗ trợ xuất kế hoạch điều trị sang các hệ thống phân phối liều
- [ ] Tích hợp với PACS để truy xuất hình ảnh DICOM
- [ ] Hỗ trợ nhiều định dạng nhập/xuất dữ liệu

## Phiên bản 1.0

### Tính năng cốt lõi
- [ ] Tính năng import/export đầy đủ với tất cả các định dạng DICOM phổ biến
- [ ] Phân đoạn đa cấu trúc tự động hoàn chỉnh
- [ ] Tối ưu hóa kế hoạch IMRT/VMAT với tất cả thuật toán hiện đại
- [ ] Tất cả các công cụ đánh giá kế hoạch theo tiêu chuẩn ICRU
- [ ] Đảm bảo chất lượng đầy đủ cho kế hoạch xạ trị

### Tài liệu
- [ ] Hoàn thiện tài liệu API
- [ ] Tài liệu hướng dẫn người dùng với hình ảnh minh họa
- [ ] Tài liệu hướng dẫn cài đặt chi tiết cho từng nền tảng

### Tương thích
- [ ] Tương thích với tất cả các hệ thống lập kế hoạch thương mại
- [ ] Tương thích với nhiều loại máy điều trị phổ biến
- [ ] Hỗ trợ đa ngôn ngữ (Anh, Việt, ...)