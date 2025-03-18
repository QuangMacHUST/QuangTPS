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


