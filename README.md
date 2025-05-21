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
