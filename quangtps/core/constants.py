"""
Định nghĩa các hằng số dùng chung trong hệ thống QuangTPS.
"""

class Constants:
    """Chứa các hằng số dùng chung trong hệ thống"""
    
    # Các loại dữ liệu DICOM
    DICOM_CT = "CT"
    DICOM_MR = "MR"
    DICOM_PT = "PT"
    DICOM_RTPLAN = "RTPLAN"
    DICOM_RTDOSE = "RTDOSE"
    DICOM_RTSTRUCT = "RTSTRUCT"
    DICOM_RTIMAGE = "RTIMAGE"
    
    # Các phép toán contour
    BOOLEAN_UNION = "UNION"
    BOOLEAN_INTERSECTION = "INTERSECTION"
    BOOLEAN_SUBTRACTION = "SUBTRACTION"
    BOOLEAN_EXCLUSIVE_OR = "EXCLUSIVE_OR"
    
    # Các thuật toán tính liều
    DOSE_CCC = "Collapsed Cone Convolution"
    DOSE_PENCIL_BEAM = "Pencil Beam"
    DOSE_AAA = "Analytical Anisotropic Algorithm"
    DOSE_ACUROS = "Acuros XB"
    DOSE_CONVOLUTION = "Convolution Superposition"
    DOSE_MONTE_CARLO = "Monte Carlo"
    DOSE_GBBS = "Grid-Based Boltzmann Solver"
    
    # Các kỹ thuật xạ trị
    TECHNIQUE_3DCRT = "3D-CRT"
    TECHNIQUE_IMRT = "IMRT"
    TECHNIQUE_VMAT = "VMAT"
    TECHNIQUE_DCAT = "DCAT"
    TECHNIQUE_SRS = "SRS/SBRT"
    TECHNIQUE_PROTON = "Proton Therapy"
    TECHNIQUE_CARBON = "Carbon Ion Therapy"
    TECHNIQUE_ELECTRON = "Electron Therapy"
    
    # Các phương pháp tối ưu hóa
    OPT_MCO = "Multi-Criteria Optimization"
    OPT_CONSTRAINT = "Constraint-based Optimization"
    OPT_OBJECTIVE = "Objective-based Optimization"
    OPT_ROBUST = "Robust Optimization"
    OPT_BIOLOGICAL = "Biological Optimization"
    
    # Các chỉ số đánh giá kế hoạch
    METRIC_CI = "Conformity Index"
    METRIC_HI = "Homogeneity Index"
    METRIC_GI = "Gradient Index"
    METRIC_TCP = "Tumor Control Probability"
    METRIC_NTCP = "Normal Tissue Complication Probability"
    
    # Định dạng báo cáo
    REPORT_PDF = "PDF"
    REPORT_DOCX = "DOCX"
    REPORT_HTML = "HTML"
    
    # Loại cấu trúc
    STRUCTURE_PTV = "PTV"
    STRUCTURE_CTV = "CTV"
    STRUCTURE_GTV = "GTV"
    STRUCTURE_OAR = "OAR"
    STRUCTURE_EXTERNAL = "EXTERNAL"
    STRUCTURE_SUPPORT = "SUPPORT"
    
    # Màu mặc định cho các cấu trúc
    STRUCTURE_COLORS = {
        "PTV": [255, 0, 0],      # Đỏ
        "CTV": [255, 153, 0],    # Cam
        "GTV": [255, 255, 0],    # Vàng
        "CORD": [0, 255, 0],     # Lục
        "HEART": [0, 0, 255],    # Lam
        "LUNG": [0, 255, 255],   # Xanh da trời
        "LIVER": [153, 0, 255],  # Tím
        "EXTERNAL": [0, 153, 0]  # Xanh lá đậm
    }
