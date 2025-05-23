# Kiến trúc Hệ thống QuangTPS

## Sơ đồ Tổng quan Kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            QuangTPS - Treatment Planning System                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Layer    │    │   UI Layer      │    │  Business Logic │    │   Data Layer    │
│                 │    │                 │    │                 │    │                 │
│ • Physicians    │───▶│ • Main Window   │───▶│ • Core Engine   │───▶│ • DICOM Files   │
│ • Physicists    │    │ • 3D Viewer     │    │ • Dose Calc     │    │ • Patient DB    │
│ • Technicians   │    │ • DVH Widgets   │    │ • Optimization  │    │ • Plan Storage  │
│ • Researchers   │    │ • Dialogs       │    │ • Evaluation    │    │ • Templates     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Workflow Chính của Hệ thống

### 1. Patient Data Import & Management
```
DICOM Import ──▶ Image Processing ──▶ Patient Database ──▶ Structure Definition
     │                   │                    │                       │
     │                   │                    │                       ▼
     │                   │                    │              ┌─────────────────┐
     │                   │                    │              │ Auto-Segmentation│
     │                   │                    │              │ Manual Contour   │
     │                   │                    │              │ Structure Edit   │
     │                   │                    │              └─────────────────┘
     │                   │                    │
     ▼                   ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ CT Images       │ │ MR Images       │ │ PET Images      │
│ Structure Sets  │ │ Dose Grids      │ │ RT Plans        │
│ Plan Files      │ │ Beam Data       │ │ Registration    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2. Treatment Planning Workflow
```
┌─────────────────┐
│ Patient Setup   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Target & OAR    │───▶│ Beam Geometry   │───▶│ Dose Calculation│
│ Delineation     │    │ Selection       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          │                       │                       ▼
          │                       │            ┌─────────────────┐
          │                       │            │ • Pencil Beam   │
          │                       │            │ • Monte Carlo   │
          │                       │            │ • Collapsed Cone│
          │                       │            │ • GPU Accel.    │
          │                       │            └─────────────────┘
          │                       │
          ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Structure       │    │ IMRT/VMAT       │
│ Optimization    │    │ Beam Setup      │
│ Objectives      │    │ MLC Sequencing  │
└─────────────────┘    └─────────────────┘
          │                       │
          │                       ▼
          │            ┌─────────────────┐
          │            │ Plan            │
          └───────────▶│ Optimization    │
                       │ • Forward       │
                       │ • Inverse       │
                       │ • MCO           │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Plan Evaluation │
                       │ • DVH Analysis  │
                       │ • QA Metrics    │
                       │ • Robustness    │
                       └─────────────────┘
```

### 3. Core Module Interaction
```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                Core System                                          │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────┤
│   DICOM I/O     │   Dose Engine   │  Optimization   │   Evaluation    │     UI      │
│                 │                 │                 │                 │             │
│ • Import/Export │ • Algorithms    │ • Objectives    │ • DVH Analysis  │ • Main Win  │
│ • Validation    │ • GPU Support   │ • IMRT/VMAT     │ • Plan Quality  │ • 3D Viewer │
│ • Conversion    │ • Monte Carlo   │ • MCO           │ • Robustness    │ • Dialogs   │
│ • Structure     │ • Parallel      │ • KBP           │ • Biological    │ • Widgets   │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────┘
         │                   │                   │                   │           │
         └─────────────────  │ ──────────────────│───────────────────│───────────┘
                            │                   │                   │
                            ▼                   ▼                   ▼
                    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                    │ Database Layer  │ │ Config System   │ │ Plugin System   │
                    │ • Patient Data  │ │ • Settings      │ │ • Extensions    │
                    │ • Plan Storage  │ │ • Preferences   │ │ • Custom Alg.   │
                    │ • Templates     │ │ • Machine Data  │ │ • Third Party   │
                    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Chi tiết Các Module Chính

### 1. DICOM Processing (quangtps/dicom/)
- **Import**: Đọc và validate DICOM files
- **Export**: Tạo RT Plan, RT Dose, RT Structure
- **Conversion**: Transform giữa coordinate systems
- **Validation**: Kiểm tra data integrity

### 2. Dose Calculation Engine (quangtps/dose/)
- **Algorithms**: Pencil Beam, Collapsed Cone, Monte Carlo
- **GPU Acceleration**: CUDA-based calculations
- **Grid Management**: 3D dose grids với optimal memory usage
- **Physics Models**: Advanced beam modeling

### 3. Optimization Engine (quangtps/optimization/)
- **Forward Planning**: Manual beam setup
- **Inverse Planning**: IMRT optimization
- **VMAT**: Volumetric arc therapy
- **MCO**: Multi-criteria optimization với Pareto surface

### 4. Evaluation System (quangtps/evaluation/)
- **DVH Analysis**: Cumulative/differential histograms
- **Plan Quality**: Clinical goals assessment
- **Robustness**: Setup/range uncertainty analysis
- **Biological**: TCP/NTCP models

### 5. User Interface (quangtps/ui/)
- **Eclipse Style**: Professional medical software appearance
- **3D Visualization**: VTK-based rendering
- **Interactive Tools**: Real-time plan modification
- **Reporting**: Comprehensive plan documentation

## Data Flow Architecture

### Input Data Flow
```
DICOM Files ──▶ Parser ──▶ Validation ──▶ Database ──▶ UI Components
     │             │            │             │            │
     │             │            │             │            ▼
     │             │            │             │    ┌─────────────────┐
     │             │            │             │    │ Patient View    │
     │             │            │             │    │ Structure View  │
     │             │            │             │    │ Planning View   │
     │             │            │             │    └─────────────────┘
     │             │            │             │
     │             │            │             ▼
     │             │            │    ┌─────────────────┐
     │             │            │    │ Core Data Model │
     │             │            │    │ • Patient       │
     │             │            │    │ • Images        │
     │             │            │    │ • Structures    │
     │             │            │    │ • Plans         │
     │             │            │    └─────────────────┘
     │             │            │
     │             │            ▼
     │             │    ┌─────────────────┐
     │             │    │ Validation      │
     │             │    │ • DICOM Check   │
     │             │    │ • Integrity     │
     │             │    │ • Completeness  │
     │             │    └─────────────────┘
     │             │
     │             ▼
     │    ┌─────────────────┐
     │    │ DICOM Parser    │
     │    │ • RT Images     │
     │    │ • RT Structures │
     │    │ • RT Plans      │
     │    │ • RT Doses      │
     │    └─────────────────┘
     │
     ▼
┌─────────────────┐
│ File System     │
│ • CT/MR/PET     │
│ • Structure Sets│
│ • Plan Files    │
│ • Dose Files    │
└─────────────────┘
```

### Processing Data Flow
```
Input Data ──▶ Core Processing ──▶ Analysis ──▶ Optimization ──▶ Output
     │               │                │              │             │
     │               │                │              │             ▼
     │               │                │              │    ┌─────────────────┐
     │               │                │              │    │ DICOM Export    │
     │               │                │              │    │ PDF Reports     │
     │               │                │              │    │ Clinical Data   │
     │               │                │              │    └─────────────────┘
     │               │                │              │
     │               │                │              ▼
     │               │                │    ┌─────────────────┐
     │               │                │    │ Plan Optimizer  │
     │               │                │    │ • IMRT Engine   │
     │               │                │    │ • VMAT Engine   │
     │               │                │    │ • MCO Navigator │
     │               │                │    └─────────────────┘
     │               │                │
     │               │                ▼
     │               │    ┌─────────────────┐
     │               │    │ Plan Evaluator  │
     │               │    │ • DVH Analysis  │
     │               │    │ • Quality Check │
     │               │    │ • Robustness    │
     │               │    └─────────────────┘
     │               │
     │               ▼
     │    ┌─────────────────┐
     │    │ Dose Calculator │
     │    │ • Algorithm Sel │
     │    │ • Grid Setup    │
     │    │ • Computation   │
     │    └─────────────────┘
     │
     ▼
┌─────────────────┐
│ Data Processor  │
│ • Image Process │
│ • Structure Proc│
│ • Beam Setup    │
│ • Physics Model │
└─────────────────┘
```

## Error Handling & Recovery

### Exception Hierarchy
```
QuangTPSError
├── ValidationError
├── IOError
├── ImageError
├── DicomError
├── CalculationError
├── OptimizationError
├── AlgorithmError
├── BeamDataError
├── TreatmentDeliveryError
└── AdaptivePlanningError
```

### Recovery Mechanisms
- **Graceful Degradation**: Fallback algorithms khi primary fails
- **Resource Management**: Automatic cleanup và memory management
- **State Recovery**: Khôi phục trạng thái trước khi lỗi
- **User Notification**: Clear error messages với recovery options

## Performance Optimization

### Computational Performance
- **GPU Acceleration**: CUDA-based dose calculations
- **Multi-threading**: Parallel processing cho heavy operations
- **Memory Management**: Efficient grid management
- **Algorithm Selection**: Auto-select best available method

### UI Performance
- **Lazy Loading**: Load data chỉ khi cần
- **Caching**: Cache frequently accessed data
- **Progressive Rendering**: Incremental display updates
- **Background Processing**: Non-blocking operations

## Security & Validation

### Data Security
- **DICOM Compliance**: Full DICOM standard support
- **Data Validation**: Comprehensive input validation
- **Access Control**: User permission management
- **Audit Trail**: Complete operation logging

### Medical Safety
- **Double Checking**: Critical calculations verified
- **Range Checking**: Safe parameter bounds
- **Clinical Validation**: Protocol compliance checking
- **Quality Assurance**: Built-in QA tools