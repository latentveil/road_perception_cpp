# Road Perception C++ — Vehicle Detection, Tracking and ONNX Runtime Deployment

A real-time road-video perception system built with **YOLOv8s + ByteTrack + ONNX Runtime + C++17 + OpenCV**.

The project covers the complete engineering pipeline from UA-DETRAC dataset preprocessing and detector training to ONNX export, PyTorch/ONNX consistency validation, manual C++ post-processing, multi-object tracking, CUDA inference and latency benchmarking.

## Pipeline

```text
UA-DETRAC
    ↓
XML annotation parsing
    ↓
YOLO dataset conversion
    ↓
YOLOv8s training / validation
    ↓
PyTorch model
    ↓
ONNX export
    ↓
ONNX Runtime validation
    ↓
C++17 + OpenCV
    ↓
Letterbox / RGB / normalization / CHW
    ↓
ONNX Runtime CUDA inference
    ↓
YOLOv8 output parsing
    ↓
Class-aware NMS
    ↓
ByteTrack
    ↓
Kalman Filter + LAPJV association
    ↓
Track ID maintenance
    ↓
Video output + latency benchmark
```

## Features

- UA-DETRAC XML annotation parsing and validation
- Sequence-level train/validation split
- YOLO-format dataset generation
- YOLOv8s vehicle detector training
- PyTorch and ONNX Runtime output consistency validation
- Manual ONNX Runtime inference pipeline
- C++17 preprocessing and post-processing
- Manual YOLOv8 raw output decoding
- Manual class-aware NMS
- ONNX Runtime CUDA Execution Provider
- ByteTrack multi-object tracking
- Low-confidence second-stage association
- IoU + detection-score fusion for first-stage association
- Kalman Filter motion prediction
- LAPJV linear assignment
- OpenCV video processing
- Avg / P50 / P95 latency benchmarking
- CMake-based C++ build

## Dataset

The project uses **UA-DETRAC**.

The development dataset contains:

- 60 official training sequences
- 82,085 annotated frames
- 598,281 vehicle instances

Class mapping:

| ID | Class |
|---:|---|
| 0 | car |
| 1 | bus |
| 2 | van |
| 3 | others |

A fixed sequence-level split is used:

- Train: 48 sequences
- Validation: 12 sequences
- Random seed: 42

The official 40-sequence test set is kept untouched during development.

Split files are stored in:

```text
configs/splits/train_sequences.txt
configs/splits/val_sequences.txt
```

Dataset images and generated labels are excluded from Git.

Expected raw-data layout:

```text
data/ua_detrac/raw/
├── Annos/
└── DETRAC-Images/
    └── DETRAC-Images/
```

## Dataset Preparation

Inspect a DETRAC annotation:

```bash
python scripts/inspect_detrac_xml.py
```

Convert annotations to YOLO format:

```bash
python scripts/convert_detrac_to_yolo.py
```

Generate the fixed sequence split:

```bash
python scripts/make_sequence_split.py
```

Prepare YOLO train/validation directories:

```bash
python scripts/prepare_yolo_dataset.py
```

Visualize one converted label:

```bash
python scripts/visualize_one_label.py
```

The training configuration is:

```text
configs/ua_detrac.yaml
```

## Detector Training

The detector is based on **YOLOv8s**.

Training command:

```bash
python scripts/train_detector.py
```

Training configuration:

```text
Input size:      640 × 640
Batch size:      32
Maximum epochs:  50
Early stopping:  patience 15
Seed:            42
```

Best checkpoint was obtained at epoch 10, with training stopping at epoch 25.

### Validation Results

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| All | 0.816 | 0.745 | 0.795 | 0.652 |
| car | 0.905 | 0.903 | 0.937 | 0.720 |
| bus | 0.952 | 0.922 | 0.965 | 0.849 |
| van | 0.752 | 0.850 | 0.872 | 0.724 |
| others | 0.654 | 0.303 | 0.407 | 0.315 |

The weaker result for `others` is consistent with its substantially smaller number of training instances.

## ONNX Export

Export the trained PyTorch model:

```bash
python scripts/export_onnx.py
```

Export configuration:

```text
Input:    1 × 3 × 640 × 640
Output:   1 × 8 × 8400
Opset:    12
Dynamic:  false
NMS:      false
```

The exported tensor contains:

```text
4 bounding-box regression channels
+
4 class-score channels
```

NMS is intentionally not embedded in the ONNX graph.

## PyTorch / ONNX Consistency Validation

Run:

```bash
python scripts/validate_onnx.py
```

Validation is performed using the same manually prepared input tensor for both PyTorch and ONNX Runtime.

Observed output:

```text
PyTorch output shape: (1, 8, 8400)
ONNX output shape:    (1, 8, 8400)

Mean absolute diff:   0.00328
Max absolute diff:    0.504
```

A post-processed single-image comparison produced the same nine detections in PyTorch and ONNX Runtime.

## Python ONNX Runtime Inference

Run manual ONNX Runtime inference with:

```bash
python scripts/infer_onnx.py
```

The script implements:

```text
OpenCV image
→ Letterbox
→ BGR to RGB
→ float32 normalization
→ HWC to CHW
→ ONNX Runtime
→ raw YOLO output parsing
→ class-aware NMS
→ original-image coordinate restoration
```

## Explicit Python Detector → ByteTrack Pipeline

The project also contains an explicit Python implementation that does not use `model.track()` as a black box:

```bash
python scripts/run_explicit_bytetrack.py
```

Pipeline:

```text
OpenCV frame
→ YOLO model.predict()
→ detection results
→ BYTETracker.update()
→ Track IDs
→ video output
```

The detector threshold is set to `0.10` so ByteTrack can use detections in the `[0.10, 0.25)` range during its low-confidence association stage.

## C++ Deployment

The final deployment pipeline is implemented in:

```text
src/main.cpp
```

ByteTrack core code is stored in:

```text
include/bytetrack/
src/bytetrack/
```

The integrated tracker contains:

- STrack state management
- Kalman Filter
- IoU distance
- detection-score fusion
- LAPJV linear assignment
- high- and low-confidence association
- lost-track buffering
- Track ID maintenance

The tracking core is adapted from the upstream `FoundationVision/ByteTrack` implementation.

The ByteTrack-derived code retains its upstream license in:

```text
licenses/ByteTrack-LICENSE
```

## C++ Dependencies

The tested environment uses:

```text
C++ standard:       C++17
Compiler:           GCC 8.3.0
OpenCV C++:         3.2.0
Eigen:              3.4.0
ONNX Runtime GPU:   1.19.2
GPU:                NVIDIA GeForce RTX 5070 Ti
NVIDIA driver:      570.144
GPU memory:         16303 MiB
```

The C++ ONNX Runtime SDK is expected at:

```text
third_party/onnxruntime-linux-x64-gpu-1.19.2/
```

This SDK directory is excluded from Git.

Expected structure:

```text
third_party/onnxruntime-linux-x64-gpu-1.19.2/
├── include/
└── lib/
```

## Build

Install the required system development packages, including OpenCV and Eigen.

Configure:

```bash
cmake -S . -B build
```

Compile:

```bash
cmake --build build -j
```

For GCC versions earlier than 9, the CMake configuration automatically links `stdc++fs` for `std::filesystem` compatibility.

## Run C++ CUDA Pipeline

The provided launcher configures the ONNX Runtime and cuDNN runtime-library paths:

```bash
./scripts/run_cpp.sh
```

Default input:

```text
videos/MVI_20012.mp4
```

Default output:

```text
results/tracking/MVI_20012_cpp_bytetrack.mp4
```

The output directory is created automatically.

The program can also receive model, video, and output paths through command-line arguments:

```bash
./scripts/run_cpp.sh path/to/model.onnx path/to/video.mp4 path/to/output.mp4
```

## C++ Processing Chain

The final C++ implementation performs:

```text
OpenCV VideoCapture
→ Letterbox resize
→ BGR to RGB
→ float32 / 255
→ HWC to CHW
→ ONNX Runtime CUDA inference
→ YOLOv8 raw tensor parsing
→ confidence filtering
→ coordinate restoration
→ class-aware NMS
→ Detection to ByteTrack Object conversion
→ Kalman prediction
→ first-stage association
→ low-score second-stage association
→ Track ID update
→ OpenCV visualization
→ VideoWriter
```

## ByteTrack Configuration

The C++ tracker is configured to follow the main behavior used by the validated Ultralytics ByteTrack baseline:

```text
Detector minimum confidence: 0.10
High-score threshold:         0.25
New-track threshold:          0.25
First association threshold:  0.80
Track buffer:                 30 frames
Score fusion:                 enabled
```

The first association uses:

```text
fused_cost = 1 - IoU × detection_score
```

The low-confidence second association uses IoU distance without score fusion.

The linear assignment implementation is LAPJV.

## Tracking Result

The complete 936-frame road sequence was processed successfully by the C++ pipeline.

Qualitative inspection showed persistent Track IDs under substantial object motion and scale change. For example, buses remained associated with the same Track ID over long portions of the sequence.

No formal MOTA, IDF1 or HOTA result is claimed in this repository.

### Demo

[Watch the 12-second C++ detection and tracking demo](demo/MVI_20012_cpp_bytetrack_demo.mp4)

The demo is a 960×540, 25 FPS H.264 excerpt generated from the final C++ ONNX Runtime + ByteTrack pipeline output.

## Performance Benchmark

Benchmark hardware:

```text
GPU:           NVIDIA GeForce RTX 5070 Ti
Driver:        570.144
Input:         640 × 640
Warmup:        50 frames
Measured:      886 frames
```

Final validated run:

| Stage | Average latency |
|---|---:|
| Preprocess | 13.414 ms |
| ONNX Runtime CUDA inference | 4.672 ms |
| Postprocess + NMS | 0.159 ms |
| ByteTrack | 1.024 ms |
| Full perception pipeline | 19.275 ms |

Latency distribution:

```text
P50: 18.930 ms
P95: 22.809 ms
```

Throughput:

```text
51.880 FPS
```

The benchmarked pipeline is defined as:

```text
preprocess
→ ONNX Runtime CUDA inference
→ YOLO post-processing / NMS
→ ByteTrack
```

Video decoding, visualization and video encoding are intentionally excluded from the latency measurement.

The current primary runtime bottleneck is CPU-side preprocessing rather than GPU inference.

## Repository Structure

```text
road_perception_cpp/
├── configs/
│   ├── splits/
│   └── ua_detrac.yaml
│
├── data/
│   └── ua_detrac/
│
├── include/
│   └── bytetrack/
│
├── licenses/
│   └── ByteTrack-LICENSE
│
├── models/
│
├── results/
│
├── scripts/
│   ├── convert_detrac_to_yolo.py
│   ├── export_onnx.py
│   ├── infer_onnx.py
│   ├── inspect_detector_output.py
│   ├── inspect_detrac_xml.py
│   ├── inspect_tracking_output.py
│   ├── make_sequence_split.py
│   ├── make_sequence_video.py
│   ├── prepare_yolo_dataset.py
│   ├── run_cpp.sh
│   ├── run_explicit_bytetrack.py
│   ├── track_video.py
│   ├── train_detector.py
│   ├── validate_onnx.py
│   └── visualize_one_label.py
│
├── src/
│   ├── bytetrack/
│   └── main.cpp
│
├── third_party/
│
├── videos/
│
├── CMakeLists.txt
├── README.md
└── requirements.txt
```

Large datasets, model weights, generated results, videos and downloaded third-party SDKs are excluded through `.gitignore`.

## Python Environment

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

The tested Python dependencies include:

```text
Ultralytics 8.4.140
PyTorch 2.7.0
TorchVision 0.22.0
NumPy 1.26.4
OpenCV Python 4.11.0.86
ONNX 1.19.1
ONNX Runtime GPU 1.19.2
ONNXSlim 0.1.96
```

## Notes

This repository focuses on the engineering path from model training to C++ deployment rather than modifying the YOLO architecture.

The main engineering work includes:

- dataset conversion and validation
- sequence-level data splitting
- detector training and evaluation
- ONNX export and numerical validation
- manual preprocessing and post-processing
- C++ CUDA inference
- ByteTrack integration
- CMake project organization
- latency benchmarking
- qualitative tracking analysis
