# YOLO to TensorFlow / TensorFlow Lite Converter

Convert YOLO object detection models (v3, v4, v5) to TensorFlow and TensorFlow Lite for deployment on edge devices, mobile phones, and embedded systems.

<table>
  <tr>
    <td><img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></td>
    <td><img src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /></td>
    <td><img src="https://img.shields.io/badge/YOLO%20v3%2Fv4%2Fv5-%23121011.svg?&style=for-the-badge" /></td>
  </tr>
</table>

## Supported Models

| Model | Backbone | Source Weights | Input Size |
|-------|----------|---------------|------------|
| YOLOv3 | Darknet53 | `.weights` (Darknet) | 416 |
| YOLOv3-tiny | Darknet53-tiny | `.weights` (Darknet) | 416 |
| YOLOv4 | CSPDarknet53 | `.weights` (Darknet) | 416 |
| YOLOv4-tiny | CSPDarknet53-tiny | `.weights` (Darknet) | 416 |
| **YOLOv5s** | CSPDarknet + SPPF + PANet | `.pt` (PyTorch) or `.npz` | 640 |

### YOLOv5 Architecture

YOLOv5 introduces several improvements over v3/v4:

- **CSPDarknet backbone** with C3 modules (simplified CSP blocks) and SiLU activation
- **SPPF** (Spatial Pyramid Pooling Fast) - sequential max-pooling for multi-scale features
- **PANet neck** - bidirectional feature aggregation (top-down + bottom-up)
- **Bounded predictions** - `(2*sigmoid(t))^2 * anchor` for width/height prevents gradient explosion
- **Configurable model sizes** via width/depth multipliers (n/s/m/l/x)

## Quick Start

### 1. Set up environment

```bash
git clone <this-repo> && cd yolo-2-tflite
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

### 2. Download weights

**YOLOv4 (Darknet):**
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (COCO, 80 classes)

**YOLOv5 (PyTorch):**
- Download from [Ultralytics YOLOv5 releases](https://github.com/ultralytics/yolov5/releases) (e.g., `yolov5s.pt`)

### 3. Convert to TensorFlow

**YOLOv4:**
```bash
python save_model.py --weights ./data/yolov4.weights \
    --output ./checkpoints/yolov4-416 \
    --input_size 416 --model yolov4
```

**YOLOv5:**
```bash
# Option A: Direct from PyTorch weights (requires torch)
python save_model.py --weights ./data/yolov5s.pt \
    --output ./checkpoints/yolov5s-640 \
    --input_size 640 --model yolov5

# Option B: Convert weights first (on a machine with torch), then use .npz
python convert_weights.py --weights ./data/yolov5s.pt --output ./data/yolov5s.npz
python save_model.py --weights ./data/yolov5s.npz \
    --output ./checkpoints/yolov5s-640 \
    --input_size 640 --model yolov5
```

### 4. Convert to TensorFlow Lite

First, save the model with `--framework tflite`:
```bash
python save_model.py --weights ./data/yolov5s.pt \
    --output ./checkpoints/yolov5s-640 \
    --input_size 640 --model yolov5 --framework tflite
```

Then convert to `.tflite`:
```bash
python convert_tflite.py --weights ./checkpoints/yolov5s-640 \
    --output ./checkpoints/yolov5s-640.tflite
```

Quantization options:
```bash
# Float16 (smaller, slight accuracy loss)
python convert_tflite.py --weights ./checkpoints/yolov5s-640 \
    --output ./checkpoints/yolov5s-640-fp16.tflite \
    --quantize_mode float16

# Int8 (smallest, requires calibration dataset)
python convert_tflite.py --weights ./checkpoints/yolov5s-640 \
    --output ./checkpoints/yolov5s-640-int8.tflite \
    --quantize_mode int8 --dataset /path/to/calibration.txt
```

### 5. Run detection

**Image detection:**
```bash
# TensorFlow SavedModel
python detect.py --weights ./checkpoints/yolov5s-640 \
    --size 640 --model yolov5 \
    --images ./data/images/kite.jpg

# TensorFlow Lite
python detect.py --weights ./checkpoints/yolov5s-640.tflite \
    --size 640 --model yolov5 --framework tflite \
    --images ./data/images/kite.jpg
```

**Video detection:**
```bash
python detect_video.py --weights ./checkpoints/yolov5s-640 \
    --size 640 --model yolov5 \
    --video ./data/video/video.mp4 \
    --output ./detections/result.avi
```

## Conversion Pipeline

```
YOLOv3/v4 (.weights)  ──┐
                         ├──> save_model.py ──> TF SavedModel (.pb)
YOLOv5 (.pt or .npz) ──┘         │
                                  ├──> convert_tflite.py ──> TFLite (.tflite)
                                  │
                                  └──> detect.py / detect_video.py (inference)
```

## Project Structure

```
.
├── core/
│   ├── backbone.py      # Backbone architectures (Darknet53, CSPDarknet, CSPDarknet-v5)
│   ├── common.py        # Building blocks (Conv, BN, Residual, C3, SPPF, SiLU)
│   ├── config.py        # Configuration (anchors, strides, training params)
│   ├── dataset.py       # Dataset loading and preprocessing
│   ├── utils.py         # Weight loading, image processing, NMS, IoU
│   └── yolov4.py        # Model architectures (YOLOv3/v4/v5) and decode functions
├── data/
│   ├── classes/
│   │   └── coco.names   # COCO 80-class labels
│   ├── images/          # Sample test images
│   └── video/           # Sample test videos
├── save_model.py        # Convert weights to TensorFlow SavedModel
├── convert_tflite.py    # Convert SavedModel to TFLite
├── convert_weights.py   # Convert YOLOv5 .pt weights to .npz (no torch needed at inference)
├── detect.py            # Image detection
├── detect_video.py      # Video/webcam detection
├── requirements.txt     # Python dependencies
└── README.md
```

## YOLOv5 Model Sizes

The YOLOv5 implementation supports different model sizes via width/depth multipliers:

| Model | Width | Depth | Params (approx) |
|-------|-------|-------|-----------------|
| YOLOv5n | 0.25 | 0.33 | 1.9M |
| YOLOv5s | 0.50 | 0.33 | 7.2M |
| YOLOv5m | 0.75 | 0.67 | 21.2M |
| YOLOv5l | 1.00 | 1.00 | 46.5M |
| YOLOv5x | 1.25 | 1.33 | 86.7M |

Default is YOLOv5s. The architecture automatically adjusts channel widths and block depths.

## Command Line Reference

```
save_model.py:
  --weights    Path to weights file (.weights for v3/v4, .pt/.npz for v5)
  --output     Path to output SavedModel directory
  --tiny       Use tiny variant (v3/v4 only)
  --input_size Input image size (default: 416, use 640 for v5)
  --score_thres Score threshold for filtering (default: 0.2)
  --framework  Output framework: tf, trt, tflite (default: tf)
  --model      Model type: yolov3, yolov4, yolov5 (default: yolov4)

convert_tflite.py:
  --weights       Path to SavedModel directory
  --output        Path to output .tflite file
  --quantize_mode Quantization: float32, float16, int8 (default: float32)
  --dataset       Calibration dataset for int8 quantization

convert_weights.py:
  --weights  Path to YOLOv5 .pt file
  --output   Path to output .npz file

detect.py:
  --framework  tf, tflite, or trt (default: tf)
  --weights    Path to model weights
  --size       Input image size (default: 416)
  --tiny       Use tiny variant (v3/v4 only)
  --model      yolov3, yolov4, or yolov5 (default: yolov4)
  --images     Comma-separated image paths
  --output     Output directory (default: ./detections/)
  --iou        IoU threshold (default: 0.45)
  --score      Score threshold (default: 0.25)

detect_video.py:
  --video          Input video path or 0 for webcam
  --output         Output video path
  --output_format  Video codec (default: XVID)
  (other flags same as detect.py)
```

## References

- [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) - Original YOLOv4 TensorFlow implementation
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5) - Official YOLOv5 PyTorch implementation
- [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) - Darknet YOLO framework
