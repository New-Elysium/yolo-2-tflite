# Convert_Darknet_YOLO_to_TensorFlow

Darknet YOLO architectures implemented in Tensorflow and Tensorflow Lite with support for **any Darknet model** including YOLOv3, YOLOv4, YOLOv4-CSP, Scaled-YOLOv4, YOLOv7, and more.

<table boarder=0>
  <tr>
    <td><img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></td>
    <td><img src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /></td>
    <td><img src="https://img.shields.io/badge/Keras%203%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white" /></td>
    <td><img src="https://img.shields.io/badge/Google%20Drive%20-%23FF9900.svg?&style=for-the-badge&logo=google-drive&logoColor=white" /></td>
    <td><img src="https://img.shields.io/badge/Darknet%20v5%20-%23121011.svg?&style=for-the-badge" /></td>
    <td><img src="https://img.shields.io/badge/YOLOv7%20Supported%20-%2300FF00.svg?&style=for-the-badge" /></td>
  </tr>
</table>

---

## 🚀 New Features

- **✅ Generic Darknet CFG Parser**: Supports ANY Darknet model configuration
- **✅ Keras 3 Compatibility**: Modern TensorFlow 2.16+ with Keras 3
- **✅ Darknet v5 Support**: Full support for hank-ai/darknet framework v5.x "Moonlit"
- **✅ All YOLO Architectures**: YOLOv3, YOLOv4, YOLOv4-CSP, Scaled-YOLOv4, YOLOv7, YOLOv7-tiny
- **✅ Dynamic Model Building**: Automatically constructs TensorFlow models from CFG files
- **✅ Universal Weights Loading**: Generic weights loader for any Darknet model

---

## Before You start:

- [ ] In the first place You need to **have Darknet YOLO weights and CFG file to work with**. Weights might be either **custom trained** or **pre-trained** on benchmark [COCO dataset](https://cocodataset.org/#home). 
- [ ] Except weights, `.names` file is required for model to have class labels reference. For benchmark COCO dataset, file `coco.names` is already available [here](https://github.com/patryklaskowski/Convert_Darknet_YOLO_to_TensorFlow/blob/master/data/classes/coco.names).

## Start

### 1. Prepare environment

```
git clone https://github.com/your-repo/yolo-2-tflite.git && \
cd yolo-2-tflite && \
python3 -m venv env && \
source env/bin/activate && \
pip install -U pip && \
pip install -r requirements.txt
```

### 2. Put `.weights` and `.cfg` files in `./data/` folder.

### 3. Prepare `.names` file respectively to your model.
`.names` file contains all class labels for specific YOLO weights where each line represents one class name.

---

## 🎯 Usage

### Generic Model Conversion (Recommended)

Convert ANY Darknet model using the generic CFG parser:

```bash
# Convert any Darknet model
python save_model.py \
  --cfg ./data/yolov7.cfg \
  --weights ./data/yolov7.weights \
  --output ./checkpoints/yolov7-416 \
  --input_size 416 \
  --generic \
  --framework tf
```

**Supported Model Types:**
- YOLOv3, YOLOv3-tiny
- YOLOv4, YOLOv4-tiny  
- YOLOv4-CSP, Scaled-YOLOv4
- YOLOv7, YOLOv7-tiny
- Any custom Darknet configuration

### Legacy Model Conversion

For backward compatibility with existing code:

```bash
# YOLOv4 (legacy method)
python save_model.py \
  --weights ./data/yolov4.weights \
  --output ./checkpoints/yolov4-416 \
  --input_size 416 \
  --model yolov4 \
  --framework tf

# YOLOv4-tiny (legacy method)  
python save_model.py \
  --weights ./data/yolov4-tiny.weights \
  --output ./checkpoints/yolov4-tiny-416 \
  --input_size 416 \
  --model yolov4 \
  --tiny \
  --framework tf
```

### Convert to TensorFlow Lite

```bash
# Generic model to TFLite
python save_model.py \
  --cfg ./data/yolov7.cfg \
  --weights ./data/yolov7.weights \
  --output ./checkpoints/yolov7-416 \
  --input_size 416 \
  --generic \
  --framework tflite

python convert_tflite.py \
  --weights ./checkpoints/yolov7-416 \
  --output ./checkpoints/yolov7-416.tflite
```

---

## 🔧 Advanced Features

### CFG Parser Analysis

Analyze any Darknet configuration:

```bash
python core/cfg_parser.py ./data/yolov7.cfg
```

### Model Building

Build models programmatically:

```python
from core.cfg_parser import CFGParser
from core.model_builder import build_model_from_cfg
from core.weights_loader import load_weights_from_cfg

# Parse CFG
parser = CFGParser('./data/yolov7.cfg')
parser.print_summary()

# Build model
model = build_model_from_cfg('./data/yolov7.cfg')

# Load weights
model = load_weights_from_cfg(model, './data/yolov7.cfg', './data/yolov7.weights')

# Save model
model.save('./checkpoints/yolov7-custom')
```

### Configuration Loading

Dynamic configuration from CFG:

```python
from core.config import load_cfg_config, get_model_type_from_cfg

# Load configuration
config = load_cfg_config('./data/yolov7.cfg')
print(f"Input size: {config.NET.WIDTH}x{config.NET.HEIGHT}")
print(f"Classes: {config.NET.CLASSES}")

# Detect model type
model_type = get_model_type_from_cfg('./data/yolov7.cfg')
print(f"Model type: {model_type}")
```

---

## 📋 Command Line Args Reference

```
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --cfg: path to cfg file (for generic models)
    (default: '')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4 (legacy mode)
    (default: yolov4)
  --generic: use generic cfg parser for any Darknet model
    (default: False)
  --score_thres: define score threshold
    (default: 0.2)
```

---

## 🏗️ Architecture Support

### Fully Supported (Legacy Mode)
- **YOLOv3**: Standard and tiny variants
- **YOLOv4**: Standard and tiny variants

### Generic Mode (via CFG Parser)
Any Darknet model that uses the supported layer types below can be converted.
This includes YOLOv3, YOLOv4, YOLOv4-CSP, Scaled-YOLOv4, YOLOv7, and custom models,
as long as the CFG file uses standard Darknet layer types.

### Layer Types Supported
- [convolutional]: Conv2D + optional BatchNorm + activation
- [maxpool]: Max pooling layers
- [route]: Layer concatenation/routing with groups support
- [shortcut]: Residual connections (element-wise add)
- [upsample]: Upsampling layers (bilinear)
- [yolo]: YOLO detection head output marker
- [spp]: Spatial Pyramid Pooling
- [dropout]: Dropout layers
- [connected]: Fully connected / dense layers
- [avgpool]: Global average pooling
- [reorg]: Space-to-depth reorganization
- [sam]: Spatial attention module
- [scale_channels]: Channel scaling

### Activation Functions
- leaky, mish, swish, linear, relu, elu, selu, gelu, hardmish, hardswish

---

## 🧪 Testing

Test your converted model:

```bash
# Test with TensorFlow model
python detect.py \
  --weights ./checkpoints/yolov7-416 \
  --size 416 \
  --images ./data/images/test.jpg

# Test with TFLite model
python detect.py \
  --weights ./checkpoints/yolov7-416.tflite \
  --size 416 \
  --framework tflite \
  --images ./data/images/test.jpg
```

---

## 🔍 Model Type Detection

The system automatically detects model types from CFG files:

- **YOLOv7**: Contains `e-elan` or `sppcspc` layers
- **YOLOv4-CSP**: Contains `csp` layers or `mish` activation
- **YOLOv4**: Contains `spp` layers
- **YOLOv3**: Default fallback

---

## 📚 References:
- [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
- [hank-ai/darknet](https://github.com/hank-ai/darknet) - Darknet v5 framework
- [jinyu121/DW2TF](https://github.com/jinyu121/DW2TF) - Darknet Weights to TensorFlow
- [Keras 3 Migration Guide](https://keras.io/guides/migrating_to_keras_3/)
