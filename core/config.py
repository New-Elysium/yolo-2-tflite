#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict
from core.cfg_parser import CFGParser


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Legacy configuration (for backward compatibility)
__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Generic configuration support
def load_cfg_config(cfg_path):
    """Load configuration from CFG file"""
    parser = CFGParser(cfg_path)
    net_config = parser.get_net_config()
    
    # Create dynamic configuration
    config = edict()
    
    # Network configuration
    config.NET = edict()
    config.NET.WIDTH = net_config.get('width', 416)
    config.NET.HEIGHT = net_config.get('height', 416)
    config.NET.CHANNELS = net_config.get('channels', 3)
    config.NET.CLASSES = net_config.get('classes', 80)
    config.NET.ANCHORS = net_config.get('anchors', [])
    config.NET.STRIDES = net_config.get('strides', [8, 16, 32])
    config.NET.XYSCALE = net_config.get('xyscale', [1.2, 1.1, 1.05])
    
    # Training configuration
    config.TRAIN = edict()
    config.TRAIN.BATCH_SIZE = net_config.get('batch', 64)
    config.TRAIN.SUBDIVISIONS = net_config.get('subdivisions', 1)
    config.TRAIN.LEARNING_RATE = net_config.get('learning_rate', 0.001)
    config.TRAIN.BURN_IN = net_config.get('burn_in', 1000)
    config.TRAIN.MAX_BATCHES = net_config.get('max_batches', 500200)
    config.TRAIN.POLICY = net_config.get('policy', 'steps')
    config.TRAIN.STEPS = net_config.get('steps', [400000, 450000])
    config.TRAIN.SCALES = net_config.get('scales', [0.1, 0.1])
    config.TRAIN.MOMENTUM = net_config.get('momentum', 0.9)
    config.TRAIN.DECAY = net_config.get('decay', 0.0005)
    config.TRAIN.ANGLE = net_config.get('angle', 0)
    config.TRAIN.SATURATION = net_config.get('saturation', 1.5)
    config.TRAIN.EXPOSURE = net_config.get('exposure', 1.5)
    config.TRAIN.HUE = net_config.get('hue', 0.1)
    config.TRAIN.JITTER = net_config.get('jitter', 0.3)
    config.TRAIN.FLIP = net_config.get('flip', 1)
    config.TRAIN.GAUSSIAN = net_config.get('gaussian', 0)
    config.TRAIN.BLUR = net_config.get('blur', 0)
    config.TRAIN.MIXUP = net_config.get('mixup', 0)
    
    # Detection configuration
    config.DETECT = edict()
    config.DETECT.SCORE_THRESHOLD = net_config.get('score_threshold', 0.25)
    config.DETECT.IOU_THRESHOLD = net_config.get('iou_threshold', 0.45)
    config.DETECT.HIER_THRESHOLD = net_config.get('hier_threshold', 0.5)
    
    return config

def get_model_type_from_cfg(cfg_path):
    """Determine model type from CFG file"""
    parser = CFGParser(cfg_path)
    layers = parser.get_layers()
    
    # Check for YOLOv7 features
    for layer in layers:
        if layer.get('type') == 'e-elan':
            return 'yolov7'
        elif layer.get('type') == 'sppcspc':
            return 'yolov7'
    
    # Check for YOLOv4-CSP features
    for layer in layers:
        if layer.get('type') == 'csp':
            return 'yolov4-csp'
        elif layer.get('activation') == 'mish':
            return 'yolov4-csp'
    
    # Check for YOLOv4
    for layer in layers:
        if layer.get('type') == 'spp':
            return 'yolov4'
    
    # Default to YOLOv3
    return 'yolov3'


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5
