#!/usr/bin/env python
# coding=utf-8
"""
YOLO Detection Script - Supports generic Darknet models via CFG parser
"""

import tensorflow as tf
import keras
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from core.cfg_parser import CFGParser
from core.config import load_cfg_config

# Configure GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Command line flags
flags.DEFINE_string('framework', 'tf', 'framework: tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file or SavedModel directory')
flags.DEFINE_string('cfg', '', 'path to cfg file (for generic models)')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny (legacy mode only)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4 (legacy mode only)')
flags.DEFINE_boolean('generic', False, 'use generic cfg parser for any Darknet model')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_string('classes', './data/classes/coco.names', 'path to classes file')


def load_model_and_config():
    """Load model and configuration based on flags"""
    if FLAGS.generic and FLAGS.cfg:
        # Generic mode: Load configuration from CFG
        cfg_parser = CFGParser(FLAGS.cfg)

        # Get network configuration
        net_config = cfg_parser.get_net_config()
        input_size = net_config.get('width', FLAGS.size)
        num_class = cfg_parser.get_num_classes()

        # Get YOLO layer configuration
        yolo_layers = cfg_parser.get_yolo_layers()
        num_anchors = len(yolo_layers)

        # Extract anchors from YOLO layers
        anchors = []
        for yolo_layer in yolo_layers:
            layer_anchors = yolo_layer.get('anchors', [])
            if isinstance(layer_anchors, list):
                # Split into pairs
                anchor_pairs = []
                for i in range(0, len(layer_anchors), 2):
                    if i + 1 < len(layer_anchors):
                        anchor_pairs.append([layer_anchors[i], layer_anchors[i+1]])
                anchors.append(anchor_pairs)

        # Flatten anchors for compatibility
        ANCHORS = [item for sublist in anchors for item in sublist]
        ANCHORS = np.array(ANCHORS).reshape(num_anchors, -1, 2)

        # Get strides from YOLO layers or use defaults
        STRIDES = []
        for i, yolo_layer in enumerate(yolo_layers):
            stride = yolo_layer.get('stride', 32 // (2 ** (num_anchors - 1 - i)))
            STRIDES.append(stride)
        STRIDES = np.array(STRIDES)

        # XYSCALE - use defaults for compatibility
        XYSCALE = [1.2, 1.1, 1.05][:num_anchors]

        logging.info(f"Generic model loaded from CFG: {FLAGS.cfg}")
        logging.info(f"Input size: {input_size}")
        logging.info(f"Number of classes: {num_class}")
        logging.info(f"Number of YOLO layers: {num_anchors}")

    else:
        # Legacy mode: Load hardcoded configuration
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        num_class = NUM_CLASS
        input_size = FLAGS.size

        logging.info(f"Legacy model: {FLAGS.model} (tiny={FLAGS.tiny})")

    # Load model
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        return interpreter, STRIDES, ANCHORS, num_class, XYSCALE, input_size
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        return saved_model_loaded, STRIDES, ANCHORS, num_class, XYSCALE, input_size


def run_inference(model, image_data, input_size, num_class, STRIDES, ANCHORS, XYSCALE):
    """Run inference on image data"""
    images_data = np.asarray([image_data]).astype(np.float32)

    if FLAGS.framework == 'tflite':
        # TFLite inference
        model.allocate_tensors()
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Print input/output details for debugging
        logging.info(f"Input details: {input_details}")
        logging.info(f"Output details: {output_details}")

        model.set_tensor(input_details[0]['index'], images_data)
        model.invoke()

        pred = [model.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        # Handle different TFLite output formats
        if len(pred) == 2:
            # Legacy format: [boxes, conf]
            boxes, pred_conf = pred[0], pred[1]
        else:
            # Generic format: might have multiple outputs
            # Try to identify which outputs are boxes and which are confidences
            boxes = None
            pred_conf = None

            for output in pred:
                # Boxes output typically has shape (1, num_anchors, grid, grid, 4)
                # Conf output typically has shape (1, num_anchors, grid, grid, num_classes+5)
                if output.shape[-1] == 4:
                    boxes = output
                elif output.shape[-1] > 4:
                    pred_conf = output

            if boxes is None or pred_conf is None:
                logging.warning(f"Could not identify box/conf outputs from {len(pred)} tensors")
                logging.warning(f"Output shapes: {[p.shape for p in pred]}")
                # Use first two outputs as fallback
                boxes, pred_conf = pred[0], pred[1]

        # Apply filter_boxes for NMS
        boxes, pred_conf = filter_boxes(
            boxes, pred_conf,
            score_threshold=FLAGS.score,
            input_shape=tf.constant([input_size, input_size])
        )

    else:
        # TensorFlow SavedModel inference
        infer = model.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)

        # Extract boxes and confidences
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    # Apply NMS
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )

    return boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()


def main(_argv):
    # Load model and configuration
    model, STRIDES, ANCHORS, NUM_CLASS, XYSCALE, input_size = load_model_and_config()

    # Load class names
    try:
        class_names = utils.read_class_names(FLAGS.classes)
        logging.info(f"Loaded {len(class_names)} classes from {FLAGS.classes}")
    except:
        logging.warning(f"Could not load classes file: {FLAGS.classes}")
        class_names = {i: f'class_{i}' for i in range(NUM_CLASS)}

    # Process images
    images = FLAGS.images
    for count, image_path in enumerate(images, 1):
        logging.info(f"Processing image {count}: {image_path}")

        # Load and preprocess image
        original_image = cv2.imread(image_path)
        if original_image is None:
            logging.error(f"Could not read image: {image_path}")
            continue

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.0

        # Run inference
        try:
            boxes, scores, classes, valid_detections = run_inference(
                model, image_data, input_size, NUM_CLASS, STRIDES, ANCHORS, XYSCALE
            )

            pred_bbox = [boxes, scores, classes, valid_detections]

            # Draw bounding boxes
            image = utils.draw_bbox(original_image, pred_bbox, classes=class_names, show_label=True)

            # Display image
            image = Image.fromarray(image.astype(np.uint8))
            if not FLAGS.dont_show:
                image.show()

            # Save image
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            output_path = f"{FLAGS.output}detection{count}.png"
            cv2.imwrite(output_path, image)
            logging.info(f"Saved detection to: {output_path}")

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
