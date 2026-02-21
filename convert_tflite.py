#!/usr/bin/env python
# coding=utf-8

"""
TensorFlow Lite Conversion Script - Supports generic Darknet models via CFG parser
"""

import tensorflow as tf
import keras
import numpy as np
import cv2
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from core.cfg_parser import CFGParser
from core.model_builder import build_model_from_cfg
from core.weights_loader import load_weights_from_cfg
from core.config import get_model_type_from_cfg

# Command line flags
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file or SavedModel directory')
flags.DEFINE_string('cfg', '', 'path to cfg file (for generic models)')
flags.DEFINE_string('output', './checkpoints/yolov4-416-fp32.tflite', 'path to output')
flags.DEFINE_integer('input_size', 416, 'input size for conversion')
flags.DEFINE_string('quantize_mode', 'float32',
                    'quantize mode (int8, float16, float32)')
flags.DEFINE_string('dataset', '', 'path to dataset for int8 calibration')
flags.DEFINE_boolean('generic', False,
                     'use generic cfg parser for any Darknet model')
flags.DEFINE_boolean('save_model_first', False,
                     'save TF model before TFLite conversion (for generic models)')
flags.DEFINE_string('model', 'yolov4', 'model type (legacy mode only)')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not (legacy mode only)')


def representative_data_gen():
    """Generate representative data for INT8 quantization"""
    if not FLAGS.dataset or not os.path.exists(FLAGS.dataset):
        logging.error(f"Dataset file not found: {FLAGS.dataset}")
        raise ValueError(f"Dataset file not found: {FLAGS.dataset}")

    # Load dataset paths
    with open(FLAGS.dataset, 'r') as f:
        image_paths = f.read().split()

    # Use up to 100 images for calibration
    num_images = min(100, len(image_paths))

    logging.info(f"Using {num_images} images for INT8 calibration")

    for i in range(num_images):
        image_path = image_paths[i]

        if os.path.exists(image_path):
            try:
                original_image = cv2.imread(image_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                image_data = cv2.resize(original_image, (FLAGS.input_size, FLAGS.input_size))
                img_in = image_data[np.newaxis, ...].astype(np.float32) / 255.0

                logging.info(f"Calibration image {i+1}/{num_images}: {image_path}")
                yield [img_in]
            except Exception as e:
                logging.warning(f"Failed to load calibration image {image_path}: {e}")
                continue
        else:
            logging.warning(f"Calibration image not found: {image_path}")
            continue


def convert_from_saved_model():
    """Convert SavedModel to TFLite"""
    logging.info(f"Converting SavedModel from: {FLAGS.weights}")
    logging.info(f"Output TFLite model: {FLAGS.output}")
    logging.info(f"Quantization mode: {FLAGS.quantize_mode}")

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)

    # Configure converter based on quantization mode
    if FLAGS.quantize_mode == 'float16':
        logging.info("Applying FLOAT16 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True

    elif FLAGS.quantize_mode == 'int8':
        logging.info("Applying INT8 quantization")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.allow_custom_ops = True
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Set representative dataset for calibration
        if FLAGS.dataset:
            converter.representative_dataset = representative_data_gen
        else:
            logging.warning("INT8 quantization requires representative dataset")
            logging.warning("Please provide --dataset flag for proper INT8 calibration")
            logging.warning("Proceeding without calibration - results may be poor")

    else:  # float32
        logging.info("No quantization (float32)")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True

    # Convert model
    logging.info("Converting model...")
    tflite_model = converter.convert()

    # Save TFLite model
    with open(FLAGS.output, 'wb') as f:
        f.write(tflite_model)

    # Get model size
    model_size = os.path.getsize(FLAGS.output) / (1024 * 1024)
    logging.info(f"TFLite model saved to: {FLAGS.output}")
    logging.info(f"Model size: {model_size:.2f} MB")

    return tflite_model


def convert_from_generic_model():
    """Convert generic Darknet model (weights + cfg) to TFLite"""
    logging.info(f"Converting generic model from CFG: {FLAGS.cfg}")
    logging.info(f"Loading weights from: {FLAGS.weights}")

    if not FLAGS.cfg or not os.path.exists(FLAGS.cfg):
        raise ValueError(f"CFG file not found: {FLAGS.cfg}")

    if not FLAGS.weights or not os.path.exists(FLAGS.weights):
        raise ValueError(f"Weights file not found: {FLAGS.weights}")

    # Detect model type
    model_type = get_model_type_from_cfg(FLAGS.cfg)
    logging.info(f"Detected model type: {model_type}")

    # Parse CFG
    parser = CFGParser(FLAGS.cfg)
    parser.print_summary()

    # Get input size
    net_config = parser.get_net_config()
    input_size = net_config.get('width', FLAGS.input_size)
    FLAGS.input_size = input_size  # Update FLAGS for consistency

    logging.info(f"Using input size: {input_size}")

    # Build model from CFG
    logging.info("Building model from CFG...")
    model = build_model_from_cfg(FLAGS.cfg)
    model.summary()

    # Load weights
    logging.info("Loading Darknet weights...")
    model = load_weights_from_cfg(model, FLAGS.cfg, FLAGS.weights)
    logging.info("Weights loaded successfully!")

    # Optionally save TF model first
    if FLAGS.save_model_first:
        tf_model_path = FLAGS.output.replace('.tflite', '_saved_model')
        logging.info(f"Saving TensorFlow model to: {tf_model_path}")
        model.save(tf_model_path)

        # Convert from SavedModel
        FLAGS.weights = tf_model_path
        return convert_from_saved_model()

    # Create TFLite converter from Keras model
    logging.info("Creating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Configure converter based on quantization mode
    if FLAGS.quantize_mode == 'float16':
        logging.info("Applying FLOAT16 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True

    elif FLAGS.quantize_mode == 'int8':
        logging.info("Applying INT8 quantization")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.allow_custom_ops = True
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Set representative dataset for calibration
        if FLAGS.dataset:
            converter.representative_dataset = representative_data_gen
        else:
            logging.warning("INT8 quantization requires representative dataset")
            logging.warning("Please provide --dataset flag for proper INT8 calibration")
            logging.warning("Proceeding without calibration - results may be poor")

    else:  # float32
        logging.info("No quantization (float32)")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True

    # Convert model
    logging.info("Converting model...")
    tflite_model = converter.convert()

    # Save TFLite model
    with open(FLAGS.output, 'wb') as f:
        f.write(tflite_model)

    # Get model size
    model_size = os.path.getsize(FLAGS.output) / (1024 * 1024)
    logging.info(f"TFLite model saved to: {FLAGS.output}")
    logging.info(f"Model size: {model_size:.2f} MB")

    return tflite_model


def demo_tflite():
    """Demo TFLite model inference"""
    logging.info(f"Loading TFLite model: {FLAGS.output}")

    try:
        interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logging.info("TFLite model loaded successfully!")
        logging.info(f"Input details: {input_details}")
        logging.info(f"Output details: {output_details}")

        # Create dummy input
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']

        logging.info(f"Input shape: {input_shape}")
        logging.info(f"Input dtype: {input_dtype}")

        # Generate random input based on dtype
        if input_dtype == np.uint8:
            input_data = np.random.randint(0, 256, input_shape, dtype=np.uint8)
        elif input_dtype == np.int8:
            input_data = np.random.randint(-128, 128, input_shape, dtype=np.int8)
        else:  # float32 or float16
            input_data = np.random.random_sample(input_shape).astype(np.float32)

        # Run inference
        logging.info("Running inference...")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get outputs
        output_data = [interpreter.get_tensor(output_details[i]['index'])
                      for i in range(len(output_details))]

        logging.info(f"Inference successful!")
        logging.info(f"Number of outputs: {len(output_data)}")
        for i, output in enumerate(output_data):
            logging.info(f"Output {i} shape: {output.shape}, dtype: {output.dtype}")
            logging.info(f"Output {i} range: [{output.min():.4f}, {output.max():.4f}]")

    except Exception as e:
        logging.error(f"Error running TFLite demo: {e}")
        import traceback
        traceback.print_exc()


def main(_argv):
    """Main conversion function"""
    logging.info("=" * 80)
    logging.info("TensorFlow Lite Conversion")
    logging.info("=" * 80)

    try:
        # Determine conversion mode
        if FLAGS.generic and FLAGS.cfg:
            # Generic mode: convert from Darknet weights + CFG
            logging.info("Mode: Generic Darknet model conversion")
            convert_from_generic_model()
        else:
            # Legacy mode: convert from SavedModel
            logging.info("Mode: SavedModel conversion")
            convert_from_saved_model()

        # Demo the converted model
        logging.info("=" * 80)
        logging.info("Testing TFLite model")
        logging.info("=" * 80)
        demo_tflite()

        logging.info("=" * 80)
        logging.info("Conversion completed successfully!")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
