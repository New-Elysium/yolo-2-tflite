#!/usr/bin/env python
# coding=utf-8

"""
Dynamic Model Builder for Darknet Configurations
Builds TensorFlow/Keras models from parsed Darknet CFG files
"""

import tensorflow as tf
import keras
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from core.cfg_parser import CFGParser, parse_activation_function, parse_padding


class ModelBuilder:
    """Builds TensorFlow models from Darknet CFG configurations"""

    def __init__(self, cfg_parser: CFGParser):
        self.cfg_parser = cfg_parser
        self.net_config = cfg_parser.get_net_config()
        self.layers = cfg_parser.get_layers()
        self.layer_outputs = []
        self.layer_names = []
        self.yolo_output_indices = []
        self.current_input = None

    def build_model(self) -> keras.Model:
        """Build the complete model from CFG"""
        input_shape = self.cfg_parser.get_input_size()

        # Create input layer
        input_layer = keras.layers.Input(shape=input_shape, name='input')
        self.current_input = input_layer
        self.layer_outputs.append(input_layer)
        self.layer_names.append('input')

        # Build each layer
        for i, layer_config in enumerate(self.layers):
            layer_output = self._build_layer(layer_config, i)
            if layer_output is not None:
                self.current_input = layer_output
            self.layer_outputs.append(self.current_input)
            self.layer_names.append(f"layer_{i}_{layer_config.get('type', 'unknown')}")

        # Collect YOLO output layers (outputs from the conv layer preceding each [yolo])
        yolo_outputs = [self.layer_outputs[idx] for idx in self.yolo_output_indices]

        if not yolo_outputs:
            yolo_outputs = [self.current_input]

        # Create model
        model = keras.Model(inputs=input_layer, outputs=yolo_outputs)
        return model

    def _build_layer(self, layer_config: Dict[str, Any], layer_idx: int) -> Optional[Any]:
        """Build a single layer from configuration"""
        layer_type = layer_config.get('type', '').lower()

        layer_builders = {
            'convolutional': self._build_convolutional,
            'maxpool': self._build_maxpool,
            'route': self._build_route,
            'shortcut': self._build_shortcut,
            'upsample': self._build_upsample,
            'yolo': self._build_yolo,
            'dropout': self._build_dropout,
            'connected': self._build_connected,
            'softmax': self._build_softmax,
            'cost': self._build_cost,
            'avgpool': self._build_avgpool,
            'local_avgpool': self._build_local_avgpool,
            'reorg': self._build_reorg,
            'region': self._build_region,
            'sam': self._build_sam,
            'scale_channels': self._build_scale_channels,
            'spp': self._build_spp,
        }

        builder = layer_builders.get(layer_type)
        if builder:
            return builder(layer_config, layer_idx)
        else:
            print(f"Warning: Unsupported layer type '{layer_type}' at index {layer_idx}, passing through")
            return None

    def _build_convolutional(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build convolutional layer"""
        filters = layer_config.get('filters', 1)
        kernel_size = layer_config.get('size', 1)
        stride = layer_config.get('stride', 1)
        pad = layer_config.get('pad', 0)
        groups = layer_config.get('groups', 1)
        batch_normalize = layer_config.get('batch_normalize', 0)
        activation = parse_activation_function(layer_config.get('activation', 'linear'))

        # Darknet: pad=1 means padding=size/2 (i.e. 'same' in TF)
        if pad:
            padding = 'same'
        else:
            padding = 'valid'

        x = self.current_input

        # Handle asymmetric padding for stride > 1 with 'same' padding
        if stride > 1 and padding == 'same':
            x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
            padding = 'valid'

        # Create convolutional layer
        conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            groups=groups,
            use_bias=not batch_normalize,
            kernel_regularizer=keras.regularizers.l2(0.0005),
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=keras.initializers.Constant(0.),
            name=f"conv2d_{layer_idx}"
        )(x)

        # Batch normalization
        if batch_normalize:
            conv = keras.layers.BatchNormalization(
                name=f"batch_normalization_{layer_idx}"
            )(conv)

        # Activation
        conv = self._apply_activation(conv, activation)

        return conv

    def _apply_activation(self, x, activation: str):
        """Apply activation function"""
        if activation == 'leaky':
            return keras.layers.LeakyReLU(negative_slope=0.1)(x)
        elif activation == 'mish':
            return x * tf.math.tanh(tf.math.softplus(x))
        elif activation == 'swish':
            return x * tf.nn.sigmoid(x)
        elif activation == 'relu':
            return keras.layers.ReLU()(x)
        elif activation == 'elu':
            return keras.layers.ELU()(x)
        elif activation == 'selu':
            return tf.nn.selu(x)
        elif activation == 'gelu':
            return tf.nn.gelu(x)
        elif activation == 'hardswish':
            return x * tf.nn.relu6(x + 3) / 6
        elif activation == 'hardmish':
            return tf.where(x >= 0, x, x * tf.nn.relu6(x + 2) / 2)
        # 'linear' and others: no activation
        return x

    def _build_maxpool(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build maxpool layer"""
        kernel_size = layer_config.get('size', 2)
        stride = layer_config.get('stride', 2)

        if stride == 1:
            # Darknet uses same padding for stride=1 maxpool
            pad = kernel_size // 2
            x = keras.layers.ZeroPadding2D(((pad, pad), (pad, pad)))(self.current_input)
            return keras.layers.MaxPool2D(
                pool_size=kernel_size,
                strides=stride,
                padding='valid',
                name=f"maxpool2d_{layer_idx}"
            )(x)
        else:
            return keras.layers.MaxPool2D(
                pool_size=kernel_size,
                strides=stride,
                padding='same',
                name=f"maxpool2d_{layer_idx}"
            )(self.current_input)

    def _build_route(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build route layer (concatenation or pass-through)"""
        layers_param = layer_config.get('layers', [])
        groups = layer_config.get('groups', 1)
        group_id = layer_config.get('group_id', 0)

        if not isinstance(layers_param, list):
            layers_param = [layers_param]

        # Resolve layer references
        # In Darknet, layer indices in [route] are relative to the CFG layer list.
        # Negative = relative to current, Positive = absolute index.
        # Our layer_outputs[0] = input, layer_outputs[i+1] = output of CFG layer i.
        route_inputs = []
        for ref in layers_param:
            if ref < 0:
                # Relative to current layer in CFG (layer_idx + ref)
                # In layer_outputs: CFG layer N is at index N+1
                abs_idx = layer_idx + ref + 1  # +1 because layer_outputs[0] is input
            else:
                abs_idx = ref + 1  # +1 because layer_outputs[0] is input
            if 0 <= abs_idx < len(self.layer_outputs):
                route_inputs.append(self.layer_outputs[abs_idx])
            else:
                print(f"Warning: Route layer {layer_idx} references invalid index {ref} (abs={abs_idx})")

        if len(route_inputs) == 0:
            return self.current_input

        if len(route_inputs) == 1:
            output = route_inputs[0]
        else:
            output = keras.layers.Concatenate(name=f"route_{layer_idx}")(route_inputs)

        # Handle groups (e.g., route_group in YOLOv4-tiny)
        if groups > 1:
            # Split along channel axis and select group
            channels = output.shape[-1]
            group_channels = channels // groups
            output = output[..., group_id * group_channels:(group_id + 1) * group_channels]

        return output

    def _build_shortcut(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build shortcut layer (residual connection)"""
        from_layer = layer_config.get('from', -1)
        activation = parse_activation_function(layer_config.get('activation', 'linear'))

        # Resolve the referenced layer
        if from_layer < 0:
            abs_idx = layer_idx + from_layer + 1
        else:
            abs_idx = from_layer + 1

        if 0 <= abs_idx < len(self.layer_outputs):
            shortcut_input = self.layer_outputs[abs_idx]
        else:
            print(f"Warning: Shortcut layer {layer_idx} references invalid index {from_layer}")
            return self.current_input

        # Add the layers
        output = keras.layers.Add(name=f"shortcut_{layer_idx}")([self.current_input, shortcut_input])

        # Apply activation if specified
        output = self._apply_activation(output, activation)

        return output

    def _build_upsample(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build upsample layer using Keras UpSampling2D for Keras 3 compatibility"""
        stride = layer_config.get('stride', 2)
        return keras.layers.UpSampling2D(
            size=(stride, stride),
            interpolation='bilinear',
            name=f"upsample_{layer_idx}"
        )(self.current_input)

    def _build_yolo(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """
        Handle YOLO detection layer.
        In Darknet, [yolo] is NOT a computational layer - it's a marker indicating
        that the preceding convolutional layer's output is a detection output.
        We just record the current output index for collection.
        """
        # Record that the current output (from the preceding conv layer) is a YOLO output
        # layer_outputs has len = layer_idx + 1 at this point (0=input, 1..layer_idx=previous layers)
        # The current output is at index layer_idx (since we haven't appended this layer yet)
        self.yolo_output_indices.append(len(self.layer_outputs))
        return self.current_input  # Pass through unchanged

    def _build_dropout(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build dropout layer"""
        probability = layer_config.get('probability', 0.5)
        return keras.layers.Dropout(rate=probability, name=f"dropout_{layer_idx}")(self.current_input)

    def _build_connected(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build fully connected layer"""
        output_size = layer_config.get('output', 1)
        activation = parse_activation_function(layer_config.get('activation', 'linear'))

        # Flatten input first
        flattened = keras.layers.Flatten(name=f"flatten_{layer_idx}")(self.current_input)

        # Dense layer
        dense = keras.layers.Dense(
            units=output_size,
            activation='linear',
            name=f"connected_{layer_idx}"
        )(flattened)

        # Apply activation
        dense = self._apply_activation(dense, activation)
        return dense

    def _build_softmax(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build softmax layer"""
        return keras.layers.Softmax(name=f"softmax_{layer_idx}")(self.current_input)

    def _build_cost(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build cost layer (training only, no-op for inference)"""
        return self.current_input

    def _build_avgpool(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build global average pooling layer"""
        return keras.layers.GlobalAveragePooling2D(
            name=f"avgpool_{layer_idx}"
        )(self.current_input)

    def _build_local_avgpool(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build local average pooling layer"""
        kernel_size = layer_config.get('size', 2)
        stride = layer_config.get('stride', 2)
        return keras.layers.AveragePooling2D(
            pool_size=kernel_size,
            strides=stride,
            padding='same',
            name=f"local_avgpool_{layer_idx}"
        )(self.current_input)

    def _build_reorg(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build reorganization layer (space_to_depth)"""
        stride = layer_config.get('stride', 2)
        return tf.nn.space_to_depth(self.current_input, block_size=stride)

    def _build_region(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build region layer (older YOLO v2 detection head, treated like [yolo])"""
        self.yolo_output_indices.append(len(self.layer_outputs))
        return self.current_input

    def _build_sam(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build SAM (Spatial Attention Module) layer.
        In Darknet, SAM performs element-wise multiplication with a sigmoid-activated
        input from a referenced layer."""
        from_layer = layer_config.get('from', -1)
        if from_layer < 0:
            abs_idx = len(self.layer_outputs) + from_layer
        else:
            abs_idx = from_layer + 1
        if 0 <= abs_idx < len(self.layer_outputs):
            attention = tf.nn.sigmoid(self.layer_outputs[abs_idx])
            return keras.layers.Multiply(name=f"sam_{layer_idx}")([self.current_input, attention])
        return self.current_input

    def _build_scale_channels(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build scale_channels layer (element-wise multiply with referenced layer)"""
        from_layer = layer_config.get('from', -1)
        if from_layer < 0:
            abs_idx = len(self.layer_outputs) + from_layer
        else:
            abs_idx = from_layer + 1
        if 0 <= abs_idx < len(self.layer_outputs):
            return keras.layers.Multiply(
                name=f"scale_channels_{layer_idx}"
            )([self.current_input, self.layer_outputs[abs_idx]])
        return self.current_input

    def _build_spp(self, layer_config: Dict[str, Any], layer_idx: int) -> Any:
        """Build SPP (Spatial Pyramid Pooling) layer"""
        # In Darknet, SPP does maxpool at multiple kernel sizes with stride=1 same-padding
        # Default kernel sizes from Darknet: 5, 9, 13
        kernel_sizes = layer_config.get('maxpool_sizes', [5, 9, 13])
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [5, 9, 13]

        pools = [self.current_input]
        for ksize in kernel_sizes:
            pad = ksize // 2
            padded = keras.layers.ZeroPadding2D(
                ((pad, pad), (pad, pad))
            )(self.current_input)
            pool = keras.layers.MaxPool2D(
                pool_size=ksize,
                strides=1,
                padding='valid',
                name=f"spp_pool_{ksize}_{layer_idx}"
            )(padded)
            pools.append(pool)

        return keras.layers.Concatenate(name=f"spp_{layer_idx}")(pools)


def build_model_from_cfg(cfg_path: str) -> keras.Model:
    """Convenience function to build model from CFG file"""
    parser = CFGParser(cfg_path)
    builder = ModelBuilder(parser)
    return builder.build_model()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        model = build_model_from_cfg(cfg_path)
        model.summary()
    else:
        print("Usage: python model_builder.py <path_to_cfg_file>")
