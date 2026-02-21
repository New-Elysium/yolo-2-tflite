#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class Mish(tf.keras.layers.Layer):
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

class SplitLayer(tf.keras.layers.Layer):
    """Split tensor along channel axis into groups and return one group."""
    def __init__(self, groups, group_id, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.group_id = group_id

    def call(self, x):
        splits = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        return splits[self.group_id]

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn:
        conv = BatchNormalization()(conv)
    if activate:
        if activate_type == "leaky":
            conv = tf.keras.layers.LeakyReLU(alpha=0.1)(conv)
        elif activate_type == "mish":
            conv = Mish()(conv)
        elif activate_type == "swish":
            conv = tf.keras.layers.Activation('swish')(conv)
    return conv

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = tf.keras.layers.Add()([short_cut, conv])
    return residual_output

def route_group(input_layer, groups, group_id):
    return SplitLayer(groups, group_id)(input_layer)

def upsample(input_layer):
    return tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(input_layer)
