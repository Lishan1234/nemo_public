import sys
import tensorflow as tf
from tensorflow.keras import layers

class ReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = tf.nn.relu(x)
        return y

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

def residual_block_(x, num_filters, kernel_size, data_format='channel_last'):
    with tf.variable_scope('residual_block'):
        res = x

        x = layers.Conv2D(num_filters,
                            (kernel_size,kernel_size),
                            padding='same',
                            data_format=data_format)(x)
        x = layers.ReLU(max_value=1.0)(x)

        x = layers.Conv2D(num_filters,
                            (kernel_size,kernel_size),
                            padding='same',
                            data_format=data_format)(x)

        x = layers.Add()([x, res])

    return x

def residual_block(x, num_filters, kernel_size, data_format='channel_last', name=None):
    with tf.variable_scope('residual_block'):
        if name is None:
            res = x

            x = layers.Conv2D(num_filters,
                                (kernel_size,kernel_size),
                                padding='same',
                                data_format=data_format)(x)
            x = ReLU()(x)

            x = layers.Conv2D(num_filters,
                                (kernel_size,kernel_size),
                                padding='same',
                                data_format=data_format)(x)

            x = layers.Add()([x, res])
        else:
            res = x

            x = layers.Conv2D(num_filters,
                                (kernel_size,kernel_size),
                                padding='same',
                                data_format=data_format,
                                name='{}_conv1'.format(name))(x)
            x = ReLU()(x)

            x = layers.Conv2D(num_filters,
                                (kernel_size,kernel_size),
                                padding='same',
                                data_format=data_format,
                                name='{}_conv2'.format(name))(x)

            x = layers.Add(name='{}_add1'.format(name))([x, res])

    return x

def mobilenetv1_block(x, num_filters, kernel_size, data_format='channel_last'):
    with tf.variable_scope('mobilenetv1_block'):
        res = x
        x = layers.SeparableConv2D(num_filters,
                            (kernel_size,kernel_size),
                            padding='same',
                            data_format=data_format,
                            activation='relu'
                            )(x)
        x = layers.SeparableConv2D(num_filters,
                            (kernel_size,kernel_size),
                            padding='same',
                            data_format=data_format,
                            )(x)
        x = layers.Add()([x, res])

    return x

def mobilenetv2_block(x, num_filters, kernel_size, expand_factor, data_format='channel_last'):
    with tf.variable_scope('mobilenetv2_block'):
        res = x
        x = layers.Conv2D(num_filters * expand_factor,
                            (1,1),
                            padding='same',
                            data_format=data_format)(x)
        x = ReLU()(x)
        x = layers.DepthwiseConv2D((kernel_size,kernel_size),
                            padding='same',
                            data_format=data_format)(x)
        x = ReLU()(x)
        x = layers.Conv2D(num_filters,
                            (1,1),
                            padding='same',
                            data_format=data_format)(x)
        x = layers.Add()([x, res])

    return x

def bilinear_upsample(x, scale, data_format='channel_last'):
    return layers.UpSampling2D(size=(scale, scale), data_format=data_format, interpolation='bilinear')(x)

def nearest_upsample(x, scale, data_format='channel_last'):
    return layers.UpSampling2D(size=(scale, scale), data_format=data_format, interpolation='nearest')(x)

def transpose_upsample(x, scale, num_filters, data_format='channel_last', num_kernels=5):
    x = layers.Conv2DTranspose(filters=num_filters,
                            kernel_size=(num_kernels,num_kernels), #should be bigger than (stride) in Qualcomm SNPE
                            strides=(scale,scale),
                            padding='same',
                            data_format=data_format)(x)
    return x

def _depth_to_space_function(x):
    input = x[0]
    scale = x[1]
    return tf.depth_to_space(input, scale)

"""
        #clip
        x = x * 255.0
        x = tf.math.round(x)
        x = tf.clip_by_value(x, -510.0, 510.0)

        #quantize
        x = x + 510.0
        x = x / 4.0
        x = tf.math.round(x)

        if self.is_debug:
            with tf.device("/cpu:0"):
                tf.print(x)

        x = x - 127.5
        x = x * 4.0
        x = x * 1/255.0
"""

class Quantize(tf.keras.layers.Layer):
    def __init__(self, is_debug=False, **kwargs):
        super(Quantize, self).__init__(**kwargs)
        self.is_debug = is_debug

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.is_debug:
            with tf.device("/cpu:0"):
                tf.print(x)
        #clip
        x = x * 255.0

        #x = tf.clip_by_value(x, -510.0, 510.0)
        x = tf.clip_by_value(x, -1020.0, 1020.0)

        #quantize
        #x = x + 510.0
        #x = x / 4.0
        x = x + 1020
        x = x / 8.0
        x = tf.math.round(x)

        if self.is_debug:
            with tf.device("/cpu:0"):
                tf.print(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        pass

class Dequantize(tf.keras.layers.Layer):
    def __init__(self, is_debug=False, **kwargs):
        super(Dequantize, self).__init__(**kwargs)
        self.is_debug = is_debug

    def build(self, input_shape):
        pass

    def call(self, x):
        #dequantize
        #x = x - 127.5
        #x = x * 4.0
        #x = x * 1/255.0
        x = x - 127.5
        x = x * 8.0
        x = x * 1/255.0

        if self.is_debug:
            with tf.device("/cpu:0"):
                tf.print(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        pass

class SubPixelUpscaling(tf.keras.layers.Layer):
    """ Sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image
    and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    (https://arxiv.org/abs/1609.05158).

    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :

        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)

    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.

    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
        u = SubPixelUpscaling(scale_factor=2)(x)

        [Optional]
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
    ```

        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.

        However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
        the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
        layer can be removed.

    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, 'channels_first' or 'channels_last'.

    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)` if data_format='channels_last'.

    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = data_format

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = tf.depth_to_space(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            return tf.TensorShape([b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor])
        else:
            b, r, c, k = input_shape
            return tf.TensorShape([b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2)])

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'data_format': self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def subpixel_upsample(x, scale, num_filters, data_format='channel_last'):
    if scale == 2:
        x = layers.Conv2D(num_filters * 4,
                        (3,3),
                        padding='same',
                        data_format=data_format)(x)
        x = SubPixelUpscaling(scale_factor=scale)(x)
    elif scale == 3:
        x = layers.Conv2D(num_filters * 9,
                        (3,3),
                        padding='same',
                        data_format=data_format)(x)
        x = SubPixelUpscaling(scale_factor=scale)(x)
    elif scale == 4:
        x = layers.Conv2D(num_filters * 4,
                        (3,3),
                        padding='same',
                        data_format=data_format)(x)
        x = SubPixelUpscaling(scale_factor=2)(x)
        x = layers.Conv2D(num_filters * 4,
                        (3,3),
                        padding='same',
                        data_format=data_format)(x)
        x = SubPixelUpscaling(scale_factor=2)(x)
    return x
