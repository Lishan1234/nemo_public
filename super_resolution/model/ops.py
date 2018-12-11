import sys
import tensorflow as tf
import tensorflow.keras.backend as K

l2 = tf.keras.regularizers.l2

class ConvBlock(tf.keras.Model):
    """Convolution Block consisting of (conv->relu)

    Arguments:
        num_filters:
        weight_decay:
        add_act:
        max_act:
    """
    def __init__(self, num_filters, weight_decay, add_act, max_act, data_format='channels_first'):
        super(ConvBlock, self).__init__()
        self.add_act = add_act

        self.conv1 = tf.keras.layers.Conv2D(num_filters,
                                            (3,3),
                                            padding='same',
                                            kernel_regularizer=l2(weight_decay),
                                            data_format=data_format)
        if self.add_act:
            self.relu1 = tf.keras.layers.ReLU(max_value=max_act)

    def call(self, x):
        output = self.conv1(x)

        if self.add_act:
            output = self.relu1(output)

        return output

class ResBlock(tf.keras.Model):
    """Residual Convolution Block consisting of (conv->relu->(conv)->(relu)) from EDSR

    Arguments:
        num_filters:
        weight_decay:
        add_act:
        max_act
    """
    def __init__(self, num_filters, weight_decay, add_act, max_act, data_format='channels_first'):
        super(ResBlock, self).__init__()
        self.conv_block1 = ConvBlock(num_filters, weight_decay, True, max_act, data_format)
        self.conv_block2 = ConvBlock(num_filters, weight_decay, add_act, max_act, data_format)

    def call(self, x):
        input = x
        output = self.conv_block1(x)
        output = self.conv_block2(output)
        output = input + output

        return output

class DSConvBlock_v1(tf.keras.Model):
    """Depthwise Seperable Convolution Block (d-conv->relu->p-conv->(relu)) from MobileNet.v1

    Arguments:
        num_filters:
        num_multiplier:
        weight_decay:
        add_act:
        max_act
    """
    def __init__(self, num_filters, num_multiplier, weight_decay, add_act, max_act, data_format='channels_first'):
        super(DSConvBlock_v1, self).__init__()
        self.add_act = add_act

        self.dwconv1 = tf.keras.layers.DepthwiseConv2D((3,3),
                                            padding='same',
                                            depth_multiplier=num_multiplier,
                                            depthwise_regularizer=l2(weight_decay),
                                            data_format=data_format)
        self.relu1 = tf.keras.layers.ReLU(max_value=max_act)
        self.pwconv1 = tf.keras.layers.Conv2D(num_filters,
                                            (1,1),
                                            padding='same',
                                            kernel_regularizer=l2(weight_decay),
                                            data_format=data_format)

        if self.add_act:
            self.relu2 = tf.keras.layers.ReLU(max_value=max_act)

    def call(self, x):
        output = self.dwconv1(x)
        output = self.relu1(output)
        output = self.pwconv1(output)

        if self.add_act:
            output = self.relu2(output)

        return output

class DSResConvBlock_v1(tf.keras.Model):
    """Depthwise Seperable Convolution Block (d-conv->relu->p-conv->(relu)) from MobileNet.v1

    Arguments:
        num_filters:
        num_multiplier:
        weight_decay:
        add_act:
        max_act:
    """
    def __init__(self, num_filters, num_multiplier, weight_decay, add_act, max_act, data_format='channels_first'):
        super(DSResConvBlock_v1, self).__init__()
        self.dsconv_block1 = DSConvBlock_v1(num_filters, num_multiplier, weight_decay, True, max_act, data_format)
        self.dsconv_block2 = DSConvBlock_v1(num_filters, num_multiplier, weight_decay, add_act, max_act, data_format)

    def call(self, x):
        input = x
        output = self.dsconv_block1(x)
        output = self.dsconv_block2(output)
        output = input + output

        return output

class DSResConvBlock_v2(tf.keras.Model):
    """Depthwise Seperable Convolution Block (p-conv->relu->d-conv->p-conv->(relu)) from MobileNet.v2

    Arguments:
        num_filters:
        num_multiiplier:
        expand_factor:
        weight_decay:
        add_act:
        max_act:
    """
    def __init__(self, num_filters, num_multiplier, expand_factor, weight_decay, add_act, max_act, data_format='channels_first'):
        super(DSResConvBlock_v2, self).__init__()
        self.add_act = add_act

        self.pwconv1 = tf.keras.layers.Conv2D(num_filters * expand_factor,
                                            (1,1),
                                            padding='same',
                                            kernel_regularizer=l2(weight_decay),
                                            data_format=data_format)
        self.relu1 = tf.keras.layers.ReLU(max_value=max_act)
        self.dwconv1 = tf.keras.layers.DepthwiseConv2D((3,3),
                                            padding='same',
                                            depth_multiplier=num_multiplier,
                                            depthwise_regularizer=l2(weight_decay),
                                            data_format=data_format)
        self.relu2 = tf.keras.layers.ReLU(max_value=max_act)
        self.pwconv2 = tf.keras.layers.Conv2D(num_filters,
                                            (1,1),
                                            padding='same',
                                            kernel_regularizer=l2(weight_decay),
                                            data_format=data_format)
        if self.add_act:
            self.relu3 = tf.keras.layers.ReLU(max_value=max_act)

    def call(self, x):
        input = x
        output = self.pwconv1(x)
        output = self.relu1(output)
        output = self.dwconv1(output)
        output = self.relu2(output)
        output = self.pwconv2(output)

        if self.add_act:
            output = self.relu3(output)

        output = input + output

        return output

def _split_func(x):
    x_left, x_right = tf.split(x, num_or_size_splits=2, axis=3)
    return x_left, x_right

def _concat_shuffle_func(x):
    x_left = x[0]
    x_right = x[1]
    shape = x_left.get_shape().as_list()
    z = tf.stack([x_left, x_right], axis=3)
    z = tf.transpose(z, [0,1,2,4,3])
    z = tf.reshape(z, [shape[0], shape[1], shape[2], 2*shape[3]])
    return z


class DSResConvBlock_v3(tf.keras.Model):
    """Depthwise Seperable Convolution Block (split->p-conv->relu->d-conv->p-conv->(relu)->concat->shuffle) from ShuffleNet.v2

    Arguments:
        num_filters:
        num_multiiplier:
        weight_decay:
        add_act:
        max_act:
    """

    def __init__(self, num_filters, num_multiplier, weight_decay, add_act, max_act, data_format='channels_first'):
        super(DSResConvBlock_v3, self).__init__()
        self.add_act = add_act

        self.split1 = tf.keras.layers.Lambda(_split_func)
        self.pwconv1 = tf.keras.layers.Conv2D(num_filters // 2,
                                            (1,1),
                                            padding='same',
                                            kernel_regularizer=l2(weight_decay),
                                            data_format=data_format)
        self.relu1 = tf.keras.layers.ReLU(max_value=max_act)
        self.dwconv1 = tf.keras.layers.DepthwiseConv2D((3,3),
                                            padding='same',
                                            depth_multiplier=num_multiplier,
                                            depthwise_regularizer=l2(weight_decay),
                                            data_format=data_format)
        self.pwconv2 = tf.keras.layers.Conv2D(num_filters // 2,
                                            (1,1),
                                            padding='same',
                                            kernel_regularizer=l2(weight_decay),
                                            data_format=data_format)
        self.relu2 = tf.keras.layers.ReLU(max_value=max_act)

        if self.add_act:
            self.relu3 = tf.keras.layers.ReLU(max_value=max_act)

        self.concat_shuffle1 = tf.keras.layers.Lambda(_concat_shuffle_func)

    def call(self, x):
        x_left, x_right = self.split1(x)
        output_right = self.pwconv1(x_right)
        output_right = self.relu1(output_right)
        output_right = self.dwconv1(output_right)
        output_right = self.pwconv2(output_right)
        output_right = self.relu2(output_right)

        if self.add_act:
            output_right = self.relu3(output_right)

        output = self.concat_shuffle1([x_left, output_right])

        return output

class TransposeConvBlock(tf.keras.Model):
    """ TransposeConvBlock for x2,x3,x4 upscale

    Arguments:
        num_filters:
        scale:
        weight_decay:
        add_act:
        max_act:
    """

    def __init__(self, num_filters, scale, weight_decay, add_act, max_act, data_format='channels_first'):
        super(TransposeConvBlock, self).__init__()
        assert scale in [2,3,4]
        self.add_act = add_act

        self.tconv1 = tf.keras.layers.Conv2DTranspose(num_filters,
                                                    (3,3),
                                                    (scale,scale),
                                                    padding='same',
                                                    kernel_regularizer=l2(weight_decay),
                                                    data_format)

        if self.add_act:
            self.relu1 = tf.keras.layers.ReLU(max_value=max_act)

    def call(self, x):
        output = self.tconv1(x)

        if self.add_act:
            output = self.relu1(output)

        return output

def _subpixel_func(x):
    input = x[0]
    scale = x[1]
    return tf.depth_to_space(input, scale)

class SubpixelConvBlock(tf.keras.Model):
    """ SubpixelConvBlock for x2,x3,x4 upscale

    Arguments:
        num_filters:
        scale:
        weight_decay:
        add_act:
        max_act:
    """

    def __init__(self, num_filters, scale, weight_decay, add_act, max_act, data_format='channels_first'):
        super(SubpixelConvBlock, self).__init__()
        assert scale in [2,3,4]

        self.add_act = add_act
        self.scale = scale

        if self.scale in [2,3]:
            self.conv1 = tf.keras.layers.Conv2D(num_filters * (self.scale ** 2),
                                                (3,3),
                                                padding='same',
                                                kernel_regularizer=l2(weight_decay),
                                                data_format=data_format)
            self.sconv1= tf.keras.layers.Lambda(_subpixel_func)
        elif self.scale in [4]:
            self.conv1 = tf.keras.layers.Conv2D(num_filters * (2 ** 2),
                                                (3,3),
                                                padding='same',
                                                kernel_regularizer=l2(weight_decay),
                                                data_format=data_format)
            self.sconv1= tf.keras.layers.Lambda(_subpixel_func)
            self.conv2 = tf.keras.layers.Conv2D(num_filters * (2 ** 2),
                                                (3,3),
                                                padding='same',
                                                kernel_regularizer=l2(weight_decay),
                                                data_format=data_format)
            self.sconv2= tf.keras.layers.Lambda(_subpixel_func)


        if self.add_act:
            self.relu1 = tf.keras.layers.ReLU(max_value=max_act)

    def call(self, x):
        if self.scale in [2,3]:
            output = self.conv1(x)
            output = self.sconv1([output, self.scale])
        elif self.scale in [4]:
            output = self.conv1(x)
            output = self.sconv1([output, self.scale / 2])
            output = self.conv2(output)
            output = self.sconv2([output, self.scale / 2])

        if self.add_act:
            output = self.relu1(output)

        return output

#Caution: Not supported on Tensorflow Lite
def _resize_bilinear_func(x, scale):
    shape = x.get_shape().as_list()
    return tf.image.resize_bilinear(x, (shape[1] * scale, shape[2] * scale))

#Caution: Not supported on Snapdragon DSP
def _resize_nearest_func(x, scale):
    shape = x.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(x, (shape[1] * scale, shape[2] * scale))

def _resize_func(x):
    input = x[0]
    scale = x[1]
    interpolation = x[2]

    if interpolation == 'bilinear':
        return _resize_bilinear_func(input, scale)
    elif interpolation == 'nearest':
        return _resize_nearest_func(input, scale)

class ResizeBlock(tf.keras.Model):
    """ ResizeBlock for x2,x3,x4 upscale

    Arguments:
        num_filters:
        scale:
        weight_decay:
        add_act:
        max_act:
        interpolation:
    """

    def __init__(self, num_filters, scale, weight_decay, add_act, max_act, interpolation):
        super(ResizeBlock, self).__init__()
        assert scale in [2,3,4]
        assert interpolation in ['bilinear', 'nearest']
        self.add_act = add_act
        self.scale = scale
        self.interpolation = interpolation

        self.resize1= tf.keras.layers.Lambda(_resize_func)

        if self.add_act:
            self.relu1 = tf.keras.layers.ReLU(max_value=max_act)


    def call(self, x):
        output = self.resize1([x, self.scale, self.interpolation])

        if self.add_act:
            output = self.relu1(output)

        return output

#Simple test script
if __name__ == "__main__":
    tf.enable_eager_execution()
    with tf.device('/gpu:0'):
        x = tf.random_normal([1, 100, 100, 3])
        y = tf.random_normal([1, 100, 100, 16])

        """
        a = ConvBlock(16, 0, False, 6)
        b = ResBlock(16, 0, False, 6)
        c = DSConvBlock_v1(16, 1, 0, False, 6)
        d = DSResConvBlock_v1(16, 1, 0, False, 6)
        e = DSResConvBlock_v2(16, 1, 6, 0, False, 6)
        f = DSResConvBlock_v3(8, 1, 0, False, 6)

        print(a(y))
        print(b(y))
        print(c(y))
        print(d(y))
        print(e(y))
        print(f(y))
        """

        z = tf.random_normal([1, 100, 100, 32])
        g = TransposeConvBlock(32, 3, 0, False, 6)
        h = SubpixelConvBlock(32, 3, 0, False, 6)
        i  = ResizeBlock(32, 3, 0, False, 6, 'bilinear')
        i_  = ResizeBlock(32, 3, 0, False, 6, 'nearest')

        print(g(z).shape)
        print(h(z).shape)
        print(i(z).shape)
        print(i_(z).shape)
