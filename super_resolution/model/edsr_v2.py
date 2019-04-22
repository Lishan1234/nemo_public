import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from model import ops

def make_model(args):
    return EDSR_v2(args)

class EDSR_v2:
    def __init__(self, args):
        self.num_blocks = args.num_blocks
        self.num_filters = args.num_filters
        self.num_reduced_filters = args.num_reduced_filters
        self.data_format = args.data_format
        self.scale = args.scale
        self.hwc = args.hwc
        self.upsample_type = args.upsample_type
        self.custom_name = args.custom_name
        self.channel_in = args.channel_in
        self.bitrate = args.bitrate
        self.mode = args.mode

    def get_name(self):
        name = ''

        name += type(self).__name__
        name += '_'
        name += self.upsample_type
        name += '_'
        name += 'B{}'.format(self.num_blocks)
        name += '_'
        name += 'F{}'.format(self.num_filters)
        name += '_'
        name += 'RF{}'.format(self.num_reduced_filters)
        name += '_'
        name += 'S{}'.format(self.scale)

        if self.bitrate is not None:
            name += '_'
            name += 'BR{}'.format(self.bitrate)

        if self.custom_name is not None:
            name += '_'
            name += '[{}]'.format(self.custom_name)

        return name

    def _build_feature_reduction(self, x):
        if self.num_reduced_filters is not None:
            assert isinstance(self.num_reduced_filters, int)
            x = layers.Conv2D(self.num_reduced_filters,
                                (1,1),
                                padding='same',
                                data_format=self.data_format)(x)
        return x

    def _build_feature_quantization(self, x):
        if self.mode == 0:
            x = ops.Quantize()(x)
        elif self.mode == 1:
            x = ops.Dequantize()(x)
        elif self.mode == 2:
            x = ops.Quantize()(x)
            x = ops.Dequantize()(x)

        return x

    def _build_upsample(self, x):
        if self.upsample_type == 'transpose':
            x = ops.transpose_upsample(x, self.scale, self.channel_in, self.data_format, 5) #compact version
        elif self.upsample_type == 'resize_bilinear':
            x = ops.bilinear_upsample(x, self.scale, self.data_format)
        else:
            raise NotImplementedError

        x = layers.Conv2D(self.channel_in,
                                    (3,3),
                                    padding='same',
                                    data_format=self.data_format,
                                    name='conv_upsample')(x)
        return x

    def build_preupsample(self):
        if self.hwc is not None:
            if self.data_format == 'channels_first':
                inputs = layers.Input(shape=(self.hwc[2], self.hwc[0], self.hwc[1]))
            else:
                inputs = layers.Input(shape=(self.hwc[0], self.hwc[1], self.hwc[2]))
        else:
            inputs = layers.Input(shape=(None, None, self.channel_in))

        outputs = layers.Conv2D(self.num_filters,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(inputs)
        res = outputs

        for _ in range(self.num_blocks):
            outputs = ops.residual_block(x=outputs,
                                        num_filters=self.num_filters,
                                        kernel_size=3,
                                        data_format=self.data_format)

        outputs = layers.Conv2D(self.num_filters,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(outputs)
        outputs = layers.Add()([outputs, res])

        outputs = self._build_feature_reduction(outputs)
        predictions = self._build_feature_quantization(outputs)

        model = Model(inputs=inputs, outputs=predictions)
        model.summary()

        return model

    def build_full(self):
        if self.hwc is not None:
            if self.data_format == 'channels_first':
                inputs = layers.Input(shape=(self.hwc[2], self.hwc[0], self.hwc[1]))
            else:
                inputs = layers.Input(shape=(self.hwc[0], self.hwc[1], self.hwc[2]))
        else:
            inputs = layers.Input(shape=(None, None, self.channel_in))

        outputs = layers.Conv2D(self.num_filters,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(inputs)
        res = outputs

        for _ in range(self.num_blocks):
            outputs = ops.residual_block(x=outputs,
                                        num_filters=self.num_filters,
                                        kernel_size=3,
                                        data_format=self.data_format)

        outputs = layers.Conv2D(self.num_filters,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(outputs)
        outputs = layers.Add()([outputs, res])

        outputs = self._build_feature_reduction(outputs)
        outputs = self._build_feature_quantization(outputs)
        predictions = self._build_upsample(outputs)

        model = Model(inputs=inputs, outputs=predictions)
        model.summary()

        return model

    #caution: input range should be 0-255
    def build_postupsample(self):
        """
        if self.hwc is not None:
            if self.data_format == 'channels_first':
                inputs = layers.Input(shape=(self.hwc[2], self.hwc[0], self.hwc[1]))
            else:
                inputs = layers.Input(shape=(self.hwc[0], self.hwc[1], self.hwc[2]))
        else:
            inputs = layers.Input(shape=(None, None, self.channel_in))

        outputs = self._build_feature_quantization(inputs)
        predictions = self._build_upsample(outputs)

        model = Model(inputs=inputs, outputs=predictions)
        model.summary()

        return model
        """
        if self.hwc is not None:
            if self.data_format == 'channels_first':
                inputs = layers.Input(shape=(self.hwc[2], self.hwc[0], self.hwc[1]))
            else:
                inputs = layers.Input(shape=(self.hwc[0], self.hwc[1], self.hwc[2]))
        else:
            inputs = layers.Input(shape=(None, None, self.channel_in))

        outputs = layers.Conv2D(self.num_filters,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(inputs)
        res = outputs

        for _ in range(self.num_blocks):
            outputs = ops.residual_block(x=outputs,
                                        num_filters=self.num_filters,
                                        kernel_size=3,
                                        data_format=self.data_format)

        outputs = layers.Conv2D(self.num_filters,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(outputs)
        outputs = layers.Add()([outputs, res])

        outputs = self._build_feature_reduction(outputs)
        outputs = self._build_feature_quantization(inputs)
        predictions = self._build_upsample(outputs)

        model = Model(inputs=inputs, outputs=predictions)
        model.summary()

        return model

    def build(self):
        if self.mode == 0:
            model = self.build_preupsample()
        elif self.mode == 1:
            model = self.build_postupsample()
        elif self.mode == 2 or self.mode == 3:
            model = self.build_full()

        return model
