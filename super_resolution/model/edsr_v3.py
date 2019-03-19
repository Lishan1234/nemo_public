import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from model import ops

def make_model(args):
    return EDSR_v3(args)

class EDSR_v3:
    def __init__(self, args):
        self.num_blocks = args.num_blocks
        self.num_filters = args.num_filters
        self.data_format = args.data_format
        self.max_relu = args.max_relu
        self.scale = args.scale
        self.hwc = args.hwc
        self.upsample_type = args.upsample_type
        self.custom_name = args.custom_name
        self.channel_in = args.channel_in
        self.bitrate = args.bitrate

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
        name += 'S{}'.format(self.scale)

        if self.bitrate is not None:
            name += '_'
            name += 'BR{}'.format(self.bitrate)

        if self.custom_name is not None:
            name += '_'
            name += '[{}]'.format(self.custom_name)

        return name

    def _build_upsample(self, x):
        if self.upsample_type == 'transpose':
            return ops.transpose_upsample(x, self.scale, self.channel_in, self.data_format) #compact version
        elif self.upsample_type == 'subpixel':
            return ops.subpixel_upsample(x, self.scale, self.channel_in, self.data_format) #compact version
        elif self.upsample_type == 'resize_bilinear':
            x = ops.bilinear_upsample(x, self.scale, self.data_format)
            x = layers.Conv2D(self.channel_in,
                                        (1,1),
                                        padding='same',
                                        data_format=self.data_format)(x)
            return x
        elif self.upsample_type == 'resize_nearest':
            x = ops.nearest_upsample(x, self.scale, self.data_format)
            x = layers.Conv2D(self.channel_in,
                                        (1,1),
                                        padding='same',
                                        data_format=self.data_format)(x)
            return x
        raise NotImplementedError

    def build(self):
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
                                        max_relu=self.max_relu,
                                        data_format=self.data_format)

        outputs = layers.Conv2D(self.num_filters,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(outputs)
        outputs = layers.Add()([outputs, res])
        predictions = self._build_upsample(outputs)
        """
        predictions = layers.Conv2D(self.channel_in,
                                        (3,3),
                                        padding='same',
                                        data_format=self.data_format)(outputs)
        """

        model = Model(inputs=inputs, outputs=predictions)
        #model.summary()

        return model
