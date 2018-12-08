import tensorflow as tf
import ops
from option import *

l2 = tf.keras.regularizers.l2

def create_conv_block(args):
    if args.conv_type == 'residual':
        return ops.ResBlock(num_filters=args.num_filters,
                        weight_decay=args.weight_decay,
                        add_act=args.add_act_conv,
                        max_act=args.max_act)
    elif args.conv_type == 'depthwise_v1':
        return ops.DSResConvBlock_v1(num_filters=args.num_filters,
                                num_multiplier=1,
                                weight_decay=args.weight_decay,
                                add_act=args.add_act_conv,
                                max_act=args.max_act)
    elif args.conv_type == 'depthwise_v2':
        return ops.DSResConvBlock_v2(num_filters=args.num_filters,
                                num_multiplier=1,
                                expand_factor=args.expand_factor,
                                weight_decay=args.weight_decay,
                                add_act=args.add_act_conv,
                                max_act=args.max_act)
    elif args.conv_type == 'depthwise_v3':
        return ops.DSResConvBlock_v3(num_filters=args.num_filters,
                                num_multiplier=1,
                                weight_decay=args.weight_decay,
                                add_act=args.add_act_conv,
                                max_act=args.max_act)

def create_upsample_block(args, scale):
    if args.upsample_type == 'transpose':
        return ops.TransposeConvBlock(num_filters=args.num_filters,
                                scale=scale,
                                weight_decay=args.weight_decay,
                                add_act=args.add_act_upsample,
                                max_act=args.max_act)
    elif args.upsample_type == 'subpixel':
        return ops.SubpixelConvBlock(num_filters=args.num_filters,
                                scale=scale,
                                weight_decay=args.weight_decay,
                                add_act=args.add_act_upsample,
                                max_act=args.max_act)
    elif args.upsample_type == 'resize_bilinear':
        return ops.ResizeBlock(num_filters=args.num_filters,
                                scale=scale,
                                weight_decay=args.weight_decay,
                                add_act=args.add_act_upsample,
                                max_act=args.max_act,
                                interpolation=args.interpolation)
    elif args.upsample_type == 'resize_nearest':
        return ops.ResizeBlock(num_filters=args.num_filters,
                                scale=scale,
                                weight_decay=args.weight_decay,
                                add_act=args.add_act_upsample,
                                max_act=args.max_act,
                                interpolation=args.interpolation)

class SingleMnasNetV0(tf.keras.Model):
    """Mnas Network baseline model using VDSR architecture
    """
    def __init__(self, args, scale):
        super(SingleMnasNetV0, self).__init__()
        self.args = args
        self.scale = scale

        self.conv_head= tf.keras.layers.Conv2D(self.args.num_filters,
                                        (3,3),
                                        padding='same',
                                        kernel_regularizer=l2(self.args.weight_decay))
        self.conv_blocks = []

        for _ in range(self.args.num_blocks):
            self.conv_blocks.append(create_conv_block(self.args))

        self.upsample_block = create_upsample_block(self.args, self.scale)
        self.conv_body = tf.keras.layers.Conv2D(self.args.num_filters,
                                        (3,3),
                                        padding='same',
                                        kernel_regularizer=l2(self.args.weight_decay))
        self.conv_tail = tf.keras.layers.Conv2D(3,
                                        (3,3),
                                        padding='same',
                                        kernel_regularizer=l2(self.args.weight_decay))

    def get_name(self):
        name = ''

        name += self.args.conv_type
        name += '_'
        name += self.args.upsample_type
        name += '_'
        name += 'B{}'.format(self.args.num_blocks)
        name += '_'
        name += 'F{}'.format(self.args.num_filters)
        name += '_'
        name += 'S{}'.format(self.scale)

        if self.args.conv_type == 'depthwise_v2':
            name += '_'
            name += 'E{}'.format(self.args.expand_factor)

        if self.args.custom_name is not None:
            name += '_'
            name += '[{}]'.format(self.args.expand_factor)

        return name

    def call(self, x):
        x_ = self.conv_head(x)

        #ConvBlocks
        output = x_
        for conv_block in self.conv_blocks:
            output = conv_block(output)
        output = self.conv_body(output)

        #Residual connection
        output = x_ = output

        #Upsample
        output = self.upsample_block(output)
        output = self.conv_tail(output)

        return output

#Simple test script
if __name__ == "__main__":
    tf.enable_eager_execution()
    with tf.device('/gpu:0'):
        x = tf.random_normal([1, 100, 100, 3])
        model = SingleMnasNetV0(args, 3)
        print(model.get_name())
        print(model(x).shape)
