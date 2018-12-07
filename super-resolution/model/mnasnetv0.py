import tensorflow as tf
import ops

#TODO: Extend to Multi-scale Network
class SingleMnasNetV0(tf.keras.Model):
"""Mnas Network baseline model using VDSR architecture

Arguments:
"""
    def __init__(self, args, scale):
        super(SingleMnasNetV0, self).__init__()

        self.upsample_type = args.upsample_type
        self.conv_type = args.conv_type

        self.conv_head= tf.keras.layers.Conv2D(args.num_filters,
                                        (3,3),
                                        padding='same',
                                        kernel_regularizer=l2(args.weight_decay))
        self.conv_blocks = []

        for _ in num_blocks:
            self.conv_blocks.append(create_conv_block(args))

        self.upsample_block = create_upsample_block(args)
        self.conv_body = tf.keras.layers.Conv2D(args.num_filters,
                                        (3,3),
                                        padding='same',
                                        kernel_regularizer=l2(args.weight_decay))
        self.conv_tail = tf.keras.layers.Conv2D(3,
                                        (3,3),
                                        padding='same',
                                        kernel_regularizer=l2(args.weight_decay))

    def get_model_name(args):
        name = ''

        name += args.conv_type
        name += '_'
        name += args.upsample_type
        name += '_'
        name += 'F{}'.format(args.num_filters)
        #TODO: Start from here

    def create_conv_block(args):
        assert args.conv_type in ['residual', 'depthwise_v1', 'depthwise_v2', 'depthwise_v3']

        if args.conv_type == 'residual':
            return Resblock(num_filters=args.num_filters,
                            weight_decay=args.weight_decay,
                            add_act=args.add_act_conv,
                            max_act=args.max_act)
        elif args.conv_type == 'depthwise_v1':
            return DSResConvBlock_v1(num_filters=args.num_filters,
                                    num_multiplier=1,
                                    weight_decayargs.weight_decay,
                                    add_act=args.add_act_conv,
                                    max_act=args.max_act)
        elif args.conv_type == 'depthwise_v2':
            return DSResConvBlock_v2(num_filters=args.num_filters,
                                    num_multiplier=1,
                                    expand_factor=args.expand_factor,
                                    weight_decayargs.weight_decay,
                                    add_act=args.add_act_conv,
                                    max_act=args.max_act)
        elif args.conv_type == 'depthwise_v3':
            return DSResConvBlock_v3(num_filters=args.num_filters,
                                    num_multiplier=1,
                                    weight_decayargs.weight_decay,
                                    add_act=args.add_act_conv,
                                    max_act=args.max_act)

    def create_upsample_block(args):
        assert self.upsample_type in ['tranpose', 'subpixel', 'resize_blinear', 'resize_nearest']


        if upsample_type == 'transpose':
            return TransposeConvBlock(num_filters=args.num_filters,
                                    scale=args.scale,
                                    weight_decay=args.weight_decay,
                                    add_act=args.add_act_upsample,
                                    max_act=args.max_act)
        elif upsample_type == 'subpixel':
            return SubpixelConvBlock(num_filters=args.num_filters,
                                    scale=args.scale,
                                    weight_decay=args.weight_decay,
                                    add_act=args.add_act_upsample,
                                    max_act=args.max_act)
        elif upsample_type == 'resize_bilinear':
            return ResizeBlock(num_filters=args.num_filters,
                                    scale=args.scale,
                                    weight_decay=args.weight_decay,
                                    add_act=args.add_act_upsample,
                                    max_act=args.max_act
                                    interpolation=args.interpolation)
        elif upsample_type == 'resize_nearest':
            return ResizeBlock(num_filters=args.num_filters,
                                    scale=args.scale,
                                    weight_decay=args.weight_decay,
                                    add_act=args.add_act_upsample,
                                    max_act=args.max_act
                                    interpolation=args.interpolation)

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
