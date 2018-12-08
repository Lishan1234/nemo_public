import argparse

parser = argparse.ArgumentParser(description='MnasNet')

#Network architecture
parser.add_argument('--conv_type', type=str, default='residual', choices=('residual', 'depthwise_v1', 'depthwise_v2', 'depthwise_v3'), help='type of convlution block')
parser.add_argument('--upsample_type', type=str, default='subpixel', choices=('transpose', 'subpixel', 'resize_bilinear', 'resize_nearest'), help='type of upsample block')
parser.add_argument('--num_blocks', type=int, default=3, help='number of convolution blocks')
parser.add_argument('--num_filters', type=int, default=32, help='number of convolution block filter')
parser.add_argument('--scale', type=int, default=3, help='super-resolution scale') #TODO: extend to multi-scale
parser.add_argument('--add_act_conv', action='store_true')
parser.add_argument('--add_act_upsample', action='store_true')
parser.add_argument('--max_act', type=float, default=None)

#Network architecture (MobileNet v2)
parser.add_argument('--expand_factor', type=int, default=6)

#Network-Training
parser.add_argument('--weight_decay', type=float, default=1e-04, help='number of convolution block filter')

#Training

#Configuration
parser.add_argument('--custom_name', type=str, default=None, help='additional model name')

args = parser.parse_args()
