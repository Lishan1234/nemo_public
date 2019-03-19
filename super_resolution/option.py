import argparse
import os

parser = argparse.ArgumentParser(description='MnasNet')

"""deprecated
parser.add_argument('--conv_type', type=str, default='residual', choices=('residual', 'depthwise_v1', 'depthwise_v2', 'depthwise_v3'), help='type of convlution block')
parser.add_argument('--weight_decay', type=float, default=1e-04)
parser.add_argument('--model_name', type=str, default='mnasnet_v0')
parser.add_argument('--load_on_memory', action='store_true')
"""

#Network architecture
parser.add_argument('--model_type', type=str, default='edsr',
                    #choices=('edsr, mobilenetv1, mobilenetv2, shufflenetv2'),
                    help='type of model')
parser.add_argument('--upsample_type', type=str, default='subpixel',
                    #choices=('transpose', 'subpixel', 'resize_bilinear', 'resize_nearest'),
                    help='type of upsample block')
parser.add_argument('--num_blocks', type=int, default=4, help='number of convolution blocks')
parser.add_argument('--num_filters', type=int, default=32, help='number of convolution filters')
parser.add_argument('--scale', type=int, default=4, help='super-resolution scale') #TODO: extend to multi-scale
parser.add_argument('--max_relu', type=float, default=None)
parser.add_argument('--data_format', type=str, default='channels_last', choices=('channels_first', 'channels_last', None))
parser.add_argument('--channel_in', type=int, default=3)

#Network architecture for building static graphs (pb or h5)
parser.add_argument('--hwc', type=str, default=None)

#Network architecture (MobileNet v2)
parser.add_argument('--expand_factor', type=int, default=6)

#Network architecture (EDSR v2)
parser.add_argument('--num_reduced_filters', type=int, default=32)
parser.add_argument('--num_reduced_kernels', type=int, default=3)


#Train
parser.add_argument('--lr_init', type=float, default=1e-04)
parser.add_argument('--num_batch', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--num_batch_per_epoch', type=int, default=1000)
parser.add_argument('--loss_type', type=str, default='l1',
                    choices=('l1', 'l2'))
parser.add_argument('--lr_decay_rate', type=float, default=0.5)
parser.add_argument('--lr_decay_epoch', type=int, default=100)

#Test
parser.add_argument('--lr', type=int, default=270) #deprecated
parser.add_argument('--hr', type=int, default=1080)

#Directory
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_dir', type=str, default='data')
#parser.add_argument('--data_dir', type=str, default='/ssd1')
parser.add_argument('--log_dir', type=str, default='log', help='Tensorboard loggin directory')

#Data
parser.add_argument('--num_patch', type=int, default=50000)
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument('--train_data', type=str, default='starcraft1')
parser.add_argument('--valid_data', type=str, default='starcraft1')
parser.add_argument('--data_type', type=str, default='keyframe')
parser.add_argument('--bitrate', type=str, default=None, help='Used for I-frame compression exp.')

#Hardware
parser.add_argument('--gpu_idx', type=int, default=0)

#Configuration
parser.add_argument('--custom_name', type=str, default=None, help='additional model name')
parser.add_argument('--num_sample', type=int, default=5)
parser.add_argument('--use_random_weights', action='store_true', default=5)

args = parser.parse_args()
if args.hwc is not None:
    args.hwc = list(map(lambda x: int(x), args.hwc.split(',')))

os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
