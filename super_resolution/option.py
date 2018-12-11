import argparse
import os

parser = argparse.ArgumentParser(description='MnasNet')

#Data & Model
parser.add_argument('--model_name', type=str, default='mnasnet_v0')
parser.add_argument('--data_name', type=str, default='bigbuckbunny_v0')

#Network architecture
parser.add_argument('--conv_type', type=str, default='residual', choices=('residual', 'depthwise_v1', 'depthwise_v2', 'depthwise_v3'), help='type of convlution block')
parser.add_argument('--upsample_type', type=str, default='subpixel', choices=('transpose', 'subpixel', 'resize_bilinear', 'resize_nearest'), help='type of upsample block')
parser.add_argument('--num_blocks', type=int, default=8, help='number of convolution blocks')
parser.add_argument('--num_filters', type=int, default=32, help='number of convolution block filter')
parser.add_argument('--scale', type=int, default=4, help='super-resolution scale') #TODO: extend to multi-scale
parser.add_argument('--add_act_conv', action='store_true')
parser.add_argument('--add_act_upsample', action='store_true')
parser.add_argument('--max_act', type=float, default=None)
parser.add_argument('--data_format', type=str, default='channels_first', choices=('channels_first', 'channels_last'))

#Network architecture (MobileNet v2)
parser.add_argument('--expand_factor', type=int, default=6)

#Training
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--weight_decay', type=float, default=1e-04)
parser.add_argument('--num_batch', type=int, default=16)
parser.add_argument('--num_epoch', type=int, default=150)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--load_on_memory', action='store_true')
parser.add_argument('--num_batch_per_epoch', type=int, default=1000)
parser.add_argument('--loss_type', type=str, default='l1')
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--lr_decay_epoch', type=int, default=100)

#Directory
parser.add_argument('--data_dir', type=str, default='dataset/data')
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--board_dir', type=str, default='./tb/')

#Dataset (bigbuckbunny_v0)
parser.add_argument('--fps', type=float, default=0.1)
parser.add_argument('--num_patch', type=int, default=10000)
#parser.add_argument('--use_tfrecord', action='store_true')

#Hardware
parser.add_argument('--gpu_idx', type=int, default=0)

#Configuration
parser.add_argument('--custom_name', type=str, default=None, help='additional model name')

args = parser.parse_args()

os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.board_dir, exist_ok=True)
