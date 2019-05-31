import argparse
import os

parser = argparse.ArgumentParser(description='MnasNet')

#Network architecture parameters
parser.add_argument('--model_type', type=str, default='edsr', help='type of model')
parser.add_argument('--upsample_type', type=str, default='transpose', help='type of upsample block')
parser.add_argument('--num_blocks', type=int, default=4, help='number of convolution blocks')
parser.add_argument('--num_filters', type=int, default=32, help='number of convolution filters')
parser.add_argument('--scale', type=int, default=4, help='super-resolution scale')
#TODO: extend to multi-scale
parser.add_argument('--max_relu', type=float, default=None)
parser.add_argument('--data_format', type=str, default='channels_last', choices=('channels_first', 'channels_last', None))
parser.add_argument('--channel_in', type=int, default=3)
parser.add_argument('--hwc', type=str, default=None)

#Network architecture parameters (MobileNet_v2)
parser.add_argument('--expand_factor', type=int, default=6)

#Network architecture parameters (EDSR_v2, EDSR_v9)
parser.add_argument('--num_reduced_filters', type=int, default=32)
parser.add_argument('--num_reduced_kernels', type=int, default=3)
parser.add_argument('--mode', type=int, default=3, choices=(0,1,2,3), help='operation mode for configuring edsr_v2')
parser.add_argument('--num_extra_blocks', type=int, default=1)
parser.add_argument('--num_extra_filters', type=int, default=16)

#Training parameters
parser.add_argument('--lr_init', type=float, default=1e-04)
parser.add_argument('--lr_decay_rate', type=float, default=0.5)
parser.add_argument('--lr_decay_epoch', type=int, default=200)
parser.add_argument('--num_batch', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--num_batch_per_epoch', type=int, default=1000)
parser.add_argument('--loss_type', type=str, default='l1',
                    choices=('l1', 'l2'))

#Directory
#parser.add_argument('--data_dir', type=str, default='/ssd1')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_dir', type=str, default='/ssd1/data')
parser.add_argument('--log_dir', type=str, default='log', help='Tensorboard loggin directory')

#Dataset
#parser.add_argument('--num_patch', type=int, default=50000)
#parser.add_argument('--patch_size', type=int, default=48)
#parser.add_argument('--data_type', type=str, default='keyframe')
#parser.add_argument('--lr', type=int, default=240) #deprecated
#parser.add_argument('--hr', type=int, default=960)
parser.add_argument('--original_resolution', type=int, default=2160)
parser.add_argument('--target_resolution', type=int, default=1080)
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--train_datatype', type=str, required=True)
parser.add_argument('--valid_data', type=str, default=None)
parser.add_argument('--valid_datatype', type=str, default=None)

#Dataset (I-frame compression)
parser.add_argument('--bitrate', type=str, default=None, help='Used for I-frame compression exp.')

#Hardware
parser.add_argument('--gpu_idx', type=int, default=0)

#Custom configuration (Debugging)
parser.add_argument('--custom_name', type=str, default=None, help='additional model name')
parser.add_argument('--num_sample', type=int, default=5)

#SDK (Huawei HiAI, Qualcomm SNPE)
parser.add_argument('--hiai_project_root', type=str, default='../hiai')
parser.add_argument('--snpe_project_root', type=str, default='../snpe')
parser.add_argument('--snpe_tensorflow_root', type=str, default='../../tensorflow')
parser.add_argument('--snpe_device_serial', type=str, default=None, help='Used for adb commands')
parser.add_argument('--snpe_copy_lib', action='store_true')
parser.add_argument('--snpe_copy_data', action='store_true')

args = parser.parse_args()
if args.hwc is not None:
    args.hwc = list(map(lambda x: int(x), args.hwc.split(',')))

if args.valid_data == None:
    args.valid_data = args.train_data

if args.valid_datatype == None:
    args.valid_datatype = args.train_datatype

os.makedirs(args.log_dir, exist_ok=True)
