import argparse
import sys

from tool.adb import *
from tool.snpe import *
from dnn.model.edsr_s import EDSR_S

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #model
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--nhwc', nargs='+', type=int, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    #codec
    parser.add_argument('--threads', type=str, default=1)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    #setup directory
    device_root_dir = '/storage/emulated/0/Android/data/android.example.testlibvpx/files'
    device_video_dir = os.path.join(device_root_dir, 'video')
    adb_mkdir(device_video_dir, args.device_id)

    #setup video
    lr_video_path = os.path.join(args.video_dir, args.lr_video_name)
    hr_video_path = os.path.join(args.video_dir, args.hr_video_name)
    adb_push(device_video_dir, lr_video_path, args.device_id)
    adb_push(device_video_dir, hr_video_path, args.device_id)

    #setup dnn
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, args.scale, None)
    model = edsr_s.build_model()
    checkpoint_dir = os.path.join(args.checkpoint_dir, model.name)
    dlc_dict = convert_model(model, args.nhwc, checkpoint_dir)

    device_checkpoint_dir = os.path.join(device_root_dir, 'checkpoint', model.name)
    adb_mkdir(device_checkpoint_dir, args.device_id)
    dlc_path = os.path.join(checkpoint_dir, dlc_dict['dlc_name'])
    adb_push(device_checkpoint_dir, dlc_path)
