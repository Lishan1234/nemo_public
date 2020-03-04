import argparse
import sys

from tool.adb import *
from tool.snpe import *
from tool.video import *
from dnn.model.nemo_s import NEMO_S
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    # parser.add_argument('--hr_video_name', type=str, required=True)

    #dataset
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #model
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)
    #parser.add_argument('--runtime', type=str, required=True)

    #cache profile
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], default='nemo')
    parser.add_argument('--threshold', type=float, required=True)

    #codec
    parser.add_argument('--threads', type=str, default=4)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    # hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    print(lr_video_file)
    assert(os.path.exists(lr_video_file))
    # assert(os.path.exists(hr_video_file))

    #load a dnn
    lr_video_profile = profile_video(lr_video_file)
    # hr_video_profile = profile_video(hr_video_file)
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]
    # scale = hr_video_profile['height'] // lr_video_profile['height']
    scale = 1080 // lr_video_profile['height']

    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)
    model = nemo_s.build_model(apply_clip=True)
    model.nhwc = nhwc
    model.scale = scale
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), model.name)
    assert(os.path.exists(checkpoint_dir))

    #setup directory
    content = os.path.basename(args.dataset_dir)
    device_root_dir = os.path.join('/sdcard/NEMO', content)
    device_video_dir = os.path.join(device_root_dir, 'video')
    device_checkpoint_dir = os.path.join(device_root_dir, 'checkpoint', args.lr_video_name, model.name)
    device_cache_profile_dir = os.path.join(device_root_dir, 'profile', args.lr_video_name, model.name)
    adb_mkdir(device_video_dir, args.device_id)
    adb_mkdir(device_checkpoint_dir, args.device_id)
    adb_mkdir(device_cache_profile_dir, args.device_id)

    #setup videos
    adb_push(device_video_dir, lr_video_file, args.device_id)

    #setup a dnn
    dlc_dict = snpe_convert_model(model, model.nhwc, checkpoint_dir)
    dlc_path = os.path.join(checkpoint_dir, dlc_dict['dlc_name'])
    adb_push(device_checkpoint_dir, dlc_path,args.device_id)

    #setup a cache profile
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random
    cache_profile = os.path.join(args.dataset_dir, 'profile', args.lr_video_name, model.name, 'NEMO_0.5.profile')
    print(cache_profile)
    adb_push(device_cache_profile_dir, cache_profile, args.device_id)
