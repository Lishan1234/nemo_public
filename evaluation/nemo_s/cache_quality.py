import argparse
import os
import math
import glob

import tensorflow as tf

from tool.video import profile_video, FFmpegOption
from tool.libvpx import *
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_file', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #dataset
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #config
    parser.add_argument('--setup_image', action='store_true')
    parser.add_argument('--remove_image', action='store_true')

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    args = parser.parse_args()

    #scale, nhwc
    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    lr_video_profile = profile_video(lr_video_file)
    hr_video_profile = profile_video(hr_video_file)
    scale = int(hr_video_profile['height'] / lr_video_profile['height'])
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]

    #model (restore)
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), nemo_s.name)
    checkpoint = nemo_s.load_checkpoint(checkpoint_dir)
    checkpoint.model.scale = scale
    checkpoint.model.nhwc = nhwc

    #cache profile
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random
    cache_profile_file = os.path.join(args.dataset_dir, 'profile', args.lr_video_name, checkpoint.model.name,
                                        '{}_{}.profile'.format(aps_class.NAME1, args.threshold))
    print(cache_profile_file)

    start_time = time.time()
    if args.setup_image:
        libvpx_save_frame(args.vpxdec_file, args.dataset_dir, args.lr_video_name)
        libvpx_save_frame(args.vpxdec_file, args.dataset_dir, args.hr_video_name)
        libvpx_setup_sr_frame(args.vpxdec_file, args.dataset_dir, args.lr_video_name, checkpoint.model)
    end_time = time.time()
    print('saving image takes {}sec'.format(end_time-start_time))

    start_time = time.time()
    libvpx_offline_cache_quality(args.vpxdec_file, args.dataset_dir, args.lr_video_name, args.hr_video_name, \
            checkpoint.model.name, cache_profile_file, lr_video_profile['height'])
    end_time = time.time()
    print('measuring sr quality takes {}sec'.format(end_time-start_time))

    start_time = time.time()
    if args.remove_image:
        lr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name)
        lr_image_files = glob.glob(os.path.join(lr_image_dir, '*.raw'))
        sr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name, checkpoint.model.name)
        sr_image_files = glob.glob(os.path.join(sr_image_dir, '*.raw'))
        hr_image_dir = os.path.join(args.dataset_dir, 'image', args.hr_video_name)
        hr_image_files = glob.glob(os.path.join(hr_image_dir, '*.raw'))

        for lr_image_file, sr_image_file, hr_image_file in zip(lr_image_files, sr_image_files, hr_image_files):
            os.remove(lr_image_file)
            os.remove(sr_image_file)
            os.remove(hr_image_file)
    end_time = time.time()
    print('removing image takes {}sec'.format(end_time-start_time))
