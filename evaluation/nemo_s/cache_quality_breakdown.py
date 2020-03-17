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

assert(tf.__version__.startswith('2'))

#--vpxdec_file $MOBINAS_CODE_ROOT/third_party/libvpx/vpxdec \

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_dir', type=str, required=True)
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dataset
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    #configuration
    parser.add_argument('--mode', type=str, choices=['nemo', 'no_residual', 'no_mv', 'no_mv_residual'], required=True)

    args = parser.parse_args()

    #cache profile
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random

    for content in args.content:
        dataset_dir = os.path.join(args.dataset_rootdir, content)

        #scale, nhwc
        video_dir = os.path.join(dataset_dir, 'video')
        lr_video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.lr_resolution)))[0])
        lr_video_profile = profile_video(lr_video_file)
        lr_video_name = os.path.basename(lr_video_file)
        hr_video_file = os.path.abspath(glob.glob(os.path.join(video_dir, '{}p*'.format(args.hr_resolution)))[0])
        hr_video_profile = profile_video(hr_video_file)
        hr_video_name = os.path.basename(hr_video_file)
        scale = int(args.hr_resolution // args.lr_resolution)

        #mode
        if args.mode == 'nemo':
            postfix = None
            vpxdec_name = 'vpxdec'
        elif args.mode == 'no_mv':
            postfix = args.mode
            vpxdec_name = 'vpxdec_no_mv'
        elif args.mode == 'no_residual':
            postfix = args.mode
            vpxdec_name = 'vpxdec_no_residual'
        elif args.mode == 'no_mv_residual':
            postfix = args.mode
            vpxdec_name = 'vpxdec_no_mv_residual'
        else:
            raise NotImplementedError

        #setup lr, hr frames
        vpxdec_file = os.path.join(args.vpxdec_dir, vpxdec_name)
        start_time = time.time()
        libvpx_save_frame(vpxdec_file, dataset_dir, lr_video_name, postfix=postfix)
        libvpx_save_frame(vpxdec_file, dataset_dir, hr_video_name, postfix=postfix)
        end_time = time.time()
        print('saving lr, hr image takes {}sec'.format(end_time - start_time))

        #model
        nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)
        ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
        checkpoint_dir = os.path.join(dataset_dir, 'checkpoint', ffmpeg_option.summary(lr_video_name), nemo_s.name)
        checkpoint = nemo_s.load_checkpoint(checkpoint_dir)

        #cache profile
        cache_profile_file = os.path.join(dataset_dir, 'profile', lr_video_name, checkpoint.model.name,
                                            '{}_{}.profile'.format(aps_class.NAME1, args.threshold))

        #setup sr frames
        vpxdec_file = os.path.join(args.vpxdec_dir, vpxdec_name)
        start_time = time.time()
        libvpx_setup_sr_frame(vpxdec_file, dataset_dir, lr_video_name, checkpoint.model, postfix=postfix)
        end_time = time.time()
        print('saving sr image takes {}sec'.format(end_time - start_time))

        #measure online cache quality: nemo
        vpxdec_file = os.path.join(args.vpxdec_dir, vpxdec_name)
        start_time = time.time()
        libvpx_offline_cache_quality(vpxdec_file, dataset_dir, lr_video_name, hr_video_name, \
                checkpoint.model.name, cache_profile_file, lr_video_profile['height'], postfix=postfix)
        end_time = time.time()
        print('measuring online cache quality takes {}sec'.format(end_time - start_time))

        #remove sr images
        start_time = time.time()
        if args.mode == 'nemo':
            sr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, checkpoint.model.name)
        else:
            sr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, checkpoint.model.name, postfix)
        sr_image_files = glob.glob(os.path.join(sr_image_dir, '*.raw'))
        for sr_image_file in sr_image_files:
            os.remove(sr_image_file)
        end_time = time.time()
        print('removing sr image takes {}sec'.format(end_time-start_time))

        #remove lr, hr images
        start_time = time.time()
        if args.mode == 'nemo':
            lr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name)
        else:
            lr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, postfix)
        lr_image_files = glob.glob(os.path.join(lr_image_dir, '*.raw'))
        if args.mode == 'nemo':
            hr_image_dir = os.path.join(dataset_dir, 'image', hr_video_name)
        else:
            hr_image_dir = os.path.join(dataset_dir, 'image', hr_video_name, postfix)
        hr_image_files = glob.glob(os.path.join(hr_image_dir, '*.raw'))
        for lr_image_file, hr_image_file in zip(lr_image_files, hr_image_files):
            os.remove(lr_image_file)
            os.remove(hr_image_file)
        end_time = time.time()
        print('removing lr, hr image takes {}sec'.format(end_time-start_time))
