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
    parser.add_argument('--baseline_num_filters', type=int, nargs='+', required=True)
    parser.add_argument('--baseline_num_blocks', type=int, nargs='+', required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    args = parser.parse_args()

    #validation
    assert(args.num_filters == args.baseline_num_filters[-1])
    assert(args.num_blocks == args.baseline_num_blocks[-1])

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

        #setup lr, hr frames
        start_time = time.time()
        libvpx_save_frame(args.vpxdec_file, dataset_dir, lr_video_name)
        libvpx_save_frame(args.vpxdec_file, dataset_dir, hr_video_name)
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
        start_time = time.time()
        libvpx_setup_sr_frame(args.vpxdec_file, dataset_dir, lr_video_name, checkpoint.model)
        end_time = time.time()
        print('saving sr image takes {}sec'.format(end_time - start_time))

        #measure online cache quality
        libvpx_offline_cache_quality(args.vpxdec_file, dataset_dir, lr_video_name, hr_video_name, \
                checkpoint.model.name, cache_profile_file, lr_video_profile['height'])

        #measure bilinear quality
        libvpx_bilinear_quality(args.vpxdec_file, dataset_dir, lr_video_name, hr_video_name)

        #remove sr images
        start_time = time.time()
        sr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, checkpoint.model.name)
        sr_image_files = glob.glob(os.path.join(sr_image_dir, '*.raw'))
        for sr_image_file in sr_image_files:
            os.remove(sr_image_file)
        end_time = time.time()
        print('removing hr image takes {}sec'.format(end_time-start_time))

        for num_blocks, num_filters in zip(args.baseline_num_blocks, args.baseline_num_filters):
            #model
            nemo_s = NEMO_S(num_blocks, num_filters, scale, args.upsample_type)
            ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
            checkpoint_dir = os.path.join(dataset_dir, 'checkpoint', ffmpeg_option.summary(lr_video_name), nemo_s.name)
            checkpoint = nemo_s.load_checkpoint(checkpoint_dir)

            #setup sr frames
            start_time = time.time()
            libvpx_setup_sr_frame(args.vpxdec_file, dataset_dir, lr_video_name, checkpoint.model)
            end_time = time.time()
            print('saving sr image takes {}sec'.format(end_time - start_time))

            #measure online sr quality
            start_time = time.time()
            libvpx_offline_dnn_quality(args.vpxdec_file, dataset_dir, lr_video_name, hr_video_name, \
                                checkpoint.model.name, lr_video_profile['height'])
            end_time = time.time()
            print('measuring cache quality takes {}sec'.format(end_time-start_time))

            #remove sr images
            start_time = time.time()
            sr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, checkpoint.model.name)
            sr_image_files = glob.glob(os.path.join(sr_image_dir, '*.raw'))
            for sr_image_file in sr_image_files:
                os.remove(sr_image_file)
            end_time = time.time()
            print('removing sr image takes {}sec'.format(end_time-start_time))

        #remove lr, hr images
        start_time = time.time()
        lr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name)
        lr_image_files = glob.glob(os.path.join(lr_image_dir, '*.raw'))
        hr_image_dir = os.path.join(dataset_dir, 'image', hr_video_name)
        hr_image_files = glob.glob(os.path.join(hr_image_dir, '*.raw'))
        for lr_image_file, hr_image_file in zip(lr_image_files, hr_image_files):
            os.remove(lr_image_file)
            os.remove(hr_image_file)
        end_time = time.time()
        print('removing lr, hr image takes {}sec'.format(end_time-start_time))
