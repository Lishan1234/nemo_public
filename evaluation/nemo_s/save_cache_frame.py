import argparse
import os
import math
import glob
import re
import numpy as np
import imageio
import sys

import tensorflow as tf

from tool.video import profile_video, FFmpegOption
from tool.libvpx import *
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S

#assert(tf.__version__.startswith('2'))

if __name__ == '__main__':
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
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    #video
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--limit', type=int, default=1800)

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
        hr_video_file = os.path.abspath(sorted(glob.glob(os.path.join(video_dir, '{}p*'.format(args.hr_resolution))))[0])
        hr_video_profile = profile_video(hr_video_file)
        assert('encoded' not in hr_video_file)
        hr_video_profile = profile_video(hr_video_file)
        hr_video_name = os.path.basename(hr_video_file)
        scale = int(args.hr_resolution // args.lr_resolution)

        #setup lr, hr frames
        start_time = time.time()
        libvpx_save_frame(args.vpxdec_file, dataset_dir, lr_video_name, skip=args.skip, limit=args.limit)
        end_time = time.time()
        print('saving lr image takes {}sec'.format(end_time - start_time))

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

        #setup cache frames
        start_time = time.time()
        libvpx_save_cache_frame(args.vpxdec_file, dataset_dir, lr_video_name, hr_video_name, \
                checkpoint.model.name, cache_profile_file, lr_video_profile['height'], skip=args.skip, limit=args.limit)
        end_time = time.time()
        print('saving cache image takes {}sec'.format(end_time - start_time))

        #convert bilinear frames
        start_time = time.time()
        bilinear_image_dir = os.path.join(dataset_dir, 'image', lr_video_name)
        bilinear_images = [f for f in os.listdir(bilinear_image_dir) if re.search(r'\d\d\d\d.raw', f)]
        bilinear_images.sort()

        for bilinear_image in bilinear_images:
            raw_array = np.fromfile(os.path.join(bilinear_image_dir, bilinear_image), dtype='uint8')
            raw_array = raw_array.reshape(lr_video_profile['height'], lr_video_profile['width'], 3)
            file_name = os.path.basename(bilinear_image).split('.')[0]
            jpg_file = os.path.join(bilinear_image_dir, '{}.jpg'.format(file_name))
            imageio.imwrite(jpg_file, raw_array)
        end_time = time.time()
        print('saving bilinear images takes {}sec'.format(end_time - start_time))

        #encode a bilinear video
        video_dir = os.path.join(dataset_dir, 'video')
        start_time = time.time()
        cmd = '/usr/bin/ffmpeg -framerate 30 -i {}/%04d.jpg -s 1920x1080 -c:v libvpx-vp9 -pix_fmt yuv420p {}/bilinear.webm'.format(bilinear_image_dir, video_dir)
        os.system(cmd)
        end_time = time.time()
        print('encoding bilinear images takes {}sec'.format(end_time - start_time))

        #convert cache frames
        start_time = time.time()
        cache_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, checkpoint.model.name, os.path.basename(cache_profile_file))
        cache_images = [f for f in os.listdir(cache_image_dir) if re.search(r'\d\d\d\d.raw', f)]
        cache_images.sort()

        for cache_image in cache_images:
            raw_array = np.fromfile(os.path.join(cache_image_dir, cache_image), dtype='uint8')
            raw_array = raw_array.reshape(lr_video_profile['height'] * scale, lr_video_profile['width'] * scale, 3)
            file_name = os.path.basename(cache_image).split('.')[0]
            jpg_file = os.path.join(cache_image_dir,  '{}.jpg'.format(file_name))
            imageio.imwrite(jpg_file, raw_array)
        end_time = time.time()
        print('saving cache images takes {}sec'.format(end_time - start_time))


        #encode a cache video
        video_dir = os.path.join(dataset_dir, 'video')
        start_time = time.time()
        cmd = '/usr/bin/ffmpeg -framerate 30 -i {}/%04d.jpg -s 1920x1080 -c:v libvpx-vp9 -pix_fmt yuv420p {}/nemo.webm'.format(cache_image_dir, video_dir)
        os.system(cmd)
        end_time = time.time()
        print('encoding cache images takes {}sec'.format(end_time - start_time))


        #remove cache images
        start_time = time.time()
        cache_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, checkpoint.model.name, os.path.basename(cache_profile_file))
        cache_images = glob.glob(os.path.join(cache_image, '*.raw'))
        for cache_image in cache_images:
            os.remove(sr_image_file)
        cache_images = glob.glob(os.path.join(cache_image, '*.jpg'))
        for cache_image in cache_images:
            os.remove(sr_image_file)
        end_time = time.time()
        print('removing cache image takes {}sec'.format(end_time-start_time))

        #remove sr images
        start_time = time.time()
        sr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, checkpoint.model.name)
        sr_images = glob.glob(os.path.join(sr_image_dir, '*.raw'))
        for sr_image in sr_images:
            os.remove(sr_image)
        end_time = time.time()
        print('removing sr image takes {}sec'.format(end_time-start_time))

        #remove lr, hr images
        start_time = time.time()
        lr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name)
        lr_images = glob.glob(os.path.join(lr_image_dir, '*.raw'))
        for lr_image in lr_images:
            os.remove(lr_image)
        end_time = time.time()
        lr_images = glob.glob(os.path.join(lr_image_dir, '*.jpg'))
        for lr_image in lr_images:
            os.remove(lr_image)
        end_time = time.time()
        print('removing lr image takes {}sec'.format(end_time-start_time))
