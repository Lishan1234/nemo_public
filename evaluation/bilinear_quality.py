import time
import os
import argparse
import shutil
import glob
import sys

import numpy as np
import tensorflow as tf

from tool.video import FFmpegOption
from tool.libvpx import libvpx_save_frame, libvpx_bilinear_quality
from dnn.utility import resolve_bilinear
from dnn.dataset import setup_images, valid_image_dataset

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--lr_resolution', type=int, nargs='+', required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)
    parser.add_argument('--ffmpeg_file', type=str, required=True)
    parser.add_argument('--vpxdec_file', type=str, required=True)

    args = parser.parse_args()

    for content in args.content:
        dataset_dir = os.path.join(args.dataset_rootdir, content)
        video_dir = os.path.join(dataset_dir, 'video')
        ffmpeg_option = FFmpegOption('none', None, None)
        hr_video_file = glob.glob(os.path.join(video_dir, '{}p*'.format(args.hr_resolution)))[1]
        assert('encoded' not in hr_video_file)
        libvpx_save_frame(args.vpxdec_file, dataset_dir, os.path.basename(hr_video_file))

        for lr_resolution in args.lr_resolution:
            lr_video_file = glob.glob(os.path.join(video_dir, '{}p*'.format(lr_resolution)))[0]
            print(lr_video_file)
            sys.exit()
            libvpx_bilinear_quality(args.vpxdec_file, dataset_dir, os.path.basename(lr_video_file), os.path.basename(hr_video_file))

        hr_image_dir = os.path.join(dataset_dir, 'image', ffmpeg_option.summary(os.path.basename(hr_video_file)))
        shutil.rmtree(hr_image_dir, ignore_errors=True)
