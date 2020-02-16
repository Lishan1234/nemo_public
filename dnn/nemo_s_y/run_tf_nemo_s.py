import time
import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from dnn.dataset import raw_dataset, train_raw_dataset, valid_raw_dataset, setup_yuv_images
from dnn.model.nemo_s_y import NEMO_S_Y
from dnn.test import SingleTester
from dnn.utility import resolve, resolve_bilinear
from tool.video import profile_video, VideoMetadata, FFmpegOption

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--vpxdec_file', type=str, required=True)

    #video metadata
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #dataset
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #log
    parser.add_argument('--custom_tag', type=str, default=None)
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()

    #scale
    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_file))
    assert(os.path.exists(hr_video_file))
    lr_video_profile = profile_video(lr_video_file)
    hr_video_profile = profile_video(hr_video_file)
    scale = hr_video_profile['height'] // lr_video_profile['height']
    width = lr_video_profile['width']
    height = lr_video_profile['height']

    #image
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    setup_yuv_images(args.vpxdec_file, args.dataset_dir, lr_video_file, args.filter_fps)
    setup_yuv_images(args.vpxdec_file, args.dataset_dir, hr_video_file, args.filter_fps)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name, 'libvpx')
    hr_image_dir = os.path.join(args.dataset_dir, 'image', args.hr_video_name, 'libvpx')

    #dnn
    nemo_s_y = NEMO_S_Y(args.num_blocks, args.num_filters, scale, args.upsample_type)
    if (hr_video_profile['height'] % lr_video_profile['height'] == 0 and
            hr_video_profile['width'] % lr_video_profile['width'] == 0):
        model = nemo_s_y.build_model()
    else:
        model = nemo_s_y.build_model(resolution=(hr_video_profile['height'], hr_video_profile['width']))

    #dataset
    lr_y_ds = raw_dataset(lr_image_dir, width, height, 1, '\d\d\d\d.y', tf.uint8)
    lr_u_ds = raw_dataset(lr_image_dir, width / 2, height / 2, 1, '\d\d\d\d.u', tf.uint8)
    lr_v_ds = raw_dataset(lr_image_dir, width / 2, height / 2, 1, '\d\d\d\d.v', tf.uint8)
    lr_yuv_ds = tf.dataset.Dataset.zip((lr_y_ds, lr_u_ds, lr_v_ds))
    hr_y_ds = raw_dataset(hr_image_dir, width, height, 1, '\d\d\d\d.y', tf.uint8)
    hr_u_ds = raw_dataset(hr_image_dir, width / 2, height / 2, 1, '\d\d\d\d.u', tf.uint8)
    hr_v_ds = raw_dataset(hr_image_dir, width / 2, height / 2, 1, '\d\d\d\d.v', tf.uint8)
    hr_yuv_ds = tf.dataset.Dataset.zip((hr_y_ds, hr_u_ds, hr_v_ds))
    yuv_ds = tf.dataset.Dataset.zip((lr_yuv_ds, hr_yuv_ds))

    #trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), model.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), model.name)
    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name), model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    tester = SingleTester(model, checkpoint_dir, log_dir, image_dir)
    #tester.test(valid_ds, args.save_image)
