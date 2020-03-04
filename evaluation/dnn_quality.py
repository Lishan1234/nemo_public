import time
import os
import argparse
import shutil
import glob

import numpy as np
import tensorflow as tf

from tool.video import FFmpegOption
from tool.libvpx import libvpx_save_frame, libvpx_bilinear_quality
from dnn.utility import resolve_bilinear
from dnn.dataset import setup_images, valid_image_dataset

postfix = 'libvpx'

def quality(vpxdec_file, dataset_dir, lr_video_name, hr_video_name, model):
    libvpx_save_frame(vpxdec_file, dataset_dir, lr_video_name)
    libvpx_save_frame(vpxdec_file, dataset_dir, hr_video_name)
    libvpx_setup_sr_frame(vpxdec_file, dataset_dir, lr_video_name, model)
    libvpx_offline_dnn_quality(vpxdec_file, dataset_dir, lr_video_name, hr_video_name, \
                                model.name, lr_video_profile['height'], start_idx, end_idx)

    lr_image_dir = os.path.join(dataset_dir, 'image', lr_video_name, postfix)
    lr_image_files = glob.globk(lr_image_dir, '*.raw')
    hr_image_dir = os.path.join(dataset_dir, 'image', hr_video_name, postfix)
    hr_image_files = glob.globk(lr_image_dir, '*.raw')

    for lr_image_file, hr_image_file in zip(lr_image_files, hr_image_files):
        os.remove(lr_image_file)
        os.remove(sr_image_file)
        os.remove(hr_image_file)
