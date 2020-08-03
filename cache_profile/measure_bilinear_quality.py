import os
import sys
import argparse
import shlex
import math
import time
import multiprocessing as mp
import shutil
import random
import itertools

import numpy as np

from nemo.tool.video import profile_video
from nemo.tool.libvpx import *
from nemo.tool.mac import count_mac_for_dnn, count_mac_for_cache
import nemo.dnn.model

class BilinearQuality():
    def __init__(self, vpxdec_path, dataset_dir, lr_video_name, hr_video_name, gop, output_width, output_height):
        self.vpxdec_path = vpxdec_path
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.output_width = output_width
        self.output_height = output_height

    def _measure_quality(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, postfix)
        os.makedirs(log_dir, exist_ok=True)

        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save high-resolution images
        libvpx_save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        #measure bilinear quality
        quality_bilinear = libvpx_bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)

        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        shutil.rmtree(hr_image_dir, ignore_errors=True)

        print(chunk_idx)

    def measure_quality(self, chunk_idx=None):
        if chunk_idx is not None:
            self._measure_quality(chunk_idx)
        else:
            lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
            lr_video_profile = profile_video(lr_video_path)
            num_chunks = int(math.ceil(lr_video_profile['duration'] / (args.gop / lr_video_profile['frame_rate'])))
            for i in range(num_chunks):
                self._measure_quality(i)

    def _aggreagate_quality(self):
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_chunks = int(math.ceil(lr_video_profile['duration'] / (args.gop / lr_video_profile['frame_rate'])))
        start_idx = 0
        end_idx = num_chunks - 1

        #log (bilinear, sr)
        log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, 'quality.txt')
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                quality = []
                chunk_log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, 'chunk{:04d}'.format(chunk_idx), 'quality.txt')
                with open(chunk_log_path, 'r') as f1:
                    lines = f1.readlines()
                    for line in lines:
                        line = line.strip()
                        quality.append(float(line.split('\t')[1]))
                    f0.write('{}\t{:.4f}\n'.format(chunk_idx, np.average(quality)))

    def aggregate_quality(self):
        self._aggreagate_quality()

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)
    parser.add_argument('--gop', type=int, default=120)

    args = parser.parse_args()

    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['NEMO_ROOT'], 'third_party', 'libvpx', 'bin', 'vpxdec_nemo_ver2_x86')
        assert(os.path.exists(args.vpxdec_path))

    #profile videos
    dataset_dir = os.path.join(args.data_dir, args.content)
    lr_video_path = os.path.join(dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(dataset_dir, 'video', args.hr_video_name)
    lr_video_profile = profile_video(lr_video_path)
    print(hr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = args.output_height // lr_video_profile['height']
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]

    #run aps
    print('content - {}, video - {}'.format(args.content, args.lr_video_name))
    aps = BilinearQuality(args.vpxdec_path, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, \
                              args.output_width, args.output_height)
    aps.measure_quality()
    aps.aggregate_quality()
