import os
import sys
import argparse
import shlex
import math
import random

import numpy as np
import tensorflow as tf

from tool.video import profile_video
from tool.libvpx import *
from tool.mac import count_mac_for_dnn, count_mac_for_cache
from dnn.model.nas_s import NAS_S
from dnn.utility import raw_quality

class APS_Random():
    def __init__(self, model, vpxdec_file, dataset_dir, lr_video_name, hr_video_name, gop, threshold):
        self.model = model
        self.vpxdec_file = vpxdec_file
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.threshold = threshold

    def run(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, postfix)

        #setup lr, sr, hr frames
        libvpx_save_frame(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.gop, chunk_idx)
        libvpx_save_frame(self.vpxdec_file, self.dataset_dir, self.hr_video_name, self.gop, chunk_idx)
        libvpx_setup_sr_frame(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.gop, chunk_idx, self.model)

        quality_bilinear = libvpx_bilinear_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    start_idx, self.gop, postfix)
        quality_dnn = libvpx_offline_dnn_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, start_idx, self.gop, postfix)

        #load frames (index)
        frames = libvpx_load_frame_index(self.dataset_dir, self.lr_video_name, self.gop, chunk_idx)

        #select/evaluate anchor points
        log_file = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, \
                                postfix, 'quality_{}_{}.txt'.format(self.__class__.__name__, self.threshold))
        num_anchor_points = 0
        cache_mac = count_mac_for_cache(self.model.nhwc[1] * self.model.scale, self.model.nhwc[2] * self.model.scale, 3)
        dnn_mac = count_mac_for_dnn(self.model.name, self.model.nhwc[1], self.model.nhwc[2])
        decode_dnn_mac = dnn_mac * self.gop
        with open(log_file, 'w') as f:
            for i in range(len(frames)):
                #select anchor points uniformly
                num_anchor_points = i + 1
                cache_profile = CacheProfile.fromframes(frames, profile_dir, '{}_{}'.format(self.__class__.__name__, num_anchor_points))
                random_frames = random.sample(frames, num_anchor_points)
                for frame in random_frames:
                    cache_profile.add_anchor_point(frame)
                cache_profile.save()

                #log
                quality_cache = libvpx_offline_cache_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, cache_profile, start_idx, self.gop, postfix)
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_error =  np.percentile(np.asarray(quality_dnn) - np.asarray(quality_cache) \
                                                            ,[95, 99, 100], interpolation='nearest')
                frame_count_1 = sum(map(lambda x : x >= 0.5, quality_diff))
                frame_count_2 = sum(map(lambda x : x >= 1.0, quality_diff))
                decode_cache_mac = dnn_mac * len(cache_profile.anchor_points) + cache_mac * (self.gop - len(cache_profile.anchor_points))
                log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\n'.format(num_anchor_points,
                                        np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear),
                                        frame_count_1, frame_count_2, '\t'.join(str(np.round(x, 2)) for x in quality_error),
                                        decode_cache_mac / 1e9, decode_dnn_mac / 1e9)
                f.write(log)

                print('{} video chunk, {} anchor points: PSNR(Cache)={:.2f}, PSNR(SR)={:.2f}, PSNR(Bilinear)={:.2f}'.format( \
                                        chunk_idx, num_anchor_points, np.average(quality_cache), np.average(quality_dnn), \
                                        np.average(quality_bilinear)))

                #check quality difference
                if np.average(quality_diff) <= self.threshold:
                    break

    def summary(self):
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name)
        summary_log_file = os.path.join(log_dir, 'quality_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        chunk_idx = 0
        with open(summary_log_file, 'w') as s_f:
            #iterate over chunks
            while True:
                chunk_log_dir = os.path.join(log_dir, 'chunk{:04d}'.format(chunk_idx))
                if not os.path.exists(chunk_log_dir):
                    break
                else:
                    chunk_log_file = os.path.join(chunk_log_dir, 'quality_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
                    with open(chunk_log_file, 'r') as c_f:
                        lines = c_f.readlines()
                        s_f.write('{}\t{}\n'.format(chunk_idx, lines[-1].strip()))
                    chunk_idx += 1
