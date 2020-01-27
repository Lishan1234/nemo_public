import os
import sys
import argparse
import shlex
import math

import numpy as np
import tensorflow as tf

from tool.video import profile_video
from tool.libvpx import *
from dnn.model.nas_s import NAS_S
from dnn.utility import raw_quality

class APS_Uniform():
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
        save_frames(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.gop, chunk_idx)
        save_frames(self.vpxdec_file, self.dataset_dir, self.hr_video_name, self.gop, chunk_idx)
        setup_sr_frames(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.gop, chunk_idx, self.model)

        quality_bilinear = libvpx_bilinear_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    start_idx, self.gop, postfix)
        quality_dnn = libvpx_offline_dnn_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, start_idx, self.gop, postfix)

        #load frames (index)
        frames = libvpx_load_frame_index(self.dataset_dir, self.lr_video_name, self.gop, chunk_idx)

        #select/evaluate anchor points
        log_file = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, \
                                postfix, 'quality_{}.txt'.format(self.__class__.__name__))
        num_anchor_points = 0
        with open(log_file, 'w') as f:
            for i in range(len(frames)):
                #select anchor points uniformly
                num_anchor_points = i + 1
                cache_profile = CacheProfile.fromframes(frames, profile_dir, '{}_{}'.format(self.__class__.__name__, num_anchor_points))
                for j in range(num_anchor_points):
                    idx = j * math.floor(len(frames) / num_anchor_points)
                    cache_profile.add_anchor_point(frames[idx])
                cache_profile.save()

                #log
                quality_cache = libvpx_offline_cache_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, cache_profile, start_idx, self.gop, postfix)
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_error =  np.percentile(np.asarray(quality_dnn) - np.asarray(quality_cache) \
                                                            ,[95, 99, 100], interpolation='nearest')
                frame_count_1 = sum(map(lambda x : x >= 0.5, quality_diff))
                frame_count_2 = sum(map(lambda x : x >= 1.0, quality_diff))
                log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\n'.format(num_anchor_points,
                                        np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear),
                                        frame_count_1, frame_count_2, '\t'.join(str(np.round(x, 2)) for x in quality_error))
                f.write(log)

                print('{} video chunk, {} anchor points: PSNR(Cache)={:.2f}, PSNR(SR)={:.2f}, PSNR(Bilinear)={:.2f}'.format( \
                                        chunk_idx, num_anchor_points, np.average(quality_cache), np.average(quality_dnn), \
                                        np.average(quality_bilinear)))

                #check quality difference
                if np.average(quality_diff) <= self.threshold:
                    break
