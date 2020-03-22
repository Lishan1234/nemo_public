import os
import sys
import argparse
import shlex
import math
import random
import shutil
import multiprocessing as mp
import itertools

import numpy as np
import tensorflow as tf

from tool.video import profile_video
from tool.libvpx import *
from tool.mac import count_mac_for_dnn, count_mac_for_cache
from dnn.model.nas_s import NAS_S
from dnn.utility import raw_quality

class APS_Exhaustive():
    NAME0="APS"
    NAME1="Exhaustive"

    def __init__(self, model, vpxdec_file, dataset_dir, lr_video_name, hr_video_name, gop, num_decoders, num_anchor_points, iteration):
        self.model = model
        self.vpxdec_file = vpxdec_file
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.num_decoders = num_decoders
        self.num_anchor_points = num_anchor_points
        self.iteration = iteration

    def run(self, chunk_idx):
        start_time = time.time()
        lr_video_file = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_file)
        total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        assert(total_frames == math.floor(total_frames))
        left_frames = total_frames - chunk_idx * self.gop
        total_frames = int(total_frames)
        left_frames = int(left_frames)

        start_idx = chunk_idx * self.gop
        end_idx = self.gop if left_frames >= self.gop else left_frames
        #end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        os.makedirs(profile_dir, exist_ok=True)

        #setup lr, sr, hr frames
        libvpx_save_frame(self.vpxdec_file, self.dataset_dir, self.lr_video_name, start_idx, end_idx, postfix)
        libvpx_save_frame(self.vpxdec_file, self.dataset_dir, self.hr_video_name, start_idx, end_idx, postfix)
        libvpx_setup_sr_frame(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.model, postfix)
        quality_bilinear = libvpx_bilinear_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    start_idx, end_idx, postfix)
        quality_dnn = libvpx_offline_dnn_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, lr_video_profile['height'], start_idx, end_idx, postfix)

        #load frames (index)
        frames = libvpx_load_frame_index(self.dataset_dir, self.lr_video_name, postfix)

        #select/evaluate anchor points
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, self.NAME1, postfix)
        os.makedirs(log_dir, exist_ok=True)
        quality_log_file = os.path.join(log_dir, 'quality.txt')

        #exhaustive search
        q0 = mp.Queue()
        q1 = mp.Queue()
        decoders = [mp.Process(target=libvpx_offline_cache_quality_mt_v1, args=(q0, q1, self.vpxdec_file, self.dataset_dir, \
                                    self.lr_video_name, self.hr_video_name, self.model.name, lr_video_profile['height'])) for i in range(self.num_decoders)]
        for decoder in decoders:
            decoder.start()

        anchor_point_sets = list(itertools.combinations(frames, 3))
        random.shuffle(anchor_point_sets)

        if self.iteration is None:
            total_iteration = len(anchor_point_sets)
        else:
            total_iteration = self.iteration

        for i in range(total_iteration):
            #select anchor points uniformly
            cache_profile = CacheProfile.fromframes(frames, profile_dir, '{}_iter{}.tmp'.format(self.NAME1, i))
            for frame in anchor_point_sets[i]:
                cache_profile.add_anchor_point(frame)
            #measure quality
            q0.put((cache_profile, start_idx, end_idx, postfix))

        count = 0
        total_start_time = time.time()
        with open(quality_log_file, 'w') as f:
            for i in range(total_iteration):
                start_time = time.time()
                item = q1.get()
                quality_cache = item
                if(len(quality_cache)) == 0:
                    print(quality_cache)

                quality_diff = np.average(np.asarray(quality_dnn) - np.asarray(quality_cache))
                if quality_diff < 0.5:
                    count += 1

                quality_log = '{:.2f}\t{:.2f}\t{:.2f}\n'.format(np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                f.write(quality_log)

                end_time = time.time()
                print('Iteration {} ({}sec): Percentage {}%, Quality difference {:.2f}dB'.format(i, end_time - start_time, count/(i+1) * 100, quality_diff))

        for decoder in decoders:
            q0.put('end')

        for decoder in decoders:
            decoder.join()
        total_end_time = time.time()
        print('{} number of iterations is finished in {}sec: {}iteration/sec'.format(total_iteration, total_end_time - total_start_time, total_iteration / (total_end_time - total_start_time)))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)
