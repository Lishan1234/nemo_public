import os
import sys
import argparse
import shlex
import math
import time
import multiprocessing as mp

import numpy as np
import tensorflow as tf

from tool.video import profile_video
from tool.libvpx import *
from tool.mac import count_mac_for_dnn, count_mac_for_cache
from dnn.model.nas_s import NAS_S
from dnn.utility import raw_quality

class APS_NEMO():
    def __init__(self, model, vpxdec_file, dataset_dir, lr_video_name, hr_video_name, gop, threshold, num_decoders):
        self.model = model
        self.vpxdec_file = vpxdec_file
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.threshold = threshold
        self.num_decoders = num_decoders

    def _select_anchor_point(self, cache_profile, ap_cache_profiles):
        max_estimated_quality = None
        max_avg_estimated_quality = None
        idx = None

        for i, ap_cache_profile in enumerate(ap_cache_profiles):
            estimated_quality = self._estimate_quality(cache_profile, ap_cache_profile)
            avg_estimated_quality = np.average(estimated_quality)
            if idx is None or avg_estimated_quality > max_avg_estimated_quality:
                max_avg_estimated_quality = avg_estimated_quality
                max_estimated_quality = estimated_quality
                idx = i

        return idx, max_estimated_quality

    def _estimate_quality(self, curr_cache_profile, new_cache_profile):
        if curr_cache_profile is not None:
            return np.maximum(curr_cache_profile.estimated_quality, new_cache_profile.measured_quality)
        else:
            return new_cache_profile.measured_quality

    def run(self, chunk_idx):
        ###########step 1: analyze anchor points##########
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

        q0 = mp.Queue()
        q1 = mp.Queue()
        decoders = [mp.Process(target=libvpx_offline_cache_quality_mt, args=(q0, q1, self.vpxdec_file, self.dataset_dir, \
                                    self.lr_video_name, self.hr_video_name, self.model.name)) for i in range(self.num_decoders)]
        for decoder in decoders:
            decoder.start()

        ap_cache_profiles = []
        for idx, frame in enumerate(frames):
            #select anchor points uniformly
            cache_profile = CacheProfile.fromframes(frames, profile_dir, '{}_{}'.format(self.__class__.__name__, frame.name))
            cache_profile.add_anchor_point(frame)
            cache_profile.save()

            #measure quality
            q0.put((cache_profile, start_idx, self.gop, postfix, idx))
            #quality_cache = libvpx_offline_cache_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
            #                                    self.model.name, cache_profile, start_idx, self.gop, postfix)
            #cache_profile.set_measured_quality(quality_cache)
            ap_cache_profiles.append(cache_profile)

            break

        for frame in frames:
            item = q1.get()
            idx = item[0]
            quality = item[1]
            ap_cache_profiles[idx].set_measured_quality(quality)

            break

        for decoder in decoders:
            q0.put('end')

        for decoder in decoders:
            decoder.join()

        ###########step 2: order anchor points##########
        ordered_cache_profiles = []
        cache_profile = None
        while len(ap_cache_profiles) > 0:
            idx, estimated_quality = self._select_anchor_point(cache_profile, ap_cache_profiles)
            ap_cache_profile = ap_cache_profiles.pop(idx)
            #print('_select_cache_profile: {} anchor points, {} chunk'.format(ap_cache_profile.anchor_points[0].name, chunk_idx))
            if len(ordered_cache_profiles) == 0:
                cache_profile = CacheProfile.fromcacheprofile(ap_cache_profile, profile_dir, '{}_{}'.format(self.__class__.__name__, len(ap_cache_profile.anchor_points)))
                cache_profile.set_estimated_quality(ap_cache_profile.measured_quality)
            else:
                cache_profile = CacheProfile.fromcacheprofile(ordered_cache_profiles[-1], profile_dir, '{}_{}'.format(self.__class__.__name__, len(ordered_cache_profiles[-1].anchor_points) + 1))
                cache_profile.add_anchor_point(ap_cache_profile.anchor_points[0])
                cache_profile.set_estimated_quality(estimated_quality)
            cache_profile.save()
            ordered_cache_profiles.append(cache_profile)

        ###########step 3: select anchor points##########
        log_file = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, \
                postfix, 'quality_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        hr_video_file = os.path.join(self.dataset_dir, 'video', self.hr_video_name)
        hr_video_profile = profile_video(hr_video_file)
        cache_mac = count_mac_for_cache(self.model.nhwc[1] * self.model.scale, self.model.nhwc[2] * self.model.scale, 3)
        dnn_mac = count_mac_for_dnn(self.model.name, self.model.nhwc[1], self.model.nhwc[2])
        decode_dnn_mac = dnn_mac * self.gop
        with open(log_file, 'w') as f:
            for cache_profile in ordered_cache_profiles:
                #log
                quality_cache = libvpx_offline_cache_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, cache_profile, start_idx, self.gop, postfix)
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_error =  np.percentile(np.asarray(quality_dnn) - np.asarray(quality_cache) \
                                                            ,[95, 99, 100], interpolation='nearest')
                frame_count_1 = sum(map(lambda x : x >= 0.5, quality_diff))
                frame_count_2 = sum(map(lambda x : x >= 1.0, quality_diff))
                decode_cache_mac = dnn_mac * len(cache_profile.anchor_points) + cache_mac * (self.gop - len(cache_profile.anchor_points))
                log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\n'.format(len(cache_profile.anchor_points), \
                                        np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear), \
                                        frame_count_1, frame_count_2, '\t'.join(str(np.round(x, 2)) for x in quality_error), \
                                        decode_cache_mac / 1e9, decode_dnn_mac / 1e9)
                f.write(log)

                print('{} video chunk, {} anchor points: PSNR(Cache)={:.2f}, PSNR(SR)={:.2f}, PSNR(Bilinear)={:.2f}'.format( \
                                        chunk_idx, len(cache_profile.anchor_points), np.average(quality_cache), np.average(quality_dnn), \
                                        np.average(quality_bilinear)))

                #check quality difference
                if np.average(quality_diff) <= self.threshold:
                    cache_profile.name = '{}_{}'.format(self.__class__.__name__, self.threshold)
                    cache_profile.save()
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

