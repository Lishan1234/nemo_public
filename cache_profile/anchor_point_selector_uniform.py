import os
import sys
import argparse
import shlex
import math
import shutil

import numpy as np
import tensorflow as tf

from tool.video import profile_video
from tool.libvpx import *
from tool.mac import count_mac_for_dnn, count_mac_for_cache
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
        start_time = time.time()
        lr_video_file = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_file)
        total_frames = lr_video_profile['frame_rate'] * lr_video_profile['duration']
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
        libvpx_save_frame(self.vpxdec_file, self.dataset_dir, self.lr_video_name, start_idx, end_idx, chunk_idx)
        libvpx_save_frame(self.vpxdec_file, self.dataset_dir, self.hr_video_name, start_idx, end_idx, chunk_idx)
        libvpx_setup_sr_frame(self.vpxdec_file, self.dataset_dir, self.lr_video_name, chunk_idx, self.model)
        quality_bilinear = libvpx_bilinear_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    start_idx, end_idx, postfix)
        quality_dnn = libvpx_offline_dnn_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, lr_video_profile['height'], start_idx, end_idx, postfix)

        #load frames (index)
        frames = libvpx_load_frame_index(self.dataset_dir, self.lr_video_name, chunk_idx)

        #select/evaluate anchor points
        quality_log_file = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, \
                postfix, 'quality_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        error_log_file = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, \
                postfix, 'error_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        mac_log_file = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, \
                postfix, 'mac_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        num_anchor_points = 0
        cache_mac = count_mac_for_cache(self.model.nhwc[1] * self.model.scale, self.model.nhwc[2] * self.model.scale, 3)
        dnn_mac = count_mac_for_dnn(self.model.name, self.model.nhwc[1], self.model.nhwc[2])
        decode_dnn_mac = dnn_mac * end_idx
        with open(quality_log_file, 'w') as f0, open(error_log_file, 'w') as f1, open(mac_log_file, 'w') as f2:
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
                                                    self.model.name, cache_profile, lr_video_profile['height'], start_idx, end_idx, postfix)
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_error =  np.percentile(np.asarray(quality_dnn) - np.asarray(quality_cache) \
                                                            ,[95, 99, 100], interpolation='nearest')
                frame_count_1 = sum(map(lambda x : x >= 0.5, quality_diff))
                frame_count_2 = sum(map(lambda x : x >= 1.0, quality_diff))
                frame_count_3 = sum(map(lambda x : x >= 2.0, quality_diff))
                frame_count_4 = sum(map(lambda x : x >= 3.0, quality_diff))
                decode_cache_mac = dnn_mac * len(cache_profile.anchor_points) + cache_mac * (end_idx - len(cache_profile.anchor_points))

                quality_log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(len(cache_profile.anchor_points), np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                f0.write(quality_log)
                error_log = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(len(cache_profile.anchor_points), frame_count_1, frame_count_2, frame_count_3, frame_count_4, '\t'.join(str(np.round(x, 2)) for x in quality_error))
                f1.write(error_log)
                mac_log = '{}\t{:.2f}\t{:.2f}\n'.format(len(cache_profile.anchor_points), decode_cache_mac / 1e9, decode_dnn_mac / 1e9)
                f2.write(mac_log)

                print('{} video chunk, {} anchor points: PSNR(Cache)={:.2f}, PSNR(SR)={:.2f}, PSNR(Bilinear)={:.2f}'.format( \
                                        chunk_idx, num_anchor_points, np.average(quality_cache), np.average(quality_dnn), \
                                        np.average(quality_bilinear)))

                #check quality difference
                if np.average(quality_diff) <= self.threshold:
                    cache_profile.save_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
                    cache_profile.name = '{}_{}'.format(self.__class__.__name__, self.threshold)
                    cache_profile.save()
                    break

        end_time = time.time()
        print('0 video chunk: {}sec'.format(end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    def summary(self, start_idx, end_idx):
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name)
        profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name)

        #log
        quality_summary_file = os.path.join(log_dir, 'quality_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        error_summary_file = os.path.join(log_dir, 'error_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        mac_summary_file = os.path.join(log_dir, 'mac_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
        with open(quality_summary_file, 'w') as sf0, open(error_summary_file, 'w') as sf1, open(mac_summary_file, 'w') as sf2:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                chunk_log_dir = os.path.join(log_dir, 'chunk{:04d}'.format(chunk_idx))
                if not os.path.exists(chunk_log_dir):
                    break
                else:
                    quality_log_file = os.path.join(chunk_log_dir, 'quality_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
                    error_log_file = os.path.join(chunk_log_dir, 'error_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
                    mac_log_file = os.path.join(chunk_log_dir, 'mac_{}_{:.2f}.txt'.format(self.__class__.__name__, self.threshold))
                    with open(quality_log_file, 'r') as lf0, open(error_log_file, 'r') as lf1, open(mac_log_file, 'r') as lf2:
                        q_lines = lf0.readlines()
                        e_lines = lf1.readlines()
                        m_lines = lf2.readlines()
                        sf0.write('{}\t{}\n'.format(chunk_idx, q_lines[-1].strip()))
                        sf1.write('{}\t{}\n'.format(chunk_idx, e_lines[-1].strip()))
                        sf2.write('{}\t{}\n'.format(chunk_idx, m_lines[-1].strip()))

        #cache profile
        cache_profile = os.path.join(log_dir, '{}_{}'.format(self.__class__.__name__, self.threshold))
        cache_data = b''
        with open(cache_profile, 'wb') as f0:
            for chunk_idx in range(start_idx, end_idx):
                chunk_cache_profile = os.path.join(profile_dir, 'chunk{:04d}'.format(chunk_idx), '{}_{}'.format(self.__class__.__name__, self.threshold))
                with open(chunk_cache_profile, 'rb') as f1:
                    f0.write(f1.read())
