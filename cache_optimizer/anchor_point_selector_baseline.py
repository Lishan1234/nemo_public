import os
import sys
import argparse
import glob
import time
import struct
import subprocess
import shlex
import multiprocessing as mp
import re
import logging
import math

import scipy.misc
import numpy as np
import tensorflow as tf

from tool.tf import single_raw_dataset, valid_raw_dataset
from tool.ffprobe import profile_video
from tool.libvpx import Frame, CacheProfile, get_num_threads
from dnn.model.edsr_s import EDSR_S

#deprecated: convert raw to png
"""
video_path = os.path.join(self.content_dir, 'video', self.compare_video)
video_info = profile_video(video_path)
images = glob.glob(os.path.join(hr_image_dir, '*.raw'))
for idx, image in enumerate(images):
    arr = np.fromfile(image, dtype=np.uint8)
    arr = np.reshape(arr, (video_info['height'], video_info['width'], 3))
    name = os.path.splitext(os.path.basename(image))[0]
    name += '.png'
    scipy.misc.imsave(os.path.join(hr_image_dir, name), arr)
"""

class APS_Baseline():
    def __init__(self, model, checkpoint_dir, vpxdec_path, content_dir, input_video, compare_video, num_decoders, gop, quality_diff):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.vpxdec_path = vpxdec_path
        self.content_dir = content_dir
        self.input_video = input_video
        self.compare_video = compare_video
        self.num_decoders = num_decoders
        self.gop = gop
        self.frames = None
        self.quality_diff = quality_diff

        self.q0 = mp.Queue()
        self.q1 = mp.Queue()
        self.q2 = mp.Queue()

        self.p0 = mp.Process(target=self._prepare_anchor_points, args=(self.q0, self.q1))
        self.p1 = mp.Process(target=self._analyze_cache_profiles, args=(self.q1, self.q2))

    def _prepare_hr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        os.makedirs(image_dir, exist_ok=True)
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame'.format(self.vpxdec_path,
                start_idx, end_idx - start_idx, self.content_dir, self.compare_video, postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def _prepare_lr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        lr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        os.makedirs(lr_image_dir, exist_ok=True)

        command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame --save-metadata'.format(self.vpxdec_path, \
                start_idx, end_idx - start_idx, self.content_dir, self.input_video, postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def _prepare_sr_frames(self, chunk_idx, model):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        lr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        sr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, self.model.name, postfix)
        os.makedirs(sr_image_dir, exist_ok=True)

        input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
        input_video_info = profile_video(input_video_path)

        single_raw_ds = single_raw_dataset(lr_image_dir, input_video_info['height'], input_video_info['width'])
        for idx, img in enumerate(single_raw_ds):
            lr = img[0]
            lr = tf.cast(lr, tf.float32)
            sr = model(lr)

            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)

            sr_image = tf.squeeze(sr).numpy()
            name = os.path.basename(img[1].numpy()[0].decode())
            sr_image.tofile(os.path.join(sr_image_dir, name))

            #validate
            #sr_image = tf.image.encode_png(tf.squeeze(sr))
            #tf.io.write_file(os.path.join('.', '{0:04d}.png'.format(idx+1)), sr_image)

    def _load_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        frames = []
        log_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'metadata.txt')
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                current_video_frame = int(line.split('\t')[0])
                current_super_frame = int(line.split('\t')[1])
                frames.append(Frame(current_video_frame, current_super_frame))

        return frames

    def _prepare_anchor_points(self, q0, q1):
        tf.enable_eager_execution()
        checkpoint = self.model.load_checkpoint(self.checkpoint_dir)

        while True:
            item = q0.get()
            if item == 'end':
                return
            else:
                chunk_idx = item
                print('_prepare_anchor_points: start {} chunk'.format(chunk_idx))
                self._prepare_lr_frames(chunk_idx)
                self._prepare_hr_frames(chunk_idx)
                self._prepare_sr_frames(chunk_idx, checkpoint.model)
                q1.put(chunk_idx)
                print('_prepare_anchor_points: end {} chunk'.format(chunk_idx))

    def _run_cache_profile(self, q0, q1):
        while True:
            item = q0.get()
            if item == 'end':
                return
            else:
                #setup
                chunk_idx = item[0]
                cache_profile = item[1]
                path = item[2]
                idx = item[3]

                #save
                cache_profile.save(path)

                #run
                command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
                --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, path)
                subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                #result
                q1.put(idx)

    def _execute_command(self, q0, q1):
        while True:
            item = q0.get()
            if item == 'end':
                return
            else:
                #setup
                command = item[0]
                idx = item[1]

                #run
                subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                #result
                q1.put(idx)

    def _analyze_cache_profiles(self, q0, q1):
        tf.enable_eager_execution()
        q2 = mp.Queue()
        q3 = mp.Queue()
        decoders = [mp.Process(target=self._execute_command, args=(q2, q3)) for i in range(self.num_decoders)]

        for decoder in decoders:
            decoder.start()

        while True:
            item = q0.get()
            if item == 'end':
                break
            else:
                chunk_idx = item
                print('_analyze_cache_profiles: start {} chunk'.format(chunk_idx))

                start_idx = chunk_idx * self.gop
                end_idx = (chunk_idx + 1) * self.gop
                postfix = 'chunk{:04d}'.format(chunk_idx)
                log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name, postfix)
                profile_dir = os.path.join(self.content_dir, 'profile', self.input_video, postfix)
                os.makedirs(profile_dir, exist_ok=True)

                frames = self._load_frames(chunk_idx)
                cache_profiles = []
                for i in range(len(frames)):
                    num_anchor_points = i + 1
                    cache_profile = CacheProfile.fromframes(frames, profile_dir, 'uniform_{}'.format(num_anchor_points))
                    for j in range(num_anchor_points):
                        idx = j * math.floor(len(frames) / num_anchor_points)
                        cache_profile.add_anchor_point(frames[idx])
                    cache_profile.save()
                    cache_profiles.append(cache_profile)
                    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
                --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --save-metadata --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, cache_profile.path)
                    q2.put([command, i])

                for _ in range(len(frames)):
                    idx = q3.get()
                    quality = []
                    path = os.path.join(log_dir, cache_profiles[idx].name, 'quality.txt')
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            quality.append(float(line.split('\t')[1]))
                    cache_profiles[idx].set_measured_quality(quality)

                #load dnn quality
                dnn_quality = []
                sr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, self.model.name, postfix)
                hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
                video_path = os.path.join(self.content_dir, 'video', self.compare_video)
                video_info = profile_video(video_path)
                with tf.device('cpu:0'):
                    valid_raw_ds = valid_raw_dataset(sr_image_dir, hr_image_dir, video_info['height'], video_info['width'], \
                                                    1, pattern='[0-9][0-9][0-9][0-9].raw')
                    for idx, img in enumerate(valid_raw_ds):
                        sr = img[0][0]
                        hr = img[1][0]
                        psnr = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
                        dnn_quality.append(psnr)

                log_path = os.path.join(log_dir, 'aps_baseline.txt')
                with open(log_path, 'w') as f:
                    for cache_profile in sorted(cache_profiles):
                        quality_error = np.percentile(np.asarray(dnn_quality) - np.asarray(cache_profile.measured_quality), [90, 95, 100], interpolation='nearest')
                        quality_error = '\t'.join(str(np.round(x, 2)) for x in quality_error)
                        log = '{}\t{:.2f}\t{}\n'.format(len(cache_profile.anchor_points), np.average(cache_profile.measured_quality), quality_error)
                        f.write(log)

                q1.put(chunk_idx)
                print('_analyze_cache_profiles: end {} chunk'.format(chunk_idx))

        for decoder in decoders:
            q2.put('end')

        for decoder in decoders:
            decoder.join()

    def start_process(self):
        self.p0.start()
        self.p1.start()

    def stop_process(self):
        self.q0.put('end')
        self.q1.put('end')

        self.p0.join()
        self.p1.join()

    def run_asynchrnous(self, chunk_idx):
        self.q0.put(chunk_idx)

    def run_synchrnous(self, chunk_idx):
        self.q0.put(chunk_idx)
        self.q2.get()

    def debug_asynchrnous(self, chunk_idx):
        #self.q0.put(chunk_idx)
        #self.q0.put('end')
        #self._prepare_anchor_points(self.q1, self.q2)
        self.q1.put(chunk_idx)
        self.q1.put('end')
        self._analyze_cache_profiles(self.q1, self.q2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache Erosion Analyzer')

    #options for libvpx
    parser.add_argument('--vpxdec_path', type=str, required=True)
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--input_video_name', type=str, required=True)
    parser.add_argument('--compare_video_name', type=str, required=True)
    parser.add_argument('--num_decoders', type=int, default=1)
    parser.add_argument('--gop', type=int, required=True)

    #options for edsr_s (DNN)
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    #opttions for anchor point selector
    parser.add_argument('--quality_diff', type=float, required=True)

    args = parser.parse_args()

    input_video_path = os.path.join(args.content_dir, 'video', args.input_video_name)
    compare_video_path = os.path.join(args.content_dir, 'video', args.compare_video_name)
    input_video_info = profile_video(input_video_path)
    compare_video_info = profile_video(compare_video_path)

    scale = int(compare_video_info['height'] / input_video_info['height'])
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, scale, None)
    checkpoint_dir = os.path.join(args.checkpoint_dir, edsr_s.name)

    aps_baseline = APS_Baseline(edsr_s, checkpoint_dir, args.vpxdec_path, args.content_dir, args.input_video_name, args.compare_video_name, args.num_decoders, args.gop, args.quality_diff)
    aps_baseline.start_process()
    aps_baseline.run_synchrnous(5)
    aps_baseline.stop_process()
