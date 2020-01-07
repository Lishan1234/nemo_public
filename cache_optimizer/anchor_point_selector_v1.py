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

import scipy.misc
import numpy as np
import tensorflow as tf

from tool.tf import single_raw_dataset, valid_raw_dataset
from tool.ffprobe import profile_video
from tool.libvpx import Frame, CacheProfile, get_num_threads
from dnn.model.edsr_s import EDSR_S
from dnn.utility import resolve_bilinear

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

class APS_v1():
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
        self.q3 = mp.Queue()

        self.p0 = mp.Process(target=self._prepare_anchor_points, args=(self.q0, self.q1))
        self.p1 = mp.Process(target=self._analyze_anchor_points, args=(self.q1, self.q2))
        self.p2 = mp.Process(target=self._select_cache_profile, args=(self.q2, self.q3))

    def _prepare_hr_frames(self, chunk_idx):
        start_time = time.time()
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        os.makedirs(image_dir, exist_ok=True)
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame'.format(self.vpxdec_path,
                start_idx, end_idx - start_idx, self.content_dir, self.compare_video, postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        end_time = time.time()
        return end_time - start_time

    def _prepare_lr_frames(self, chunk_idx):
        start_time = time.time()
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        lr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        os.makedirs(lr_image_dir, exist_ok=True)

        command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame --save-metadata'.format(self.vpxdec_path, \
                start_idx, end_idx - start_idx, self.content_dir, self.input_video, postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        end_time = time.time()
        return end_time - start_time

    def _prepare_sr_frames(self, chunk_idx, model):
        start_time = time.time()
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        lr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        sr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, self.model.name, postfix)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name, postfix)
        os.makedirs(sr_image_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
        input_video_info = profile_video(input_video_path)
        compare_video_path = os.path.join(self.content_dir, 'video', self.compare_video)
        compare_video_info = profile_video(compare_video_path)

        input_raw_ds = single_raw_dataset(lr_image_dir, input_video_info['height'], input_video_info['width'])
        compare_raw_ds = iter(single_raw_dataset(hr_image_dir, compare_video_info['height'], compare_video_info['width'], \
                                            pattern='[0-9][0-9][0-9][0-9].raw'))

        sr_psnr_values = []
        #bilinear_psnr_values = []
        for idx, img in enumerate(input_raw_ds):
            lr = img[0]
            lr = tf.cast(lr, tf.float32)
            sr = model(lr)

            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)

            lr_name = os.path.basename(img[1].numpy()[0].decode())
            sr_image = tf.squeeze(sr).numpy()
            sr_image.tofile(os.path.join(sr_image_dir, lr_name))

            if re.match('\d\d\d\d.raw', lr_name):
                hr, hr_name = next(compare_raw_ds)

                #bilinear = resolve_bilinear(lr, compare_video_info['height'], compare_video_info['width'])
                #bilinear = tf.cast(bilinear, tf.uint8)
                #bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
                #bilinear_psnr_values.append(bilinear_psnr_value)

                sr_psnr_value = tf.image.psnr(hr[0], sr, max_val=255)[0].numpy()
                sr_psnr_values.append(sr_psnr_value)

                #validate name
                #hr_name = os.path.basename(hr_name.numpy()[0].decode())
                #print(lr_name, hr_name)

            #validate image
            #sr_image = tf.image.encode_png(tf.squeeze(sr))
            #tf.io.write_file(os.path.join('.', '{0:04d}.png'.format(idx+1)), sr_image)

        log_path = os.path.join(log_dir, 'quality_sr.txt')
        with open(log_path, 'w') as f:
            f.write('\n'.join(str(np.round(sr_psnr_value, 2)) for sr_psnr_value in sr_psnr_values))

        #log_path = os.path.join(log_dir, 'quality_bilinear.txt')
        #with open(log_path, 'w') as f:
        #    f.write('\n'.join(str(np.round(bilinear_psnr_value, 2)) for bilinear_psnr_value in bilinear_psnr_values))

        end_time = time.time()
        return end_time - start_time

    def _prepare_anchor_points(self, q0, q1):
        tf.enable_eager_execution()
        checkpoint = self.model.load_checkpoint(self.checkpoint_dir)

        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log_path = os.path.join(log_dir, 'latency_{}_0.txt'.format(self.__class__.__name__))
        log_file = open(log_path, 'w')

        while True:
            item = q0.get()
            if item == 'end':
                return
            else:
                start_time = time.time()
                chunk_idx = item
                print('_prepare_anchor_points: start {} chunk'.format(chunk_idx))
                elapsed_time1 = self._prepare_lr_frames(chunk_idx)
                elapsed_time2 = self._prepare_hr_frames(chunk_idx)
                elapsed_time3 = self._prepare_sr_frames(chunk_idx, checkpoint.model)
                q1.put(chunk_idx)
                print('_prepare_anchor_points: end {} chunk'.format(chunk_idx))
                end_time = time.time()
                elapsed_time = end_time - start_time
                log_file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(chunk_idx, elapsed_time, \
                                    elapsed_time1, elapsed_time2, elapsed_time3, start_time, end_time))
                log_file.flush()

        log_file.close()

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
                start_time = time.time()
                subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                end_time = time.time()

                #result
                q1.put([idx, end_time - start_time])

    def _analyze_anchor_points(self, q0, q1):
        q2 = mp.Queue()
        q3 = mp.Queue()
        decoders = [mp.Process(target=self._execute_command, args=(q2, q3)) for i in range(self.num_decoders)]

        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log_path = os.path.join(log_dir, 'latency_{}_1.txt'.format(self.__class__.__name__))
        log_file = open(log_path, 'w')

        for decoder in decoders:
            decoder.start()

        while True:
            item = q0.get()
            if item == 'end':
                break
            else:
                start_time = time.time()
                chunk_idx = item
                print('_analyze_anchor_points: start {} chunk'.format(chunk_idx))

                start_idx = chunk_idx * self.gop
                end_idx = (chunk_idx + 1) * self.gop
                postfix = 'chunk{:04d}'.format(chunk_idx)
                log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name, postfix)
                profile_dir = os.path.join(self.content_dir, 'profile', self.input_video, postfix)
                os.makedirs(profile_dir, exist_ok=True)

                start_time1 = time.time()
                frames = self._load_frames(chunk_idx)
                ap_cache_profiles = []
                for idx, frame in enumerate(frames):
                    ap_cache_profile = CacheProfile.fromframes(frames, profile_dir, 'frame_{}'.format(frame.name))
                    ap_cache_profile.add_anchor_point(frame)
                    ap_cache_profile.save()
                    ap_cache_profiles.append(ap_cache_profile)

                    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
                --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, ap_cache_profile.path)
                    q2.put([command, idx])
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1

                elapsed_time3 = []
                start_time2 = time.time()
                for _ in range(len(frames)):
                    item = q3.get()
                    idx = item[0]
                    elapsed_time3.append(item[1])
                    quality = []
                    path = os.path.join(log_dir, ap_cache_profiles[idx].name, 'quality.txt')
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            quality.append(float(line.split('\t')[1]))
                    ap_cache_profiles[idx].set_measured_quality(quality)
                end_time2 = time.time()
                elapsed_time2 = end_time2 - start_time2

                q1.put([chunk_idx, ap_cache_profiles, frames])
                print('_analyze_anchor_points: end {} chunk'.format(chunk_idx))
                end_time = time.time()
                elapsed_time = end_time - start_time
                log_file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(chunk_idx, elapsed_time, elapsed_time1, \
                                            elapsed_time2, np.average(elapsed_time3), start_time, end_time))
                log_file.flush()

        log_file.close()

        for decoder in decoders:
            q2.put('end')

        for decoder in decoders:
            decoder.join()

    def _select_anchor_point(self, cache_profile, ap_cache_profiles):
        max_avg_quality = None
        idx = None

        for i, ap_cache_profile in enumerate(ap_cache_profiles):
            avg_quality = np.average(self._estimate_quality(cache_profile, ap_cache_profile))
            if idx is None or avg_quality > max_avg_quality:
                max_avg_quality = avg_quality
                idx = i

        return idx

    def _estimate_quality(self, cache_profile, ap_cache_profile):
        if cache_profile is not None:
            return np.maximum(cache_profile.estimated_quality, ap_cache_profile.measured_quality)
        else:
            return ap_cache_profile.measured_quality

    def _select_cache_profile(self, q0, q1):
        tf.enable_eager_execution()
        q2 = mp.Queue()
        q3 = mp.Queue()
        decoders = [mp.Process(target=self._execute_command, args=(q2, q3)) for i in range(self.num_decoders)]

        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log_path = os.path.join(log_dir, 'latency_{}_2.txt'.format(self.__class__.__name__))
        log_file = open(log_path, 'w')

        for decoder in decoders:
            decoder.start()

        while True:
            item = q0.get()
            if item == 'end':
                break
            else:
                start_time = time.time()
                chunk_idx = item[0]
                ap_cache_profiles = item[1]
                frames = item[2]
                print('_select_cache_profile: start {} chunk'.format(chunk_idx))

                start_idx = chunk_idx * self.gop
                end_idx = (chunk_idx + 1) * self.gop
                postfix = 'chunk{:04d}'.format(chunk_idx)
                profile_dir = os.path.join(self.content_dir, 'profile', self.input_video, postfix)
                log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name, postfix)
                os.makedirs(profile_dir, exist_ok=True)

                #load dnn quality
                start_time1 = time.time()
                #deprecated
                """
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
                """

                log_path = os.path.join(log_dir, 'quality_sr.txt')
                dnn_quality = []
                with open(log_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        dnn_quality.append(float(line))
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1

                #select anchor points
                start_time2 = time.time()
                cache_profiles = []
                cache_profile = None
                total = 0
                while len(ap_cache_profiles) > 0:
                    idx = self._select_anchor_point(cache_profile, ap_cache_profiles)
                    ap_cache_profile = ap_cache_profiles.pop(idx)
                    #print('_select_cache_profile: {} anchor points, {} chunk'.format(ap_cache_profile.anchor_points[0].name, chunk_idx))
                    if len(cache_profiles) == 0:
                        cache_profile = CacheProfile.fromcacheprofile(ap_cache_profile, profile_dir, '{}_{}'.format(self.__class__.__name__, len(ap_cache_profile.anchor_points)))
                        cache_profile.set_estimated_quality(ap_cache_profile.measured_quality)
                    else:
                        cache_profile = CacheProfile.fromcacheprofile(cache_profiles[-1], profile_dir, '{}_{}'.format(self.__class__.__name__, len(cache_profiles[-1].anchor_points) + 1))
                        cache_profile.add_anchor_point(ap_cache_profile.anchor_points[0])
                        cache_profile.set_estimated_quality(self._estimate_quality(cache_profiles[-1], ap_cache_profile))
                    cache_profile.save()
                    cache_profiles.append(cache_profile)

                    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
                --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --save-metadata --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, os.path.join(cache_profile.path))
                    q2.put([command, total])
                    total += 1

                    quality_diff = np.average(dnn_quality) - np.average(cache_profile.estimated_quality)
                    if quality_diff < self.quality_diff:
                        break
                end_time2 = time.time()
                elapsed_time2 = end_time2 - start_time2

                elapsed_time4 = []
                #select a cache profile
                start_time3 = time.time()
                min_num_anchor_points = len(frames)
                selected_idx = None
                for _ in range(total):
                    quality = []
                    item = q3.get()
                    idx = item[0]
                    elapsed_time4.append(item[1])
                    path = os.path.join(log_dir, cache_profiles[idx].name, 'quality.txt')
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            quality.append(float(line.split('\t')[1]))
                    cache_profiles[idx].set_measured_quality(quality)

                    quality_diff = np.average(dnn_quality) - np.average(cache_profiles[idx].measured_quality)
                    num_anchor_points = len(cache_profiles[idx].anchor_points)
                    if quality_diff <= self.quality_diff and num_anchor_points <= min_num_anchor_points:
                        selected_idx = idx
                        min_num_anchor_points = num_anchor_points
                end_time3 = time.time()
                elapsed_time3 = end_time3 - start_time3

                #save a selected cache profile
                selected_cache_profile = CacheProfile.fromcacheprofile(cache_profiles[idx], profile_dir, 'cp_final_{}'.format(len(cache_profiles[idx].anchor_points)))
                selected_cache_profile.save()

                #save a log
                quality_dnn = np.average(dnn_quality)
                log_path = os.path.join(log_dir, 'quality_{}.txt'.format(self.__class__.__name__))
                with open(log_path, 'w') as f:
                    for cache_profile in cache_profiles:
                        quality_cache_measured = np.average(cache_profile.measured_quality)
                        quality_diff = quality_dnn - quality_cache_measured
                        quality_cache_error = np.percentile(np.asarray(dnn_quality) - np.asarray(cache_profile.measured_quality), [90, 95, 100], interpolation='nearest')
                        quality_cache_error = '\t'.join(str(np.round(x, 2)) for x in quality_cache_error)
                        quality_cache_estimated = np.average(cache_profile.estimated_quality)

                        log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{:.2f}\n'.format(len(cache_profile.anchor_points), quality_cache_measured, quality_dnn, \
                                                                quality_diff, quality_cache_error, quality_cache_estimated)
                        f.write(log)

                q1.put(chunk_idx)
                print('_select_cache_profile: end {} chunk'.format(chunk_idx))
                end_time = time.time()
                elapsed_time = end_time - start_time
                log_file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(chunk_idx, elapsed_time, elapsed_time1, \
                                            elapsed_time2, elapsed_time3, np.average(elapsed_time4), start_time, end_time))
                log_file.flush()

        log_file.close()

        for decoder in decoders:
            q2.put('end')

        for decoder in decoders:
            decoder.join()

    def start_process(self):
        self.p0.start()
        self.p1.start()
        self.p2.start()
        pass

    def stop_process(self):
        self.q0.put('end')
        self.q1.put('end')
        self.q2.put('end')

        self.p0.join()
        self.p1.join()
        self.p2.join()

    def run_asynchrnous(self, chunk_idx=None):
        if chunk_idx is None:
            input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
            input_video_info = profile_video(input_video_path)
            num_chunks = int(input_video_info['duration'] // (self.gop / input_video_info['frame_rate']))
            for i in range(num_chunks):
                self.q0.put(i)
            for i in range(num_chunks):
                self.q3.get()
        else:
            self.q0.put(chunk_idx)

    def run_synchrnous(self, chunk_idx=None):
        if chunk_idx is None:
            input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
            input_video_info = profile_video(input_video_path)
            num_chunks = int(input_video_info['duration'] // (self.gop / input_video_info['frame_rate']))
            for i in range(num_chunks):
                self.q0.put(i)
                self.q3.get()
        else:
            self.q0.put(chunk_idx)
            self.q3.get()

    def debug_synchrnous(self, chunk_idx=None):
        if chunk_idx is None:
            input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
            input_video_info = profile_video(input_video_path)
            num_chunks = int(input_video_info['duration'] // (self.gop / input_video_info['frame_rate']))
            for i in range(num_chunks):
                self.q0.put(i)
                self.q0.put('end')
                self._prepare_anchor_points(self.q0, self.q1)
                self.q1.put('end')
                self._analyze_anchor_points(self.q1, self.q2)
                self.q2.put('end')
                self._select_cache_profile(self.q2, self.q3)
        else:
            self.q0.put(chunk_idx)
            self.q0.put('end')
            self._prepare_anchor_points(self.q0, self.q1)
            self.q1.put('end')
            self._analyze_anchor_points(self.q1, self.q2)
            self.q2.put('end')
            self._select_cache_profile(self.q2, self.q3)

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

    #options for anchor point selector
    parser.add_argument('--quality_diff', type=float, required=True)

    #options for mode
    parser.add_argument('--chunk_idx', default=None)
    parser.add_argument('--mode', default='async', choices=['async', 'sync', 'debug'])

    args = parser.parse_args()

    input_video_path = os.path.join(args.content_dir, 'video', args.input_video_name)
    compare_video_path = os.path.join(args.content_dir, 'video', args.compare_video_name)
    input_video_info = profile_video(input_video_path)
    compare_video_info = profile_video(compare_video_path)

    scale = int(compare_video_info['height'] / input_video_info['height'])
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, scale, None)
    checkpoint_dir = os.path.join(args.checkpoint_dir, edsr_s.name)

    aps_v1 = APS_v1(edsr_s, checkpoint_dir, args.vpxdec_path, args.content_dir, args.input_video_name, args.compare_video_name, args.num_decoders, args.gop, args.quality_diff)
    if args.mode == 'async':
        aps_v1.start_process()
        aps_v1.run_asynchrnous(args.chunk_idx)
        aps_v1.stop_process()
    elif args.mode == 'sync':
        aps_v1.start_process()
        aps_v1.run_synchrnous(args.chunk_idx)
        aps_v1.stop_process()
    elif args.mode == 'debug':
        aps_v1.debug_synchrnous(args.chunk_idx)
