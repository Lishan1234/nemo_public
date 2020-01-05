import os
import sys
import argparse
import glob
import time
import struct
import subprocess
import shlex
import multiprocessing as mp

import scipy.misc
import numpy as np
#import tensorflow as tf
from tqdm import tqdm

from tool.ffprobe import profile_video
#from tool.tf import single_raw_dataset
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

class CRA():
    def __init__(self, model, checkpoint_dir, vpxdec_path, content_dir, input_video, compare_video, num_decoders, gop):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.vpxdec_path = vpxdec_path
        self.content_dir = content_dir
        self.input_video = input_video
        self.compare_video = compare_video
        self.num_decoders = num_decoders
        self.gop = gop
        self.frames = None

        q0 = mp.Queue()
        q1 = mp.Queue()
        q2 = mp.Queue()
        q3 = mp.Queue()

        p0 = mp.Process(target=self.prepare_anchor_points, args=(q0, q1))
        p1 = mp.Process(target=self.analyze_anchor_points, args=(q1, q2))
        p2 = mp.Process(target=self.select_cache_profile, args=(q2, q3))

    def _prepare_hr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        os.makedirs(image_dir, exist_ok=True)
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={}'.format(self.vpxdec_path,
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

    def _prepare_sr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        sr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, self.model.name, postfix)
        os.makedirs(sr_image_dir, exist_ok=True)

        input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
        input_video_info = profile_video(input_video_path)

        single_raw_ds = single_raw_dataset(lr_image_dir, input_video_info['height'], input_video_info['width'])
        sr_psnr_values = []
        for idx, img in tqdm(enumerate(single_raw_ds)):
            lr = img[0]
            lr = tf.cast(lr, tf.float32)
            sr = self.model(lr)

            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            sr = tf.squeeze(sr).numpy()

            #validate
            #sr_png = tf.image.encode_png(sr)
            #tf.io.write_file(os.path.join('.', 'tmp.png'), sr_png)

            #TODO: measure PSNR value and log (quality.txt)

            name = os.path.basename(img[1].numpy()[0].decode())
            sr.tofile(os.path.join(sr_image_dir, name))

    def _load_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        frames = []
        metadata_log_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'metadata.txt')
        with open(metadata_log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                current_video_frame = int(line.split('\t')[0])
                current_super_frame = int(line.split('\t')[1])
                frames.append(Frame(current_video_frame, current_super_frame))

        return frames

    def _select_anchor_point(self, cache_profile, anchor_points):
        max_quality = None
        target_anchor_point = None

        for anchor_point in anchor_points:
            quality = self._estimate_quality(cache_profile, anchor_point)
            if target_anchor_point is None or np.average(quality) > np.average(max_quality):
                target_anchor_point = anchor_point
                max_quality = quality

        return target_anchor_point, max_quality

    def _estimate_quality(self, cache_profile, anchor_point):
        if cache_profile is not None:
            return np.maximum(cache_profile.estimated_quality, anchor_point.measured_quality)
        else:
            return anchor_point.estimated_quality

    def start_process(self):
        p0.start()
        p1.start()
        p2.start()

    def stop_process(self):
        q0.put('end')
        q1.put('end')
        q2.put('end')

        p0.join()
        p1.join()
        p2.join()

    def run_aynchrnous(self. chunk_idx):
        q0.put(chunk_idx)

    def run_synchrnous(self, chunk_idx):
        q0.put(chunk_idx)
        q3.get()

    def _prepare_anchor_points(self, q0, q1):
        import tensorflow as tf
        tf.enable_eager_execution()
        checkpoint = self.model.load_checkpoint(self.checkpoint_dir)

        while True:
            item = q0.get()
            if item == 'end':
                return
            else:
                chunk_idx = item
                self._prepare_lr_frames(chunk_idx)
                self._prepare_hr_frames(chunk_idx)
                self._prepare_sr_frames(chunk_idx, checkpoint.model)
                q1.put(chunk_idx)

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

    def analyze_anchor_points(self, q0, q1):
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

                start_idx = chunk_idx * self.gop
                end_idx = (chunk_idx + 1) * self.gop
                postfix = 'chunk{:04d}'.format(chunk_idx)
                profile_dir = os.path.join(self.content_dir, 'profile', self.input_video, postfix)
                os.makedirs(profile_dir, exist_ok=True)

                frames = self.load_frames(chunk_idx)
                ap_cache_profiles = []
                for idx, frame in enumerate(frames):
                    ap_cache_profile = CacheProfile.fromframes(frames, profile_dir, 'ap_{}'.format(frame.name))
                    ap_cache_profile.add_anchor_point(frame)
                    ap_cache_profile.save()
                    ap_cache_profiles.append(ap_cache_profile)

                    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
                --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, ap_cache_profile.path)
                    q2.put([command, idx])

                for _ in range(len(frames)):
                    idx = q3.get()
                    quality = []
                    path = os.path.join(log_dir, ap_cache_profiles[idx].name, 'quality.txt')
                    with open(path, 'r') as f:
                        lines = f.readline()
                        for line in lines:
                            line = line.strip()
                            quality.append(int(line))
                    ap_cache_profiles[idx].measured_quality = quality

                q1.put([chunk_idx, ap_cache_profiles, frames])

        for decoder in decoders:
            q2.put('end')

        for decoder in decoders:
            decoder.join()

    def select_cache_profile(self):
        q_cache = mp.Queue()
        decoders = [mp.Process(target=self._run_cache_profile, args=(q_cache,)) for i in range(self.num_decoders)]

        while True:
            item = q_analyze.get()
            if item == 'end':
                break
            else:
                chunk_idx = item[0]
                ap_cache_profiles = item[1]
                frames = item[2]

                start_idx = chunk_idx * self.gop
                end_idx = (chunk_idx + 1) * self.gop
                postfix = 'chunk{:04d}'.format(chunk_idx)
                profile_dir = os.path.join(self.content_dir, 'profile', self.input_video, postfix)
                log_dir = os.path.join(self.content_dir, 'log', self.input_video, postfix)
                os.makedirs(profile_dir, exist_ok=True)

                #load dnn quality

                #select anchor points
                cache_profiles = []
                cache_profile = None
                idx = 0
                while len(ap_cache_profiles) > 0:
                    anchor_point, quality = self._select_anchor_point(cache_profile, ap_cache_profiles)
                    if len(cache_profiles) == 0:
                        cache_profile = CacheProfile.fromframes(frames, profile_dir, 'cp_{}'.format(len(cache_profile.anchor_points)))
                    else:
                        cache_profile = CacheProfile.fromcacheprofile(cache_profiles[-1], profile_dir, 'cp_{}'.format(len(cache_profile.anchor_points)))
                    cache_profile.add_anchor_point(anchor_point)
                    cache_profile.estimate_quality = quality
                    cache_profiles.append(cache_profile)

                    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
                --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, os.path.join(cache_profile.path)
                    q2.put([command, idx])
                    idx += 1

                    if np.average(dnn_quality - quality) < self.quality_diff:
                        break

                #select a cache profile
                min_num_anchor_points = len(frames)
                selected_idx = None
                total = idx
                for _ in enumerate(range(total)):
                    quality = []
                    idx = q3.get()
                    path = os.path.join(log_dir, cache_profiles[idx].name, 'quality.txt')
                    with open(path, 'r') as f:
                        lines = f.readline()
                        for line in lines:
                            line = line.strip()
                            quality.append(int(line))
                    cache_profiles[idx].measured_quality = quality

                    quality_diff = np.average(dnn_quality - cache_profiles[idx].measured_quality)
                    num_anchor_points = len(cache_profile[idx].anchor_points)
                    if quality_diff <= self.quality_diff and num_anchor_points <= min_num_anchor_points:
                        selected_idx = idx
                        min_num_anchor_points = num_anchor_points

                #save a selected cache profile
                selected_cache_profile = CacheProfile.fromcacheprofile(cache_profiles[idx], profile_dir, 'cp_final_{}'.format(selected_cache_profile.count_anchor_points()))
                selected_cache_profile.save()

                #save a log
                if self.debug:
                    log_path = os.path.join(log_dir, 'anchor_point_selection.txt')
                    with open(log_path, 'w') as f:
                        for cache_profile in cache_profiles:
                            quality_error = np.percentile(dnn_quality - cache_profile.measured_quality, [90, 95, 100], interpolation='nearest')
                            quality_error = '\t'.join(np.round(x, 2) for x in quality_error)
                            log = '{}\t{:.2f}\t{:.2f}\t{}\n'.format(len(cache_profile.anchor_points), np.average(cache_profile.estimated_quality), \
                                                    np.average(cache_profile.measured_quality), quality_error)
                            f.write(log)

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

    args = parser.parse_args()

    input_video_path = os.path.join(args.content_dir, 'video', args.input_video_name)
    compare_video_path = os.path.join(args.content_dir, 'video', args.compare_video_name)
    input_video_info = profile_video(input_video_path)
    compare_video_info = profile_video(compare_video_path)

    scale = int(compare_video_info['height'] / input_video_info['height'])
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, scale, None)
    checkpoint_dir = os.path.join(args.checkpoint_dir, edsr_s.name)
    checkpoint = edsr_s.load_checkpoint(checkpoint_dir)
    cra = CRA(checkpoint.model, args.vpxdec_path, args.content_dir, args.input_video_name, args.compare_video_name, args.num_decoders, args.gop)
    #cra._prepare_hr_frames(0)
    #cra.profile_anchor_points(0)

    p = mp.Process(target=cra.print)
    p.start()
    p.join()
