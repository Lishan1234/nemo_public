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

def run_cache_profile(cache_profile, path, command):
    with open(path, "wb") as f:
        byte_value = 0
        for i, frame in enumerate(cache_profile.frames):
            if frame in cache_profile.anchor_points:
                byte_value += 1 << (i % 8)

            if i % 8 == 7:
                f.write(struct.pack("=B", byte_value))
                byte_value = 0

        if len(cache_profile.frames) % 8 != 0:
            f.write(struct.pack("=B", byte_value))

    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def run_cache_profile_mt(q):
    while True:
        item = q.get()
        if item == 'end':
            print('end')
            return
        else:
            cache_profile = item[0]
            path = item[1]
            command = item[2]

            with open(path, "wb") as f:
                byte_value = 0
                for i, frame in enumerate(cache_profile.frames):
                    if frame in cache_profile.anchor_points:
                        byte_value += 1 << (i % 8)

                    if i % 8 == 7:
                        f.write(struct.pack("=B", byte_value))
                        byte_value = 0

                if len(cache_profile.frames) % 8 != 0:
                    f.write(struct.pack("=B", byte_value))

            print(path)

            subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

#TODO: select proper number of threads for each resolution
class CRA():
    def __init__(self, model, vpxdec_path, content_dir, input_video, compare_video, num_decoders, gop):
        self.model = model
        self.vpxdec_path = vpxdec_path
        self.content_dir = content_dir
        self.input_video = input_video
        self.compare_video = compare_video
        self.num_decoders = num_decoders
        self.gop = gop
        self.frames = None

        q_prepare = mp.Queue()
        q_analyze = mp.Queue()
        q_result = mp.Queue()

        p_prepare = mp.Process(target=self.prepare, args=(q_prepare, q_analyze))
        p_analyze = mp.Process(target=self.analyze, args=(q_analyze, q_result))
        p_prepare.start()
        p_analyze.start()

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

    def _run_cache_profile(self, q_cache, q_result):
        while True:
            item = q_cache.get()
            if item == 'end':
                return
            else:
                #setup
                frames = item[0]
                frame = item[1]
                chunk_idx = item[2]
                last_profile = item[3]

                start_idx = chunk_idx * self.gop
                end_idx = (chunk_idx + 1) * self.gop
                postfix = 'chunk{:04d}'.format(chunk_idx)

                cache_profile = CacheProfile.fromframes(frames)
                cache_profile.add_anchor_point(frame)
                path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'cra_ap{}'.format(frame.name))
                cmd = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
                --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, path)

                #save
                with open(path, "wb") as f:
                    byte_value = 0
                    for i, frame in enumerate(cache_profile.frames):
                        if frame in cache_profile.anchor_points:
                            byte_value += 1 << (i % 8)

                        if i % 8 == 7:
                            f.write(struct.pack("=B", byte_value))
                            byte_value = 0

                    if len(cache_profile.frames) % 8 != 0:
                        f.write(struct.pack("=B", byte_value))

                #run
                subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                #result
                if last_profile:
                    queue_result.put(chunk_idx)

    #TODO: handle cache_profiles
    def run_parallel(self, chunk_idx):
        if cache_prfiles is None:
            self.q_prepare.put(chunk_idx)

    def get_parallel(self, chunk_idx):
        item = self.q_result.get()
        return item

    def close_parallel():
        self.q_prepare.put('end')
        self.p_prepare.join()
        self.p_analyze.join()

    #TODO: model
    def prepare(self, q_prepare, q_analyze):
        import tensorflow as tf
        tf.enable_eager_execution()
        checkpoint = self.model.load_checkpoint(self.checkpoint_dir)

        while True:
            item = q_prepare.get()
            if item == 'end':
                q_analyze.put('end')
                return
            else:
                chunk_idx = item
                self._prepare_lr_frames(chunk_idx)
                self._prepare_hr_frames(chunk_idx)
                self._prepare_sr_frames(chunk_idx, checkpoint.model)

                q_analyze.put(chunk_idx)

    #TODO: _run_cache_profile to handle 2 cases
    def analyze(self, q_analyze, q_result):
        q_cache = mp.Queue()
        decoders = [mp.Process(target=self._run_cache_profile, args=(q_cache,)) for i in range(self.num_decoders)]

        for decoder in decoders:
            decoder.start()

        while True:
            item = q_analyze.get()
            if item == 'end':
                break
            else:
                chunk_idx = item
                frames = self.load_frames(chunk_idx)
                for i, frame in enumerate(frames):
                    if i == len(frames) - 1:
                        q_cache.put([frames, frame, chunk_idx, True])
                    else:
                        q_cache.put([frames, frame, chunk_idx, False])

        for decoder in decoders:
            q_cache.put('end')

        for decoder in decoders:
            decoder.join()

    #TODO: remove, deprecated
    def prepare(self, chunk_idx):
        self._prepare_hr_frames(chunk_idx)
        self._prepare_lr_frames(chunk_idx)
        self._prepare_sr_frames(chunk_idx)

    #TODO: remove, deprecated
    def profile_anchor_points(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        #TODO: multi-thread version
        frames = self._load_frames(chunk_idx)
        """
        start_time = time.time()
        for frame in frames:
            cache_profile = CacheProfile.fromframes(frames)
            cache_profile.add_anchor_point(frame)
            path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'cra_ap{}'.format(frame.name))
            cmd = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
            --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
            --save-quality --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, path)
            run_cache_profile(cache_profile, path, cmd)
        """

        q = mp.Queue()
        decoders = [mp.Process(target=run_cache_profile_mt, args=(q,)) for i in range(self.num_decoders)]

        start_time = time.time()
        for decoder in decoders:
            decoder.start()

        for frame in frames[0:4]:
            cache_profile = CacheProfile.fromframes(frames)
            cache_profile.add_anchor_point(frame)
            path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'cra_ap{}'.format(frame.name))
            cmd = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
            --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
            --save-quality --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, path)
            q.put([cache_profile, path, cmd])

        for decoder in decoders:
            q.put('end')

        for decoder in decoders:
            decoder.join()

        print(time.time() - start_time)

def print_tmp():
    import tensorflow as tf
    tf.enable_eager_execution()
    input_tensor = tf.random.uniform((1, 200, 200, 3), 0, 255)
    print(input_tensor)
    print(1)

if __name__ == '__main__':
    #tf.enable_eager_execution()

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
