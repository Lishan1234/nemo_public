import os
import sys
import argparse
import glob
import time
import struct

import scipy.misc
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tool.ffprobe import profile_video
from tool.tensorflow import single_raw_dataset
from tool.libvpx import Frame

from dnn.model.edsr_s import EDSR_S

#TODO: fastest png converter before latency measurement
#Note: load a model outside of CRA
class CRA():
    def __init__(self, model, vpxdec_path, content_dir, input_video, compare_video, num_threads, gop):
        self.model = model
        self.vpxdec_path = vpxdec_path
        self.content_dir = content_dir
        self.input_video = input_video
        self.compare_video = compare_video
        self.num_threads = num_threads
        self.gop = gop
        self.frames = None

    def _prepare_hr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        #check exists
        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)

        #video metdata
        video_path = os.path.join(self.content_dir, 'video', self.compare_video)
        video_info = profile_video(video_path)

        #vpxdec (hr, save_frame)
        os.makedirs(hr_image_dir, exist_ok=True)
        cmd = '{} --codec=vp9 --noblit --threads={} --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame'.format(self.vpxdec_path, self.num_threads, \
                start_idx, end_idx - start_idx, self.content_dir, self.compare_video, postfix)
        os.system(cmd)

        #convert raw to png (deprecated)
        """
        images = glob.glob(os.path.join(hr_image_dir, '*.raw'))
        for idx, image in enumerate(images):
            arr = np.fromfile(image, dtype=np.uint8)
            arr = np.reshape(arr, (video_info['height'], video_info['width'], 3))
            name = os.path.splitext(os.path.basename(image))[0]
            name += '.png'
            scipy.misc.imsave(os.path.join(hr_image_dir, name), arr)
        """

    def _prepare_sr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        #check exists
        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        lr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        sr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, self.model.name, postfix)

        #video metdata
        input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
        compare_video_path = os.path.join(self.content_dir, 'video', self.compare_video)
        input_video_info = profile_video(input_video_path)
        compare_video_info = profile_video(compare_video_path)

        #vpxdec (lr, save_frame)
        #os.makedirs(lr_image_dir)
        cmd = '{} --codec=vp9 --noblit --threads={} --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame'.format(self.vpxdec_path, self.num_threads, \
                start_idx, end_idx - start_idx, self.content_dir, self.input_video, postfix)
        os.system(cmd)

        #convert raw to png (deprecated)
        """
        images = glob.glob(os.path.join(lr_image_dir, '*.raw'))
        for idx, image in enumerate(images):
            arr = np.fromfile(image, dtype=np.uint8)
            arr = np.reshape(arr, (video_info['height'], video_info['width'], 3))
            name = os.path.splitext(os.path.basename(image))[0]
            name += '.png'
            scipy.misc.imsave(os.path.join(lr_image_dir, name), arr)
        """

        #sr raw files
        os.makedirs(sr_image_dir, exist_ok=True)
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
            """
            sr_png = tf.image.encode_png(sr)
            tf.io.write_file(os.path.join('.', 'tmp.png'), sr_png)
            """

            name = os.path.basename(img[1].numpy()[0].decode())
            sr.tofile(os.path.join(sr_image_dir, name))

    def _prepare_frame_index(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        self.frames = []
        metadata_log_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'metadata.txt')
        with open(metadata_log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                current_video_frame = int(line.split('\t')[0])
                current_super_frame = int(line.split('\t')[1])
                self.frames.append(Frame(current_video_frame, current_super_frame))

    def prepare(self, chunk_idx):
        self._prepare_hr_frames(chunk_idx)
        self._prepare_sr_frames(chunk_idx)

    def save_cache_profile(self, chunk_idx, new_anchor_point, existing_anchor_points):
        if self.frames is None:
            self._prepare_frame_index(chunk_idx)

        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'cra_{}_{}'.format(new_anchor_point.name, len(existing_anchor_points)))

        with open(cache_profile_path, "wb") as f:
            byte_value = 0
            for i, frame in enumerate(self.frames):
                if frame == new_anchor_point or frame in existing_anchor_points:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frames) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    #TODO
    def run_cache_profile(self, chunk_idx, new_anchor_point, existing_anchor_points):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'cra_{}_{}'.format(new_anchor_point.name, len(existing_anchor_points)))

        cmd = '{} --codec=vp9 --noblit --threads={} --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --compare-video={} --postfix={} --decode-mode=2 \
                --dnn-mode=2 --cache-policy=1 --save-quality --save-metadata --dnn-name={} --cache-profile={}'.format(self.vpxdec_path, self.num_threads, \
                start_idx, end_idx - start_idx, self.content_dir, self.input_video, self.compare_video, postfix, self.model.name, cache_profile_path)
        print(cmd)
        os.system(cmd)

    #TODO: quality_sr.txt

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Cache Erosion Analyzer')

    #options for libvpx
    parser.add_argument('--vpxdec_path', type=str, required=True)
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--input_video_name', type=str, required=True)
    parser.add_argument('--compare_video_name', type=str, required=True)
    parser.add_argument('--num_threads', type=int, default=1)
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

    cra = CRA(checkpoint.model, args.vpxdec_path, args.content_dir, args.input_video_name, args.compare_video_name, args.num_threads, args.gop)
    #cra.prepare(0)
    #cra._prepare_sr_frames(0)
    #cra.save_cache_profile(0, Frame(0,0), [])
    cra.run_cache_profile(0, Frame(0,0), [])
