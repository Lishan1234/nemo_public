import math
import os
import struct
import copy
import subprocess
import shlex

import tensorflow as tf

from tool.video import profile_video
from dnn.dataset import single_raw_dataset

class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index

    @property
    def name(self):
        return '{}.{}'.format(self.video_index, self.super_index)

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.video_index == other.video_index and self.super_index == other.super_index:
                return True
            else:
                return False
        else:
            return False

class CacheProfile():
    def __init__(self, frames, cache_profile, save_dir, name):
        assert (frames is None or cache_profile is None)

        if frames is not None:
            self.frames = frames
            self.anchor_points = []
            self.estimated_quality = None
            self.measured_quality = None

        if cache_profile is not None:
            self.frames = copy.deepcopy(cache_profile.frames)
            self.anchor_points = copy.deepcopy(cache_profile.anchor_points)
            self.estimated_quality = copy.deepcopy(cache_profile.estimated_quality)
            self.measured_quality = copy.deepcopy(cache_profile.measured_quality)

        self.save_dir = save_dir
        self.name = name

    @classmethod
    def fromframes(cls, frames, save_dir, name):
        return cls(frames, None, save_dir, name)

    @classmethod
    def fromcacheprofile(cls, cache_profile, save_dir, name):
        return cls(None, cache_profile, save_dir, name)

    @property
    def path(self):
        return os.path.join(self.save_dir, self.name)

    def add_anchor_point(self, frame, quality=None):
        self.anchor_points.append(frame)
        self.quality = quality

    def count_anchor_points(self):
        return len(self.anchor_points)

    def set_estimated_quality(self, quality):
        self.estimated_quality = quality

    def set_measured_quality(self, quality):
        self.measured_quality = quality

    def save(self):
        path = os.path.join(self.save_dir, self.name)
        with open(path, "wb") as f:
            byte_value = 0
            for i, frame in enumerate(self.frames):
                if frame in self.anchor_points:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frames) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

def libvpx_save_frames(vpxdec_file, content_dir, video_name, gop, chunk_idx):
    start_idx = chunk_idx * gop
    end_idx = (chunk_idx + 1) * gop
    postfix = 'chunk{:04d}'.format(chunk_idx)

    lr_image_dir = os.path.join(content_dir, 'image', video_name, postfix)
    os.makedirs(lr_image_dir, exist_ok=True)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} \
            --content-dir={} --input-video={} --postfix={} --save-frame --save-metadata'.format(vpxdec_file, \
            start_idx, end_idx - start_idx, content_dir, video_name, postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def libvpx_save_metadata(vpxdec_file, content_dir, video_name, gop, chunk_idx):
    start_idx = chunk_idx * gop
    end_idx = (chunk_idx + 1) * gop
    postfix = 'chunk{:04d}'.format(chunk_idx)

    lr_image_dir = os.path.join(content_dir, 'image', video_name, postfix)
    os.makedirs(lr_image_dir, exist_ok=True)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} \
            --content-dir={} --input-video={} --postfix={} --save-metadata'.format(vpxdec_file, \
            start_idx, end_idx - start_idx, content_dir, video_name, postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def libvpx_load_frames(content_dir, video_name, gop, chunk_idx):
    start_idx = chunk_idx * gop
    end_idx = (chunk_idx + 1) * gop
    postfix = 'chunk{:04d}'.format(chunk_idx)

    frames = []
    log_path = os.path.join(content_dir, 'log', video_name, postfix, 'metadata.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            current_video_frame = int(line.split('\t')[0])
            current_super_frame = int(line.split('\t')[1])
            frames.append(Frame(current_video_frame, current_super_frame))

    return frames

def libvpx_setup_sr_frames(vpxdec_file, content_dir, video_name, gop, chunk_idx, model):
    start_idx = chunk_idx * gop
    end_idx = (chunk_idx + 1) * gop
    postfix = 'chunk{:04d}'.format(chunk_idx)

    lr_image_dir = os.path.join(content_dir, 'image', video_name, postfix)
    sr_image_dir = os.path.join(content_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    input_video_path = os.path.join(content_dir, 'video', video_name)
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

def measure_offline_cache_quality(vpxdec_file, content_dir, input_video, compare_video, postfix, model, cache_profile):
    #run sr-integrated decoder
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --skip={} --limit={} --content-dir={} \
    --input-video={} --compare-video={} --postfix={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
    --save-quality --dnn-name={} --cache-profile={}'.format(vpxdec_file, start_idx, end_idx - start_idx, content_dir, input_video, compare_video, postfix, model, cache_profile)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    log_dir = os.path.join(content_dir, 'log', input_video, model, postfix)
    log_file = os.log_file.join(log_dir, cache_profiles[idx].name, 'quality.txt')
    quality = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

#ref: https://developers.google.com/media/vp9/settings/vod
def get_num_threads(resolution):
    tile_size = 256
    if resolution >= tile_size:
        num_tiles = resolution // tile_size
        log_num_tiles = math.floor(math.log(num_tiles, 2))
        num_threads = (2**log_num_tiles) * 2
    else:
        num_threads = 2
    return num_threads

def count_mac_for_cache(width, height, channel):
    return width * height * channel * 8

if __name__ == '__main__':
    frame_list = [Frame(0,1)]
    frame1 = Frame(0,1)
    print(frame1 == frame_list[0])
