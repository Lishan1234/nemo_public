import math
import os
import struct
import copy
import subprocess
import shlex
import time
import gc

import tensorflow as tf

from nemo.tool.video import profile_video
from nemo.dnn.dataset import single_raw_dataset, single_raw_dataset_with_name

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

class AnchorPointSet():
    def __init__(self, frames, anchor_point_set, save_dir, name):
        assert (frames is None or anchor_point_set is None)

        if frames is not None:
            self.frames = frames
            self.anchor_points = []
            self.estimated_quality = None
            self.measured_quality = None

        if anchor_point_set is not None:
            self.frames = copy.deepcopy(anchor_point_set.frames)
            self.anchor_points = copy.deepcopy(anchor_point_set.anchor_points)
            self.estimated_quality = copy.deepcopy(anchor_point_set.estimated_quality)
            self.measured_quality = copy.deepcopy(anchor_point_set.measured_quality)

        self.save_dir = save_dir
        self.name = name

    @classmethod
    def create(cls, frames, save_dir, name):
        return cls(frames, None, save_dir, name)

    @classmethod
    def load(cls, anchor_point_set, save_dir, name):
        return cls(None, anchor_point_set, save_dir, name)

    @property
    def path(self):
        return os.path.join(self.save_dir, self.name)

    def add_anchor_point(self, frame, quality=None):
        self.anchor_points.append(frame)
        self.quality = quality

    def num_anchor_points(self):
        return len(self.anchor_points)

    def set_cache_profile_name(self, name):
        self.name = name

    def get_estimated_quality(self, quality):
        return self.estimated_quality

    def get_measured_quality(self, quality):
        return self.measured_quality

    def set_estimated_quality(self, quality):
        self.estimated_quality = quality

    def set_measured_quality(self, quality):
        self.measured_quality = quality

    def save_cache_profile(self):
        path = os.path.join(self.save_dir, self.name)

        num_remained_bits = 8 - (len(self.frames) % 8)
        num_remained_bits = num_remained_bits % 8

        with open(path, "wb") as f:
            f.write(struct.pack("=I", num_remained_bits))

            byte_value = 0
            for i, frame in enumerate(self.frames):
                if frame in self.anchor_points:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frames) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def remove_cache_profile(self):
        cache_profile_path = os.path.join(self.save_dir, self.name)
        if os.path.exist(cache_profile_path):
            os.remove(path)

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

def libvpx_load_frame_index(content_dir, video_name, postfix=None):
    frames = []
    if postfix is None:
        log_path = os.path.join(content_dir, 'log', video_name, 'metadata.txt')
    else:
        log_path = os.path.join(content_dir, 'log', video_name, postfix, 'metadata.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            current_video_frame = int(line.split('\t')[0])
            current_super_frame = int(line.split('\t')[1])
            frames.append(Frame(current_video_frame, current_super_frame))

    return frames

def libvpx_save_frame(vpxdec_file, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None):
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads=16 --save-rgbframe'.format(vpxdec_file, dataset_dir, video_name)
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def libvpx_save_metadata(vpxdec_file, dataset_dir, video_name, skip=None, limit=None, postfix=None):
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --content={} \
        --input-video-name={} --threads=16 --save-rgbframe'.format(vpxdec_file, dataset_dir, content, video_name)
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def libvpx_setup_sr_frame(vpxdec_file, dataset_dir, video_name, model, postfix=None):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    input_video_path = os.path.join(dataset_dir, 'video', video_name)
    input_video_info = profile_video(input_video_path)

    single_raw_ds = single_raw_dataset_with_name(lr_image_dir, input_video_info['width'], input_video_info['height'], 3, exp='.raw')
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
        #tf.io.write_file(os.path.join(sr_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

def libvpx_bilinear_quality(vpxdec_file, dataset_dir, input_video_name, reference_video_name,
                               output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_file = os.path.join(dataset_dir, 'log', input_video_name, postfix, 'quality.txt')
    else:
        log_file = os.path.join(dataset_dir, 'log', input_video_name, 'quality.txt')

    #run sr-integrated decoder
    if not os.path.exists(log_file):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
            --output-width={} --output-height={} --save-quality'.format(vpxdec_file, dataset_dir, input_video_name, reference_video_name, \
                                                                        output_width, output_height)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def libvpx_offline_dnn_quality(vpxdec_file, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_file = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, 'quality.txt')
    else:
        log_file = os.path.join(dataset_dir, 'log', input_video_name, model_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = profile_video(input_video_path)['height']
    scale = output_height // input_resolution

    if not os.path.exists(log_file):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
        --dnn-scale={} --dnn-name={} --output-width={} --output-height={} --decode-mode=decode_sr --dnn-mode=offline_dnn --save-quality'.format(vpxdec_file, \
                                dataset_dir, input_video_name, reference_video_name, scale, model_name, output_width, output_height)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def libvpx_save_cache_frame(vpxdec_file, content_dir, input_video_name, reference_video_name,  \
                                model_name, cache_profile_file, resolution, skip=None, limit=None, postfix=None):
    #log file
    log_dir = os.path.join(content_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_file))
    if postfix is not None:
        log_dir = os.path.join(log_dir, postfix)
    log_file = os.path.join(log_dir, 'quality.txt')

    #run sr-integrated decoder
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --content-dir={} \
    --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
    --save-quality --save-frame --save-metadata --dnn-name={} --cache-profile={} --resolution={}'.format(vpxdec_file, content_dir, input_video_name, \
                                                    reference_video_name, model_name, cache_profile_file, resolution)
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    #subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL)

def libvpx_offline_cache_quality(vpxdec_file, content_dir, input_video_name, reference_video_name,  \
                                model_name, cache_profile_file, resolution, skip=None, limit=None, postfix=None):
    #log file
    log_dir = os.path.join(content_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_file))
    if postfix is not None:
        log_dir = os.path.join(log_dir, postfix)
    log_file = os.path.join(log_dir, 'quality.txt')

    #run sr-integrated decoder
    if not os.path.exists(log_file):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --content-dir={} \
        --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
        --save-quality --save-metadata --dnn-name={} --cache-profile={} --resolution={}'.format(vpxdec_file, content_dir, input_video_name, \
                                                        reference_video_name, model_name, cache_profile_file, resolution)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        #subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def libvpx_offline_cache_quality_mt_v1(q0, q1, vpxdec_file, content_dir, input_video_name, reference_video_name, model_name, resolution):
    while True:
        item = q0.get()
        if item == 'end':
            return
        else:
            start_time = time.time()
            anchor_point_set = item[0]
            skip = item[1]
            limit = item[2]
            postfix = item[3]

            #log file
            log_dir = os.path.join(content_dir, 'log', input_video_name, model_name, os.path.basename(anchor_point_set.path))
            if postfix is not None:
                log_dir = os.path.join(log_dir, postfix)
            log_file = os.path.join(log_dir, 'quality.txt')

            #run sr-integrated decoder
            anchor_point_set.save()
            command = '{} --codec=vp9 --noblit --frame-buffers=50 --content-dir={} \
            --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
            --save-quality --save-metadata --dnn-name={} --cache-profile={} --resolution={}'.format(vpxdec_file, content_dir, input_video_name, \
                                                            reference_video_name, model_name, anchor_point_set.path, resolution)
            if skip is not None:
                command += ' --skip={}'.format(skip)
            if limit is not None:
                command += ' --limit={}'.format(limit)
            if postfix is not None:
                command += ' --postfix={}'.format(postfix)
            subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            anchor_point_set.remove()

            #load quality from a log file
            quality = []
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    quality.append(float(line.split('\t')[1]))
            end_time = time.time()

            q1.put(quality)

def libvpx_offline_cache_quality_mt(q0, q1, vpxdec_file, content_dir, input_video_name, reference_video_name, model_name, resolution):
    while True:
        item = q0.get()
        if item == 'end':
            return
        else:
            start_time = time.time()
            cache_profile_file = item[0]
            skip = item[1]
            limit = item[2]
            postfix = item[3]
            idx = item[4]

            #log file
            log_dir = os.path.join(content_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_file))
            if postfix is not None:
                log_dir = os.path.join(log_dir, postfix)
            log_file = os.path.join(log_dir, 'quality.txt')

            #run sr-integrated decoder
            if not os.path.exists(log_file):
            #if True:
                command = '{} --codec=vp9 --noblit --frame-buffers=50 --content-dir={} \
                --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                --save-quality --save-metadata --dnn-name={} --cache-profile={} --resolution={}'.format(vpxdec_file, content_dir, input_video_name, \
                                                                reference_video_name, model_name, cache_profile_file, resolution)
                #command = '{} --codec=vp9 --noblit --frame-buffers=50 --content-dir={} \
                #--input-video={} --compare-video={} --decode-mode=2 --dnn-mode=2 --cache-policy=1 \
                #--dnn-name={} --cache-profile={}'.format(vpxdec_file, content_dir, input_video_name, \
                #                                                reference_video_name, model_name, anchor_point_set.path)
                if skip is not None:
                    command += ' --skip={}'.format(skip)
                if limit is not None:
                    command += ' --limit={}'.format(limit)
                if postfix is not None:
                    command += ' --postfix={}'.format(postfix)
                subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                #result = subprocess.check_output(shlex.split(command)).decode('utf-8')
                #result = result.split('\n')

            #load quality from a log file
            quality = []
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    quality.append(float(line.split('\t')[1]))
            end_time = time.time()

            q1.put((idx, quality))

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
