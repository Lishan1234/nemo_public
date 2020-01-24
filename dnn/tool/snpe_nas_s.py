import os, time, sys, time
import subprocess
import argparse
import collections
import json

import numpy as np
import tensorflow as tf

from dnn.dataset import setup_images
from dnn.model.nas_s import NAS_S
from dnn.utility import FFmpegOption, resolve_bilinear
from tool.snpe import snpe_convert_model, snpe_convert_dataset, snpe_dlc_viewer, snpe_benchmark
from tool.ffprobe import profile_video
from tool.adb import adb_pull
from tool.tf import valid_raw_dataset

tf.enable_eager_execution()

#TODO: a) meausre latency (with a pretrained model), b) measure quality by sampling 1.0 fps
#TODO: class for dataset
class Profiler():
    device_rootdir = '/data/local/tmp/snpebm'

    def __init__(self, model, log_dir, checkpoint_dir, lr_image_dir, hr_image_dir, image_format='png'):
        assert(os.path.exists(checkpoint_dir))

        self.model = model
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.lr_image_dir = lr_image_dir
        self.hr_image_dir = hr_image_dir
        self.image_format = image_format

        os.makedirs(log_dir, exist_ok=True)

    def run(self, device_id, runtime, perf='default'):
        #prepare model
        dlc_profile = snpe_convert_model(self.model, self.model.nhwc, self.checkpoint_dir)

        #prepare html
        dlc_path =  os.path.join(self.checkpoint_dir, dlc_profile['dlc_name'])
        html_path = os.path.join(self.log_dir, '{}.html'.format(dlc_profile['dlc_name']))
        snpe_dlc_viewer(dlc_path, html_path)

        #prepare raw images
        lr_raw_dir, lr_raw_list = snpe_convert_dataset(self.lr_image_dir, self.image_format)
        hr_raw_dir, _ = snpe_convert_dataset(self.hr_image_dir, self.image_format)

        #prepare json
        json_path = self._create_json(self.model, self.log_dir, device_id, runtime, dlc_path, lr_raw_dir, lr_raw_list)

        #run benchmark
        #TODO: remove result dir
        #snpe_benchmark(json_path)

        #measure latency
        #json_data = open(output_json_path).read()
        #data = json.loads(json_data)
        #return float(data['Execution_Data']['GPU_FP16']['Total Inference Time']['Avg_Time'])

        #output_json_path = os.path.join(result_dir, 'latest_results', 'benchmark_stats_{}.json'.format(self.model.name))
        #assert(os.path.exists(output_json_path))

        #measure quality: a) download output raw images, b) measure PSNR
        device_dir = os.path.join(self.device_rootdir, self.model.name, 'output')
        host_dir = os.path.join(lr_raw_dir, self.model.name, runtime)
        os.makedirs(host_dir, exist_ok=True)
        with open(lr_raw_list, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                device_file = os.path.join(device_dir, 'Result_{}'.format(idx), '{}.raw'.format(dlc_profile['output_name']))
                host_file = os.path.join(host_dir, line.split('/')[1])
                #adb_pull(device_file, host_file, device_id)

        bilinear_psnr_values = []
        valid_raw_ds = valid_raw_dataset(lr_raw_dir, hr_raw_dir, self.model.nhwc[1], self.model.nhwc[2],
                                                        self.model.scale, precision=tf.float32)
        for idx, imgs in enumerate(valid_raw_ds):
            lr = imgs[0][0]
            hr = imgs[1][0]

            bilinear = resolve_bilinear(lr, self.model.nhwc[1] * self.model.scale, self.model.nhwc[2] * self.model.scale)
            bilinear = tf.cast(bilinear, tf.uint8)
            hr = tf.clip_by_value(hr, 0, 255)
            hr = tf.round(hr)
            hr = tf.cast(hr, tf.uint8)

            bilinear_psnr_value = tf.image.psnr(bilinear, hr, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

        sr_psnr_values = []
        valid_raw_ds = valid_raw_dataset(host_dir, hr_raw_dir, self.model.nhwc[1] * self.model.scale,
                                                        self.model.nhwc[2] * self.model.scale,
                                                        1, precision=tf.float32)
        for idx, imgs in enumerate(valid_raw_ds):
            sr = imgs[0][0]
            hr = imgs[1][0]

            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            hr = tf.clip_by_value(hr, 0, 255)
            hr = tf.round(hr)
            hr = tf.cast(hr, tf.uint8)

            sr_psnr_value = tf.image.psnr(sr, hr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

        #measure size, parameters
        #TODO

        #log in as a json file
        #TODO: refer snpe.py


    def measure_size(self):
        #TODO
        pass

    def measure_quality(self):
        #TODO
        pass

    @classmethod
    def _create_json(cls, model, log_dir, device_id, runtime, dlc_path, raw_dir, raw_list, perf='default'):
        result_dir = os.path.join(log_dir, device_id, runtime)
        json_path = os.path.join(result_dir, 'benchmark.json')
        os.makedirs(result_dir, exist_ok=True)

        benchmark = collections.OrderedDict()
        benchmark['Name'] = model.name
        benchmark['HostRootPath'] = os.path.abspath(log_dir)
        benchmark['HostResultsDir'] = os.path.abspath(result_dir)
        benchmark['DevicePath'] = cls.device_rootdir
        benchmark['Devices'] = [device_id]
        benchmark['HostName'] = 'localhost'
        benchmark['Runs'] = 1
        benchmark['Model'] = collections.OrderedDict()
        benchmark['Model']['Name'] = model.name
        benchmark['Model']['Dlc'] = dlc_path
        benchmark['Model']['InputList'] = raw_list
        benchmark['Model']['Data'] = [raw_dir]
        benchmark['Runtimes'] = [runtime]
        benchmark['Measurements'] = ['timing']
        benchmark['ProfilingLevel'] = 'detailed'
        benchmark['BufferTypes'] = ['float']

        with open(json_path, 'w') as outfile:
            json.dump(benchmark, outfile, indent=4)

        return json_path

    def measure_all(self, device_id, runtime):
        #TODO
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')

    #video metadata
    parser.add_argument('--train_filter_type', type=str, default='uniform')
    parser.add_argument('--train_filter_fps', type=float, default=1.0)
    parser.add_argument('--test_filter_type', type=str, default='uniform')
    parser.add_argument('--test_filter_fps', type=float, default=1.0)

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    #device
    parser.add_argument('--device_id', type=str, default=True)
    parser.add_argument('--runtime', type=str, required=True)

    args = parser.parse_args()

    #scale & nhwc
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]

    #model & checkpoint directory
    nas_s = NAS_S(args.num_blocks, args.num_filters, scale)
    model = nas_s.build_model()
    model.scale = scale
    model.nhwc = nhwc
    train_ffmpeg_option = FFmpegOption(args.train_filter_type, args.train_filter_fps, None)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', train_ffmpeg_option.summary(args.lr_video_name), model.name)
    assert(os.path.exists(checkpoint_dir))

    #image
    test_ffmpeg_option = FFmpegOption(args.test_filter_type, args.test_filter_fps, None)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', test_ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, test_ffmpeg_option.filter())

    #profiler
    log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, model.name, 'snpe')
    profiler = Profiler(model, log_dir, checkpoint_dir, lr_image_dir, hr_image_dir)

    #measurement
    profiler.run(args.device_id, args.runtime)
