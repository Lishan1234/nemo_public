import os, time, sys, time
import subprocess
import argparse
import collections
import json

import numpy as np

from dnn.model.nas_s import NAS_S
from dnn.utility import FFmpegOption
from tool.snpe import snpe_convert_model, snpe_dlc_viewer, snpe_benchmark
from tool.ffprobe import profile_video

#TODO: a) meausre latency (with a pretrained model), b) measure quality by sampling 1.0 fps
#TODO: class for dataset
class Profiler():
    def __init__(self, model, nhwc, log_dir, checkpoint_dir):
        assert(os.path.exists(checkpoint_dir))

        self.model = model
        self.nhwc = nhwc
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.dlc_profile = None

        os.makedirs(log_dir, exist_ok=True)

    def setup(self):
        #model
        self.dlc_profile = snpe_convert_model(self.model, self.nhwc, self.checkpoint_dir)

        #dlc info. html
        dlc_path =  os.path.join(self.checkpoint_dir, self.dlc_profile['dlc_name'])
        html_path = os.path.join(self.log_dir, '{}.html'.format(self.dlc_profile['dlc_name']))
        snpe_dlc_viewer(dlc_path, html_path)

        #library
        #raw images
        #script

    def measure_latency(self, device_id, runtime, total_run, total_num, perf='default'):
        json_path = self._create_json(device_id, runtime, total_run, total_num)
        snpe_benchmark(json_path)

        output_json_path = os.path.join(result_dir, 'latest_results', 'benchmark_stats_{}.json'.format(self.model.name))
        assert(os.path.exists(output_json_path))

        json_data = open(output_json_path).read()
        data = json.loads(json_data)
        return float(data['Execution_Data']['GPU_FP16']['Total Inference Time']['Avg_Time'])

    def measure_size(self):
        #TODO
        pass

    def measure_quality(self):
        #TODO
        pass

    def _create_json(self, device_id, runtime, total_run, total_num, perf='default'):
        result_dir = os.path.join(self.log_dir, device_id, runtime)
        os.makedirs(result_dir, exist_ok=True)
        json_path = os.path.join(result_dir, 'benchmark.json')

        benchmark = collections.OrderedDict()
        benchmark['Name'] = self.model.name
        benchmark['HostRootPath'] = os.path.abspath(self.log_dir)
        benchmark['HostResultsDir'] = os.path.abspath(result_dir)
        benchmark['DevicePath'] = '/data/local/tmp/snpebm'
        benchmark['Devices'] = [device_id]
        benchmark['HostName'] = 'localhost'
        benchmark['Runs'] = total_run
        benchmark['Model'] = collections.OrderedDict()
        benchmark['Model']['Name'] = self.model.name
        benchmark['Model']['Dlc'] = os.path.join(self.dlc_profile['dlc_path'])
        benchmark['Model']['RandomInput'] = total_num
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

    #test
    parser.add_argument('--total_run', type=int, default=5)
    parser.add_argument('--total_num', type=int, default=5)

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
    train_ffmpeg_option = FFmpegOption(args.train_filter_type, args.train_filter_fps, None)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', train_ffmpeg_option.summary(args.lr_video_name), model.name)
    assert(os.path.exists(checkpoint_dir))

    #profiler
    log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, model.name, 'snpe')
    profiler = Profiler(model, nhwc, log_dir, checkpoint_dir)
    profiler.setup()

    #measurement
    profiler.measure_latency(args.device_id, args.runtime, args.total_run, args.total_num)
