import time
import os
import sys
import argparse
import math
import json

import numpy as np

from model.common import NormalizeConfig
from model.edsr_s import EDSR_S
from dataset import valid_image_dataset, single_image_dataset, setup_images
from utility import resolve, resolve_bilinear, VideoMetadata, FFmpegOption, upscale_factor, video_fps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--rootdir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)

    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #video metadata
    parser.add_argument('--filter_type', type=str,  default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')
    parser.add_argument('--bitrate', type=int, nargs='+', required=True)

    #architecture
    parser.add_argument('--num_filters', type=int, nargs='+', required=True)
    parser.add_argument('--num_blocks', type=int, nargs='+', required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)
    parser.add_argument('--runtime', type=str, required=True)

    args = parser.parse_args()

    #setting
    ffmpeg_option_0 = FFmpegOption('none', None, None) #for a pretrained DNN
    ffmpeg_option_1 = FFmpegOption(args.filter_type, args.filter_fps, args.upsample) #for a test video
    result_dir = os.path.join(args.rootdir, args.dataset_name, 'result')
    result_path = os.path.join(result_dir, 'summary_s.txt')
    os.makedirs(result_dir, exist_ok=True)

    #scale
    lr_video_path = os.path.join(args.rootdir, args.dataset_name, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.rootdir, args.dataset_name, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    scale = upscale_factor(lr_video_path, hr_video_path)

    with open(result_path, 'w') as f_result:
        log = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('num_blocks', 'num_filters', \
                                    'latency', 'flops', \
                                    'size', 'num_params', \
                                    '\t'.join('Raw(SR)\tRaw(Bilinear)' if bitrate == 0 \
                                    else '{}kbps(SR)\t{}kbps(Bilinear)'.format(bitrate, bitrate) for bitrate in args.bitrate))
        f_result.write(log)

        for num_blocks in args.num_blocks:
            for num_filters in args.num_filters:
                #dnn name
                edsr_s = EDSR_S(num_blocks, num_filters, scale, None)

                #latency
                latency_log_dir = os.path.join(args.rootdir, 'model', edsr_s.name, args.device_id, args.runtime, '1')
                latency_log_path = os.path.join(latency_log_dir, 'summary.json')
                assert(os.path.exists(latency_log_path))

                with open(latency_log_path, encoding='utf-8') as f:
                    data = json.loads(f.read())
                    latency = np.round(data['latency'], 2)
                    flops = data['flops']
                    size = np.round(data['dlc_size'], 2)
                    num_params = data['num_params']

                #quality
                psnr_values = []
                for bitrate in args.bitrate:
                    lr_video_title, lr_video_format = os.path.splitext(args.lr_video_name)
                    if bitrate is not 0:
                        lr_video_name = '{}_{}kbps{}'.format(lr_video_title, bitrate, lr_video_format)
                    else:
                        lr_video_name = args.lr_video_name
                    quality_log_dir = os.path.join(args.rootdir, args.dataset_name, 'log', ffmpeg_option_0.summary(lr_video_name), edsr_s.name, \
                                    ffmpeg_option_1.summary(lr_video_name))
                    quality_log_path = os.path.join(quality_log_dir, 'quality.txt')
                    print(quality_log_path)
                    assert(os.path.exists(quality_log_path))

                    with open(quality_log_path, 'r') as f:
                        line = f.readline()
                        result = line.split('\t')
                        assert(result[0] == 'Average')
                        psnr_values.append(np.round(float(result[1]), 2))
                        psnr_values.append(np.round(float(result[2]), 2))

                #log
                log = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(num_blocks, num_filters, \
                                                        latency, flops, \
                                                        size, num_params, \
                                                        "\t".join(str(psnr_value) for psnr_value in psnr_values))
                f_result.write(log)
