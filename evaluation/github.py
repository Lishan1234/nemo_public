import argparse
import os
import glob
import operator
import json
import sys

import numpy as np

contents = ['product_review', 'how_to', 'vlogs', 'game_play', 'skit', 'haul', 'challenge','favorite', 'education',  'unboxing']
indexes = [1, 2, 3]
resolutions = [240, 360, 480]
device_names = ['xiaomi_redmi_note7', 'xiaomi_mi9', 'lg_gpad5']

num_blocks_info= {
    'low': {
        240: 4,
        360: 4,
        480: 4
    },
    'medium': {
        240: 8,
        360: 4,
        480: 4
    },
    'high': {
        240: 8,
        360: 4,
        480: 4
    },
}

num_filters_info= {
    'low': {
        240: 9,
        360: 8,
        480: 4
    },
    'medium': {
        240: 21,
        360: 18,
        480: 9
    },
    'high': {
        240: 32,
        360: 29,
        480: 18
    },
}

def get_model_quality(num_filters, resolution):
    if resolution == 240:
        if num_filters == 9:
            return 'Low'
        elif num_filters == 21:
            return 'Medium'
        elif num_filters == 32:
            return 'High'
    elif resolution == 360:
        if num_filters == 8:
            return 'Low'
        elif num_filters == 18:
            return 'Medium'
        elif num_filters == 29:
            return 'High'
    elif resolution == 480:
        if num_filters == 4:
            return 'Low'
        elif num_filters == 9:
            return 'Medium'
        elif num_filters == 18:
            return 'High'

video_name_info = {
    240: '240p_512kbps_s0_d300.webm',
    360: '360p_1024kbps_s0_d300.webm',
    480: '480p_1600kbps_s0_d300.webm',
}

algorithm_info = {
    'low': 'nemo_0.5_8',
    'medium': 'nemo_0.5_16',
    'high': 'nemo_0.5_16'
}

def get_model_name(num_blocks, num_filters, resolution):
    if resolution == 240:
        scale = 4
    elif resolution == 360:
        scale = 3
    elif resolution == 240:
        scale = 2

    return 'NEMO_S_B{}_F{}_S{}_deconv'.format(num_blocks, num_filters, scale)

def load_nemo_fps(log_path1, log_path2):
    nemo_fps = []

    idx = 0
    with open(log_path1, 'r') as f1, open(log_path2, 'r') as f2:
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()
        for f2_line in f2_lines:
            f2_line = f2_line.split('\t')
            num_frames = int(f2_line[2])

            nemo_latency = 0
            for f1_line in f1_lines[idx:idx+num_frames]:
                f1_line = f1_line.split('\t')
                nemo_latency += float(f1_line[2])

            nemo_fps.append(num_frames / nemo_latency * 1000)

            idx += num_frames

    return nemo_fps

def load_dnn_fps(log_path):
    total_latency = 0
    num_frames = 0
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            total_latency += float(line[2])
            num_frames += 1

    return (num_frames / total_latency) * 1000

def load_nemo_quality(log_path):
    quality_cache = []
    quality_dnn = []
    quality_bilinear = []

    with open(log_path, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_cache.append(float(quality_line[3]))
            quality_dnn.append(float(quality_line[4]))
            quality_bilinear.append(float(quality_line[5]))

    return quality_cache, quality_dnn, quality_bilinear

def load_dnn_quality(log_path):
    quality_dnn = []

    with open(log_path, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_dnn.append(float(quality_line[1]))

    return quality_dnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file0 = os.path.join(log_dir, 'github.txt')
    with open(log_file0, 'w') as f0:
        for resolution in resolutions:
            for content_name in contents:
                for index in indexes:
                    log = '| {} |'.format(content_name)

                    for device_name in device_names:
                        qualities = []

                        for index in indexes:
                            content = '{}{}'.format(content_name, index)
                            video_dir = os.path.join(args.data_dir, content, 'video')
                            video_name = video_name_info[resolution]
                            log_dir = os.path.join(args.data_dir, content, 'log')

                            json_file = os.path.join(args.data_dir, content, 'log', video_name, 'nemo_device_to_quality.json')
                            with open(json_file, 'r') as f:
                                json_data = json.load(f)
                            num_filters = json_data[device_name]['num_filters']
                            qualities.append(get_model_quality(num_filters, resolution))

                        log += ' {} |'.format(', '.join(qualities))

                f0.write('{}\n'.format(log))






