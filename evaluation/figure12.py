import argparse
import os
import glob
import operator
import json

import numpy as np

contents = ['product_review', 'how_to', 'vlogs', 'game_play', 'skit', 'haul', 'challenge','favorite', 'education',  'unboxing']
indexes = [1, 2, 3]
resolution = 240
quality = 'high'
device_name = 'xiaomi_mi9'

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

def load_anchor_point_fraction(log_path):
        anchor_point_fraction = []

        with open(log_path, 'r') as f:
            quality_lines = f.readlines()
            for quality_line in quality_lines:
                quality_line = quality_line.strip().split('\t')
                num_anchor_points = int(quality_line[1])
                num_frames = int(quality_line[2])
                anchor_point_fraction.append(num_anchor_points / num_frames * 100)

        return anchor_point_fraction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'figure12.txt')

    anchor_point_fraction = []
    with open(log_file, 'w') as f0:
        for content_name in contents:
            for index in indexes:
                content = '{}{}'.format(content_name, index)
                video_name = video_name_info[resolution]
                json_file = os.path.join(args.data_dir, content, 'log', video_name, 'nemo_device_to_quality.json')
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                num_blocks = json_data[device_name]['num_blocks']
                num_filters = json_data[device_name]['num_filters']
                algorithm_name = json_data[device_name]['algorithm_type']
                model_name = get_model_name(num_blocks, num_filters, resolution)
                nemo_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_{}.txt'.format(algorithm_name))
                anchor_point_fraction.append(np.average(load_anchor_point_fraction(nemo_log_path)))

        anchor_point_fraction.sort()
        count = 0
        f0.write('0\t0\n')
        for value in anchor_point_fraction :
            f0.write('{:.2f}\t{:.2f}\n'.format(count/len(anchor_point_fraction ), value))
            f0.write('{:.2f}\t{:.2f}\n'.format((count+1)/len(anchor_point_fraction ),value))
            count += 1
