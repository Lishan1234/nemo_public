import argparse
import os
import glob
import operator

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

def load_num_anchor_points(log_path):
    num_anchor_points = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip().split('\t')
            num_anchor_points.append(float(line[1]))

    return num_anchor_points

def load_num_anchor_points(log_path):
    num_anchor_points = []

    with open(log_path, 'r') as f:
        quality_lines = f.readlines()
        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            num_anchor_points.append(int(quality_line[1]))

    return num_anchor_points

def load_chunk_quality(log_path, chunk_idx):
    with open(log_path, 'r') as f:
        quality_lines = f.readlines()

        quality_line = quality_lines[chunk_idx].strip().split('\t')
        return float(quality_line[3]) - float(quality_line[2])

def load_frame_quality(log_path):
    quality = []

    with open(log_path, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality.append(float(quality_line[1]))

    return quality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    #setting
    quality = 'high'
    resolution = 240

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file_0 = os.path.join(log_dir, 'figure20_a.txt')
    log_file_1 = os.path.join(log_dir, 'figure20_b.txt')

    with open(log_file_0, 'w') as f_0, open(log_file_1, 'w') as f_1:
        #video, directory
        content = 'education1'
        video_dir = os.path.join(args.data_dir, content, 'video')
        video_name = video_name_info[resolution]
        log_dir = os.path.join(args.data_dir, content, 'log')

        num_blocks = num_blocks_info[quality][resolution]
        num_filters = num_filters_info[quality][resolution]
        model_name = get_model_name(num_blocks, num_filters, resolution)

        nemo_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_nemo_0.5.txt')
        num_anchor_points = load_num_anchor_points(nemo_log_path)

        #chunk quality
        for i in range(75):
            postfix = 'chunk{:04d}'.format(i)
            nemo_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, postfix, 'quality_nemo_0.5.txt')
            nemo_chunk_quality = load_chunk_quality(nemo_log_path, -1)
            uniform_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, postfix, 'quality_uniform_0.5.txt')
            uniform_chunk_quality = load_chunk_quality(uniform_log_path, num_anchor_points[i] - 1)
            random_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, postfix, 'quality_random_0.5.txt')
            random_chunk_quality = load_chunk_quality(random_log_path, num_anchor_points[i] - 1)

            f_0.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(i, nemo_chunk_quality, uniform_chunk_quality, random_chunk_quality))

        #frame quality
        postfix = 'chunk0013'
        nemo_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, postfix, 'nemo_0.5_8', 'quality.txt')
        nemo_frame_quality = load_frame_quality(nemo_log_path)

        uniform_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, postfix, 'uniform_0.5_8', 'quality.txt')
        uniform_frame_quality = load_frame_quality(uniform_log_path)

        random_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, postfix, 'random_0.5_8', 'quality.txt')
        random_frame_quality = load_frame_quality(random_log_path)

        dnn_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, postfix, 'quality.txt')
        dnn_frame_quality = load_frame_quality(dnn_log_path)

        count = 0
        for nq, uq, rq, dq in zip(nemo_frame_quality, uniform_frame_quality, random_frame_quality, dnn_frame_quality):
            f_1.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(count, dq - nq, dq - uq, dq - rq))
            count += 1
