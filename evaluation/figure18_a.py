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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    #setting
    resolution = 240
    quality = 'high'

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'figure18_a.txt')

    nemo_num_anchor_points= []
    random_num_anchor_points= []
    uniform_num_anchor_points= []
    with open(log_path, 'w') as f0:
        for content_name in contents:
            for index in indexes:
                #video, directory
                content = '{}{}'.format(content_name, index)
                video_dir = os.path.join(args.data_dir, content, 'video')
                video_name = video_name_info[resolution]
                log_dir = os.path.join(args.data_dir, content, 'log')

                num_blocks = num_blocks_info[quality][resolution]
                num_filters = num_filters_info[quality][resolution]
                model_name = get_model_name(num_blocks, num_filters, resolution)

                nemo_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_nemo_0.5.txt')
                nemo_num_anchor_points.extend(load_num_anchor_points(nemo_log_path))
                uniform_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_uniform_0.5.txt')
                uniform_num_anchor_points.extend(load_num_anchor_points(uniform_log_path))
                random_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_random_0.5.txt')
                random_num_anchor_points.extend(load_num_anchor_points(random_log_path))


        nemo_num_anchor_points.sort()
        random_num_anchor_points.sort()
        uniform_num_anchor_points.sort()

        count = 0
        f0.write('0\t0\t0\t0\n')
        for na, ua, ra in zip(nemo_num_anchor_points, uniform_num_anchor_points, random_num_anchor_points):
            f0.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(count/len(nemo_num_anchor_points), na, ua, ra))
            f0.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format((count+1)/len(nemo_num_anchor_points), na, ua, ra))
            count += 1

        avg_nemo_num_anchor_points = np.average(nemo_num_anchor_points)
        avg_random_num_anchor_points = np.average(random_num_anchor_points)
        avg_uniform_num_anchor_points = np.average(uniform_num_anchor_points)
        gain_over_random = (avg_random_num_anchor_points - avg_nemo_num_anchor_points) / avg_random_num_anchor_points * 100
        gain_over_uniform = (avg_uniform_num_anchor_points - avg_nemo_num_anchor_points) / avg_uniform_num_anchor_points * 100
        f0.write('gain over random - {}, gain over uniform - {}'.format(gain_over_random, gain_over_uniform))
