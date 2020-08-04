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

def load_nemo_quality(log_path):
    quality_cache = []
    quality_dnn = []

    with open(log_path, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_cache.append(float(quality_line[3]) - float(quality_line[5]))
            quality_dnn.append(float(quality_line[4]) - float(quality_line[5]))

    return quality_cache, quality_dnn

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

    #setting
    resolution = 240
    quality = 'high'

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'figure18b.txt')

    nemo_qualities = []
    fast_qualities = []
    dnn_qualities = []
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
                fast_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_fast.txt')
                nemo_quality, dnn_quality = load_nemo_quality(nemo_log_path)
                fast_quality, _ = load_nemo_quality(fast_log_path)

                nemo_qualities.append(np.average(nemo_quality))
                fast_qualities.append(np.average(nemo_quality))
                dnn_qualities.append(np.average(nemo_quality))


        nemo_quality.sort()
        fast_quality.sort()
        dnn_quality.sort()

        diff = np.asarray(nemo_quality) - np.asarray(fast_quality)

        count = 0
        f0.write('0\t0\t0\t0\n')
        for nq, fq, pfq in zip(nemo_quality, fast_quality, dnn_quality):
            if count % 10 == 0:
                f0.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(count/len(nemo_quality), nq, fq, pfq))
                f0.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format((count+1)/len(nemo_quality), nq, fq, pfq))
            count += 1
        f0.write('nemo gain: min - {}, max - {}, avg - {}'.format(np.min(diff), np.max(diff), np.average(diff)))
