import argparse
import os
import glob
import operator
import json

import numpy as np

contents = ['product_review', 'how_to', 'vlogs', 'game_play', 'skit', 'haul', 'challenge','favorite', 'education',  'unboxing']
indexes = [1, 2, 3]
quality = 'high'
device_names = ['xiaomi_mi9', 'xiaomi_redmi_note7', 'lg_gpad5']

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
    720: '720p_2640kbps_s0_d300.webm',
    1080: '1080p_4400kbps_s0_d300.webm',
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
    elif resolution == 480:
        scale = 2

    return 'NEMO_S_B{}_F{}_S{}_deconv'.format(num_blocks, num_filters, scale)

#note: KB
def get_model_size(model_name):
    if model_name == 'NEMO_S_B4_F9_S4_deconv':
        return 49
    elif model_name == 'NEMO_S_B8_F21_S4_deconv':
        return 304
    elif model_name == 'NEMO_S_B8_F32_S4_deconv':
        return 658
    elif model_name == 'NEMO_S_B4_F8_S3_deconv':
        return 43
    elif model_name == 'NEMO_S_B4_F18_S3_deconv':
        return 129
    elif model_name == 'NEMO_S_B4_F29_S3_deconv':
        return 298
    elif model_name == 'NEMO_S_B4_F4_S2_deconv':
        return 26
    elif model_name == 'NEMO_S_B4_F9_S2_deconv':
        return 49
    elif model_name == 'NEMO_S_B4_F18_S2_deconv':
        return 129
    else:
        raise ValueError

def load_nemo_quality(log_path):
    quality_nemo = []

    with open(log_path, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_nemo.append(float(quality_line[3]))

    return quality_nemo

def load_bilinear_quality(log_path):
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

    #hard-coded configuration
    hr_resolution = 1080
    bilinear_resolution = [240, 360, 480, 720, 1080]
    cache_resolution = [240, 360, 480]
    num_blocks = {}
    num_filters = {}
    num_blocks[240] = 8
    num_blocks[360] = 8
    num_blocks[480] = 8
    num_filters[240] = 32
    num_filters[360] = 29
    num_filters[480] = 18


    for device_name in device_names:
        #log
        for content_name in contents:
            #bilienar
            bilinear_quality = {}
            nemo_quality = {}

            for index in indexes:
                content = '{}{}'.format(content_name, index)

                for resolution in bilinear_resolution:
                    video_path = os.path.join(args.data_dir, content, 'video', video_name_info[resolution])
                    video_name = video_name_info[resolution]
                    bilinear_log_path = os.path.join(args.data_dir, content, 'log', video_name, 'quality.txt')
                    bilinear_quality[resolution] = load_bilinear_quality(bilinear_log_path)

                model_size = 0
                #nemo
                for resolution in cache_resolution:
                    video_dir = os.path.join(args.data_dir, content, 'video')
                    video_name = video_name_info[resolution]
                    log_dir = os.path.join(args.data_dir, content, 'log')

                    json_file = os.path.join(args.data_dir, content, 'log', video_name, 'nemo_device_to_quality.json')
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    num_blocks = json_data[device_name]['num_blocks']
                    num_filters = json_data[device_name]['num_filters']
                    algorithm_name = json_data[device_name]['algorithm_type']
                    model_name = get_model_name(num_blocks, num_filters, resolution)

                    nemo_log_path = os.path.join(args.data_dir, content, 'log', video_name_info[resolution], get_model_name(num_blocks, num_filters, resolution), 'quality_{}.txt'.format(algorithm_name))
                    nemo_quality[resolution] = load_nemo_quality(nemo_log_path)
                    model_size += get_model_size(model_name)

                log_dir = os.path.join(args.data_dir, 'evaluation', 'figure17', device_name)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, '{}.txt'.format(content))

                with open(log_path, 'w') as f:
                    for i in range(75):
                        f.write('{}'.format(i))
                        f.write('\t{}'.format(nemo_quality[240][i]))
                        f.write('\t{}'.format(nemo_quality[360][i]))
                        f.write('\t{}'.format(nemo_quality[480][i]))
                        f.write('\t{}'.format(bilinear_quality[240][i]))
                        f.write('\t{}'.format(bilinear_quality[360][i]))
                        f.write('\t{}'.format(bilinear_quality[480][i]))
                        f.write('\t{}'.format(bilinear_quality[720][i]))
                        f.write('\t{}'.format(bilinear_quality[1080][i]))
                        f.write('\t{}\n'.format(model_size))
