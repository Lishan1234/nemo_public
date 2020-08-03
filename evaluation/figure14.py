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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file0 = os.path.join(log_dir, 'figure14_gain.txt')
    log_file1 = os.path.join(log_dir, 'figure14_absoulte.txt')
    with open(log_file0, 'w') as f0, open(log_file1, 'w') as f1:
        for content_name in contents:
            #bilinear
            bilinear_quality_per_content = []

            #nemo
            nemo_quality_per_content = []
            nemo_quality_gain_per_content = []
            nemo_num_filters_per_content = []

            #dnn
            dnn_quality_per_content = {}
            dnn_quality_gain_per_content = {}
            model_qualities = ['low', 'medium', 'high']
            for model_quality in model_qualities:
                dnn_quality_per_content[model_quality] = []
                dnn_quality_gain_per_content[model_quality]  = []

            for index in indexes:
                content = '{}{}'.format(content_name, index)
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
                nemo_num_filters_per_content.append(str(num_filters))

                nemo_log_path = os.path.join(log_dir, video_name, model_name, 'quality_{}.txt'.format(algorithm_name))
                nemo_quality, _, bilinear_quality = load_nemo_quality(nemo_log_path)
                nemo_quality_gain = list(map(operator.sub, nemo_quality, bilinear_quality))
                nemo_quality_per_content.extend(nemo_quality)
                nemo_quality_gain_per_content.extend(nemo_quality_gain)
                bilinear_quality_per_content.extend(bilinear_quality)

                for model_quality in model_qualities:
                    model_name = get_model_name(num_blocks_info[model_quality][resolution], num_filters_info[model_quality][resolution], resolution)
                    dnn_log_path = os.path.join(log_dir, video_name, model_name, 'quality.txt')
                    dnn_quality = load_dnn_quality(dnn_log_path)
                    dnn_quality_gain = list(map(operator.sub, dnn_quality, bilinear_quality))
                    dnn_quality_per_content[model_quality].extend(dnn_quality)
                    dnn_quality_gain_per_content[model_quality].extend(dnn_quality_gain)

            avg_nemo_quality_gain_per_content = np.average(nemo_quality_gain_per_content)
            avg_low_quality_gain_per_content = np.average(dnn_quality_gain_per_content['low'])
            avg_medium_quality_gain_per_content = np.average(dnn_quality_gain_per_content['medium'])
            avg_high_quality_gain_per_content = np.average(dnn_quality_gain_per_content['high'])
            std_nemo_quality_gain_per_content = np.std(nemo_quality_gain_per_content)
            std_low_quality_gain_per_content = np.std(dnn_quality_gain_per_content['low'])
            std_medium_quality_gain_per_content = np.std(dnn_quality_gain_per_content['medium'])
            std_high_quality_gain_per_content = np.std(dnn_quality_gain_per_content['high'])
            log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\n'.format(content_name, avg_nemo_quality_gain_per_content,
                                                            avg_low_quality_gain_per_content,
                                                            avg_medium_quality_gain_per_content,
                                                            avg_high_quality_gain_per_content,
                                                            std_nemo_quality_gain_per_content,
                                                            std_low_quality_gain_per_content,
                                                            std_medium_quality_gain_per_content,
                                                            std_high_quality_gain_per_content,
                                                            '\t'.join(nemo_num_filters_per_content))
            f0.write(log)

            avg_nemo_quality_per_content = np.average(nemo_quality_per_content)
            avg_low_quality_per_content = np.average(dnn_quality_per_content['low'])
            avg_medium_quality_per_content = np.average(dnn_quality_per_content['medium'])
            avg_high_quality_per_content = np.average(dnn_quality_per_content['high'])
            std_nemo_quality_per_content = np.std(nemo_quality_per_content)
            std_low_quality_per_content = np.std(dnn_quality_per_content['low'])
            std_medium_quality_per_content = np.std(dnn_quality_per_content['medium'])
            std_high_quality_per_content = np.std(dnn_quality_per_content['high'])
            log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\n'.format(content_name, avg_nemo_quality_per_content,
                                                            avg_low_quality_per_content,
                                                            avg_medium_quality_per_content,
                                                            avg_high_quality_per_content,
                                                            std_nemo_quality_per_content,
                                                            std_low_quality_per_content,
                                                            std_medium_quality_per_content,
                                                            std_high_quality_per_content,
                                                            '\t'.join(nemo_num_filters_per_content))
            f1.write(log)

