import argparse
import os
import glob
import operator
import json
import sys

import numpy as np

from nemo.tool.video import profile_video
from nemo.tool.mobile import playback_time

contents = ['product_review', 'how_to', 'vlogs', 'game_play', 'skit', 'haul', 'challenge','favorite', 'education',  'unboxing']
indexes = [1, 2, 3]
resolution = 240
quality = 'high'
device_names = ['xiaomi_mi9']

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

def load_num_frames(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()

        return len(lines)

def load_monsoon_results(log_path):
        time = []
        current = []
        power = []

        with open(log_path, 'r') as f:
            lines = f.readlines()

            for idx, line in enumerate(lines):
                if idx == 0 :
                   continue
                else:
                    results = line.strip().split(',')
                    time.append(float(results[0]))
                    current.append(float(results[1]))
                    power.append(float(results[2]))

        return time[-1] - time[0], current, power

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

def get_model_name(num_blocks, num_filters, resolution):
    if resolution == 240:
        scale = 4
    elif resolution == 360:
        scale = 3
    elif resolution == 240:
        scale = 2

    return 'NEMO_S_B{}_F{}_S{}_deconv'.format(num_blocks, num_filters, scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    content = 'education1'

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_path0= os.path.join(log_dir, 'figure15_a.txt')
    log_path1= os.path.join(log_dir, 'figure15_b.txt')
    log_path2= os.path.join(log_dir, 'figure15_c.txt')

    with open(log_path0, 'w') as f0, open(log_path1, 'w') as f1, open(log_path2, 'w') as f2:
        video_name = video_name_info[resolution]
        video_path = os.path.join(args.data_dir, content, 'video', video_name)
        video_profile = profile_video(video_path)
        fps = video_profile['frame_rate']

        for device_name in device_names:
            #model name, algorithm name
            json_file = os.path.join(args.data_dir, content, 'log', video_name, 'nemo_device_to_quality.json')
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            num_blocks = json_data[device_name]['num_blocks']
            num_filters = json_data[device_name]['num_filters']
            algorithm_name = json_data[device_name]['algorithm_type']
            model_name = get_model_name(num_blocks, num_filters, resolution)

            #no dnn
            bilinear_metadata_log_path = os.path.join(args.data_dir, content, 'log', video_name, device_name, 'power', 'metadata.txt')
            bilinear_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, device_name, 'power', 'monsoon.csv')
            time, current, power = load_monsoon_results(bilinear_monsoon_log_path)
            total_frame = load_num_frames(bilinear_metadata_log_path)
            bilinear_avg_power = np.average(power)
            bilinear_total_energy = bilinear_avg_power * time * 60 #caution: translate 1 minute to 60 seconds
            bilinear_avg_energy = bilinear_total_energy / total_frame / 1000
            bilinear_fps = total_frame / time
            bilinear_battery_life = playback_time(np.average(current), device_name) / (fps / bilinear_fps) / 60

            #per-frame dnn
            dnn_metadata_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, device_name, 'power', 'metadata.txt')
            dnn_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, device_name, 'power', 'monsoon.csv')
            time, current, power = load_monsoon_results(dnn_monsoon_log_path)
            total_frame = load_num_frames(dnn_metadata_log_path)
            dnn_avg_power = np.average(power)
            dnn_total_energy = dnn_avg_power * time * 60 / 1000
            dnn_avg_energy = dnn_total_energy / total_frame
            dnn_fps = total_frame / time
            dnn_battery_life = playback_time(np.average(current), device_name) / (fps / dnn_fps) / 60

            #nemo
            nemo_metadata_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, algorithm_name, device_name, 'power', 'metadata.txt')
            nemo_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, algorithm_name, device_name, 'power', 'monsoon.csv')
            time, current, power = load_monsoon_results(nemo_monsoon_log_path)
            total_frame = load_num_frames(nemo_metadata_log_path)
            nemo_avg_power = np.average(power)
            nemo_total_energy = nemo_avg_power * time * 60 / 1000
            nemo_avg_energy = nemo_total_energy / total_frame
            nemo_fps = total_frame / time
            nemo_battery_life = playback_time(np.average(current), device_name) / (fps / nemo_fps) / 60

            f0.write('{}\t{}\t{}\t{}\n'.format(device_name, bilinear_avg_energy, nemo_avg_energy, dnn_avg_energy))
            f1.write('{}\t{}\t{}\t{}\n'.format(device_name, bilinear_battery_life, nemo_battery_life, dnn_battery_life))

        #trade-off
        nemo_quality_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_nemo_0.5_16.txt')
        nemo_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'nemo_0.5_16', device_name, 'power', 'monsoon.csv')
        time, current, power = load_monsoon_results(nemo_monsoon_log_path)
        nemo_avg_power = np.average(power)/ 1000
        nemo_quality, _, _ = load_nemo_quality(nemo_quality_log_path)

        f2.write('0.5\t{}\t{}\n'.format(nemo_avg_power, np.average(nemo_quality)))

        nemo_quality_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_nemo_0.75_16.txt')
        nemo_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'nemo_0.75_16', device_name, 'power', 'monsoon.csv')
        time, current, power = load_monsoon_results(nemo_monsoon_log_path)
        nemo_avg_power = np.average(power)/ 1000
        nemo_quality, _, _ = load_nemo_quality(nemo_quality_log_path)

        f2.write('0.75\t{}\t{}\n'.format(nemo_avg_power, np.average(nemo_quality)))

        nemo_quality_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_nemo_1.0_16.txt')
        nemo_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'nemo_1.0_16', device_name, 'power', 'monsoon.csv')

        time, current, power = load_monsoon_results(nemo_monsoon_log_path)
        nemo_avg_power = np.average(power)/ 1000
        nemo_quality, _, _ = load_nemo_quality(nemo_quality_log_path)

        f2.write('1.0\t{}\t{}\n'.format(nemo_avg_power, np.average(nemo_quality)))

        nemo_quality_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_nemo_1.5_16.txt')
        nemo_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'nemo_1.5_16', device_name, 'power', 'monsoon.csv')
        time, current, power = load_monsoon_results(nemo_monsoon_log_path)
        nemo_avg_power = np.average(power)/ 1000
        nemo_quality, _, _ = load_nemo_quality(nemo_quality_log_path)

        f2.write('1.5\t{}\t{}\n'.format(nemo_avg_power, np.average(nemo_quality)))

        nemo_quality_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'quality_nemo_2.0_16.txt')
        nemo_monsoon_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, 'nemo_2.0_16', device_name, 'power', 'monsoon.csv')
        time, current, power = load_monsoon_results(nemo_monsoon_log_path)
        nemo_avg_power = np.average(power)/ 1000
        nemo_quality, _, _ = load_nemo_quality(nemo_quality_log_path)

        f2.write('2.0\t{}\t{}\n'.format(nemo_avg_power, np.average(nemo_quality)))
