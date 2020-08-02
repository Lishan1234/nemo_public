import argparse
import os
import glob
import operator
import sys
import json

import numpy as np

from nemo.tool.video import profile_video

contents = ['product_review', 'how_to', 'vlogs', 'game_play', 'skit', 'haul', 'challenge','favorite', 'education',  'unboxing']
indexes = [1, 2, 3]
resolution = 240
quality = 'high'
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

def load_flir_results(log_path):
        time = []
        current = []
        temperature = []

        with open(log_path, 'r') as f:
            lines = f.readlines()

            for idx, line in enumerate(lines):
                temperature_points = []
                if idx == 0 :
                   continue
                else:
                    results = line.strip().split(',')
                    time.append(float(results[0]))

                    temperature_points.append(float(results[1]))
                    temperature_points.append(float(results[2]))
                    temperature_points.append(float(results[3]))
                    temperature_points.append(float(results[4]))
                    temperature.append(max(temperature_points))

        return time, temperature

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
    INTERVAL = 1000

    #log
    log_dir = os.path.join(args.data_dir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_path0 = os.path.join(log_dir, 'figure16_b.txt')
    log_path1 = os.path.join(log_dir, 'figure16_c.txt')
    log_path2 = os.path.join(log_dir, 'figure16_d.txt')

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

            max_len = 0

            #no dnn
            bilinear_flir_log_path = os.path.join(args.data_dir, content, 'log', video_name, device_name, 'temperature', 'temperature.csv')
            bilinear_metadata_log_path = os.path.join(args.data_dir, content, 'log', video_name, device_name, 'temperature', 'metadata.txt')
            total_frame = load_num_frames(bilinear_metadata_log_path)
            time, temperature = load_flir_results(bilinear_flir_log_path)
            bilinear_fps = total_frame / (time[-1] - time[0]) * 1000
            time = [x * (bilinear_fps / fps) for x in time]
            bilinear_time = time
            max_len = len(bilinear_time)
            bilinear_temperature = []
            for i in range(total_frame):
                bilinear_temperature.append(temperature[int(i / total_frame * len(temperature))])
            if max_len < total_frame:
                max_len = total_frame

            #nemo
            nemo_flir_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, algorithm_name, device_name, 'temperature', 'temperature.csv')
            nemo_metadata_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, algorithm_name, device_name, 'temperature', 'metadata.txt')
            total_frame = load_num_frames(nemo_metadata_log_path)
            time, temperature = load_flir_results(nemo_flir_log_path)
            nemo_fps = total_frame / (time[-1] - time[0]) * 1000
            time = [x * (nemo_fps / fps) for x in time]
            nemo_times = time
            nemo_temperature = []
            for i in range(total_frame):
                nemo_temperature.append(temperature[int(i / total_frame * len(temperature))])
            if max_len < total_frame:
                max_len = total_frame

            #dnn
            dnn_flir_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, device_name, 'temperature', 'temperature.csv')
            dnn_metadata_log_path = os.path.join(args.data_dir, content, 'log', video_name, model_name, device_name, 'temperature', 'metadata.txt')
            total_frame = load_num_frames(dnn_metadata_log_path)
            time, temperature = load_flir_results(dnn_flir_log_path)
            total_frame = load_num_frames(dnn_metadata_log_path)
            dnn_fps = total_frame / (time[-1] - time[0]) * 1000
            time = [x * (dnn_fps / fps) for x in time]
            dnn_times = time
            dnn_temperature = []
            for i in range(total_frame):
                dnn_temperature.append(temperature[int(i / total_frame * len(temperature))])
            if max_len < total_frame:
                max_len = total_frame

            num_samples = max_len // 100

            if device_name == 'xiaomi_redmi_note7':
                f_target = f0
            elif device_name == 'xiaomi_mi9':
                f_target = f1
            elif device_name == 'lg_gpad5':
                f_target = f2

            for i in range(num_samples):
                f_target.write('{}'.format(i * 100))
                if i * 100 < len(bilinear_temperature):
                    f_target.write('\t{:.2f}'.format(bilinear_temperature[i * 100]))
                else:
                    f_target.write('\t\t')
                if i * 100 < len(nemo_temperature):
                    f_target.write('\t{:.2f}'.format(nemo_temperature[i * 100]))
                else:
                    f_target.write('\t')
                if i * 100 < len(dnn_temperature):
                    f_target.write('\t{:.2f}'.format(dnn_temperature[i * 100]))
                else:
                    f_target.write('\t')
                f_target.write('\n')
