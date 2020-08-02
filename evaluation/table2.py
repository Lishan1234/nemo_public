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
qualities = ['low', 'medium', 'high']
device_names = ['samsung_a70', 'samsung_note8', 'samsung_s6', 'samsung_s10+']

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

def load_latency(log_path):
    anchor_point = []
    non_anchor_frame = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):
            line = line.split('\t')
            if len(line) != 3:
                anchor_point.append(float(line[2]))
            else:
                non_anchor_frame.append(float(line[2]))

    return np.average(anchor_point), np.average(non_anchor_frame)

def load_latency_1(anchor_point_latency, log_path1, log_path2):
    nemo_fps = []
    improvement = []

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
            improvement.append(anchor_point_latency * num_frames / nemo_latency)

            idx += num_frames

    return np.average(nemo_fps), np.average(improvement)

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
    log_path0 = os.path.join(log_dir, 'table2.txt')

    with open(log_path0, 'w') as f0:
        video_name = video_name_info[resolution]
        video_path = os.path.join(args.data_dir, content, 'video', video_name)
        video_profile = profile_video(video_path)
        fps = video_profile['frame_rate']

        for device_name in device_names:
            selected_quality = None
            for quality in qualities:
                num_blocks = num_blocks_info[quality][resolution]
                num_filters = num_filters_info[quality][resolution]
                latency_log_path = os.path.join(args.data_dir, content, 'log', video_name, get_model_name(num_blocks, num_filters, resolution), algorithm_info[quality], device_name, 'latency.txt')
                anchor_point_latency, non_anchor_frame_latency = load_latency(latency_log_path)

                is_real_time = True
                log_path = os.path.join(args.data_dir, content, 'log', video_name, get_model_name(num_blocks, num_filters, resolution), 'quality_{}.txt'.format(algorithm_info[quality]))
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split('\t')
                        num_anchor_points = int(line[1])
                        num_frames = int(line[2])
                        total_anchor_point_latency = num_anchor_points * anchor_point_latency
                        total_non_anchor_frame_latency = (num_frames - num_anchor_points) * non_anchor_frame_latency
                        total_latency = total_anchor_point_latency + total_non_anchor_frame_latency
                        if total_latency > (120 / fps) * 1000:
                            is_real_time = False
                            break
                    if is_real_time is True:
                        selected_quality = quality

            num_blocks = num_blocks_info[selected_quality][resolution]
            num_filters = num_filters_info[selected_quality][resolution]
            latency_log_path = os.path.join(args.data_dir, content, 'log', video_name, get_model_name(num_blocks, num_filters, resolution), algorithm_info[selected_quality], device_name, 'latency.txt')
            anchor_point_latency, non_anchor_frame_latency = load_latency(latency_log_path)
            log_path = os.path.join(args.data_dir, content, 'log', video_name, get_model_name(num_blocks, num_filters, resolution), 'quality_{}.txt'.format(algorithm_info[quality]))
            nemo_fps, improvement = load_latency_1(anchor_point_latency, latency_log_path, log_path)
            print(device_name, selected_quality, nemo_fps, improvement)

            f0.write('{}\t{}\t{}\t{}\n'.format(device_name, selected_quality, nemo_fps, improvement))
