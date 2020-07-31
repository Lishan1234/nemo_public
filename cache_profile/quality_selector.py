import argparse
import os
import math
import glob
import numpy as np
import json

from nemo.tool.video import profile_video
from nemo.tool.mobile import id_to_name
import nemo.dnn.model

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play_1': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

qualities = ['low', 'medium', 'high']

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

algorithm_info = {
    'low': 'nemo_0.5_8',
    'medium': 'nemo_0.5_24',
    'high': 'nemo_0.5_24',
}

class QualitySelector():
    def __init__(self, data_dir, sample_content, content, video_name, gop, models, device_names):
        self.data_dir = data_dir
        self.sample_content = sample_content
        self.content = content
        self.video_name = video_name
        self.gop = gop
        self.models = models
        self.device_names = device_names
        self.latency_dict = {}

    def load_sample_video_latency(self):
        for device_name in self.device_names:
            self.latency_dict[device_name] = {}

            #load non-anchor frames' latencies
            latency_log = os.path.join(self.data_dir, self.sample_content, 'log', self.video_name, self.models['low'].name, algorithm_info['low'], device_name, 'latency.txt')
            metadata_log = os.path.join(self.data_dir, self.sample_content, 'log', self.video_name, self.models['low'].name, algorithm_info['low'], device_name, 'metadata.txt')
            with open(latency_log, 'r') as f0, open(metadata_log, 'r') as f1:
                latency_lines = f0.readlines()
                metadata_lines = f1.readlines()
                non_anchor_frame = []

                for latency_line, metadata_line in zip(latency_lines, metadata_lines):
                    latency_result = latency_line.strip().split('\t')
                    metadata_result = metadata_line.strip().split('\t')

                    if len(latency_result) >= 3 and len(metadata_result) >= 3:
                        if int(metadata_result[2]) == 0:
                            non_anchor_frame.append(float(latency_result[2]))

                self.latency_dict[device_name]['non_anchor_frame'] = np.average(non_anchor_frame)

            #load anchor points' latencies
            for quality in qualities:
                latency_log = os.path.join(self.data_dir, self.sample_content, 'log', self.video_name, self.models[quality].name, device_name, 'latency.txt')
                metadata_log = os.path.join(self.data_dir, self.sample_content, 'log', self.video_name, self.models[quality].name, device_name, 'metadata.txt')

                with open(latency_log, 'r') as f0, open(metadata_log, 'r') as f1:
                    latency_lines = f0.readlines()
                    metadata_lines = f1.readlines()
                    anchor_point = []

                    for latency_line, metadata_line in zip(latency_lines, metadata_lines):
                        latency_result = latency_line.strip().split('\t')
                        metadata_result = metadata_line.strip().split('\t')

                        #anchor point
                        if len(latency_result) >= 3 and len(metadata_result) >= 3:
                            if int(metadata_result[2]) == 1:
                                anchor_point.append(float(latency_result[2]))

                    self.latency_dict[device_name][self.models[quality].name] = {}
                    self.latency_dict[device_name][self.models[quality].name]['anchor_point'] = np.average(anchor_point)

    def select_quality(self):
        device_to_quality = {}
        video_path = os.path.join(self.data_dir, self.content, 'video', self.video_name)
        fps = profile_video(video_path)['frame_rate']
        height = profile_video(video_path)['height']

        for device_name in self.device_names:
            device_to_quality[device_name] = {}
            selected_quality = None

            for quality in qualities:
                is_real_time = True
                log_path = os.path.join(self.data_dir, self.content, 'log', self.video_name, self.models[quality].name, 'quality_{}.txt'.format(algorithm_info[quality]))
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split('\t')
                        num_anchor_points = int(line[1])
                        num_frames = int(line[2])
                        anchor_point_latency = num_anchor_points * self.latency_dict[device_name][self.models[quality].name]['anchor_point']
                        non_anchor_frame_latency = (num_frames - num_anchor_points) * self.latency_dict[device_name]['non_anchor_frame']
                        total_latency = anchor_point_latency + non_anchor_frame_latency
                        if total_latency > (self.gop / fps) * 1000:
                            is_real_time = False
                            break
                    if is_real_time is True:
                        selected_quality = quality

            if selected_quality is not None:
                device_to_quality[device_name] = {}
                device_to_quality[device_name]['num_blocks'] = num_blocks_info[selected_quality][height]
                device_to_quality[device_name]['num_filters'] = num_filters_info[selected_quality][height]
                device_to_quality[device_name]['algorithm_type'] = algorithm_info[selected_quality]
            else:
                print('Cannot process at real time')
                device_to_quality[device_name]['num_blocks'] = None
                device_to_quality[device_name]['num_filters'] = None
                device_to_quality[device_name]['algorithm_type'] = None

        json_path = os.path.join(self.data_dir, self.content, 'log', self.video_name, 'nemo_device_to_quality.json')
        with open(json_path, 'w') as f:
            json.dump(device_to_quality, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--video_name', type=str, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)

    #dnn
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--upsample_type', type=str, default='deconv')

    #quality selector
    parser.add_argument('--sample_content', type=str, default='product_review1')
    parser.add_argument('--gop', type=int, default=120)

    #device
    parser.add_argument('--device_names', type=str, nargs='+', required=True)

    args = parser.parse_args()

    #profile videos
    data_dir = os.path.join(args.data_dir, args.content)
    lr_video_path = os.path.join(data_dir, 'video', args.video_name)
    lr_video_profile = profile_video(lr_video_path)
    scale = args.output_height // lr_video_profile['height']

    #setup dnns
    models = {}
    for quality in qualities:
        num_blocks = num_blocks_info[quality][lr_video_profile['height']]
        num_filters = num_filters_info[quality][lr_video_profile['height']]
        model = nemo.dnn.model.build(args.model_type, num_blocks, num_filters, scale, args.upsample_type)
        models[quality] = model

    qs = QualitySelector(args.data_dir, args.sample_content, args.content, args.video_name, args.gop, models, args.device_names)
    qs.load_sample_video_latency()
    qs.select_quality()
