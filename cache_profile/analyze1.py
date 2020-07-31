import argparse
import os
import math
import glob
import numpy as np
import json

import nemo
from nemo.tool.video import profile_video
from nemo.tool.mobile import id_to_name
import nemo.dnn.model

contents = ['product_review', 'how_to', 'vlogs', 'skit', 'game_play', 'haul', 'challenge', 'education', 'favorite', 'unboxing']
indexes = [1, 2, 3]
resolution = 240
quality = "high"

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)

    #dnn
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--upsample_type', type=str, default='deconv')

    args = parser.parse_args()

    #setup dnns
    quality_gain_per_content = {}

    for content in contents:
        for index in indexes:
            video_name = video_name_info[resolution]
            content_name = '{}{}'.format(content, index)

            video_path = os.path.join(args.data_dir, content_name, 'video', video_name)
            video_profile = profile_video(video_path)
            scale = args.output_height // video_profile['height']
            num_blocks = num_blocks_info[quality][video_profile['height']]
            num_filters = num_filters_info[quality][video_profile['height']]
            model_name = nemo.dnn.model.build(args.model_type, num_blocks, num_filters, scale, args.upsample_type).name

            nemo_05_24_log_path = os.path.join(args.data_dir, content_name, 'log', video_name, model_name, 'quality_nemo_0.5_24.txt')
            assert(os.path.exists(nemo_05_24_log_path))

            with open(nemo_05_24_log_path, 'r') as f:
                f_lines = f.readlines()
                quality_gain = []
                num_chunks = []
                for f_line in f_lines:
                    quality_gain.append(float(f_line.split('\t')[3]) - float(f_line.split('\t')[5]))
                quality_gain_per_content[content_name] = np.average(quality_gain)


    sort_orders = sorted(quality_gain_per_content.items(), key=lambda x: x[1], reverse=True)
    for i in sort_orders:
        print(i[0], i[1])
