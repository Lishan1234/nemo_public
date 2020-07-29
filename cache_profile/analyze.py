import argparse
import os
import math
import glob
import numpy as np
import json

from nemo.tool.video import profile_video
from nemo.tool.mobile import id_to_name
import nemo.dnn.model

#contents = ['product_review, how_to, vlogs, skit, game_play, haul, challenge, education, favorite, unboxing']
#indexes = [1, 2, 3]
#resolution = [240, 360, 480]

contents = ['game_play', 'skit', 'haul']
indexes = [1, 2]
resolutions = [240, 360, 480]
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

video_name_info = {
    240: '240p_512kbps_s0_d300.webm',
    360: '360p_1024kbps_s0_d300.webm',
    480: '480p_1600kbps_s0_d300.webm',
}

algorithms = ['nemo_0.5_12', 'nemo_0.5_24']

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
    log_dir = os.path.join(args.data_dir, 'log')
    log_path = os.path.join(log_dir, 'cache_profile_analysis.txt')
    os.makedirs(log_dir, exist_ok=True)
    with open (log_path, 'w') as f:
        for content in contents:
            for index in indexes:
                for resolution in resolutions:
                    video_name = video_name_info[resolution]
                    content_name = '{}{}'.format(content, index)

                    for quality in qualities:
                        video_path = os.path.join(args.data_dir, content_name, 'video', video_name)
                        video_profile = profile_video(video_path)
                        scale = args.output_height // video_profile['height']
                        num_blocks = num_blocks_info[quality][video_profile['height']]
                        num_filters = num_filters_info[quality][video_profile['height']]
                        model_name = nemo.dnn.model.build(args.model_type, num_blocks, num_filters, scale, args.upsample_type).name

                        nemo_05_log_path = os.path.join(args.data_dir, content_name, 'log', video_name, model_name, 'quality_nemo_0.5.txt')
                        nemo_05_24_log_path = os.path.join(args.data_dir, content_name, 'log', video_name, model_name, 'quality_nemo_0.5_24.txt')

                        if not os.path.exists(nemo_05_24_log_path):
                            f.write('{}\t{}\t{}\tnemo_0.5_24\tfailed\n'.format(video_name, content_name, model_name))
                        else:
                            with open(nemo_05_log_path, 'r') as f0, open(nemo_05_24_log_path, 'r') as f1:
                                f0_lines = f0.readlines()
                                f1_lines = f1.readlines()

                                nemo_05_24_num_abnormal_chunks = 0
                                nemo_075_24_num_abnormal_chunks = 0
                                nemo_05_24_pass = 0
                                nemo_075_24_pass = 0
                                count = 0

                                for f0_line, f1_line in zip(f0_lines, f1_lines):
                                    nemo_05_num_chunks = int(f0_line.split('\t')[1])
                                    nemo_05_24_num_chunks = int(f1_line.split('\t')[1])
                                    nemo_05_quality = float(f0_line.split('\t')[3])
                                    nemo_05_24_quality = float(f1_line.split('\t')[3])
                                    per_frame_sr_quality = float(f0_line.split('\t')[4])

                                    count += 1

                                    if nemo_05_24_num_chunks == 24:
                                        nemo_05_24_quality_degradation = per_frame_sr_quality - nemo_05_24_quality

                                        if nemo_05_24_quality_degradation > 0.5:
                                            nemo_05_24_num_abnormal_chunks += 1

                                        if nemo_05_24_quality_degradation > 0.75:
                                            nemo_075_24_num_abnormal_chunks += 1

                                if (nemo_05_24_num_abnormal_chunks / count) * 100 <= 10:
                                    nemo_05_24_pass = 1
                                if (nemo_075_24_num_abnormal_chunks / count) * 100 <= 10:
                                    nemo_075_24_pass = 1
                                f.write('{}\t{}\t{}\tnemo_0.5_24\t{}\t{}\n'.format(video_name, content_name, model_name, nemo_05_24_pass, nemo_075_24_pass))
