import argparse
import os
import glob
import operator

import numpy as np

from tool.video import profile_video
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S

from evaluation.libvpx_results import *
from evaluation.cache_profile_results import *
from tool.mac import *

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

def load_num_anchor_points(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    num_anchor_points = []

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()
        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            num_anchor_points.append(int(quality_line[1]))

    return num_anchor_points

def load_chunk_quality(log_dir, chunk_idx):
    quality_log_file = os.path.join(log_dir, 'quality.txt')

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        quality_line = quality_lines[chunk_idx].strip().split('\t')
        return float(quality_line[1])

def load_frame_quality(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    quality = []

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality.append(float(quality_line[1]))

    return quality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dataset
    parser.add_argument('--gop', type=int, default=120)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point
    parser.add_argument('--threshold', type=float, required=True)

    args = parser.parse_args()

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file_0 = os.path.join(log_dir, 'eval03_03_chunk_cache_erosion_{}p.txt'.format(args.lr_resolution))
    log_file_1 = os.path.join(log_dir, 'eval03_03_frame_cache_erosion_{}p.txt'.format(args.lr_resolution))

    with open(log_file_0, 'w') as f_0, open(log_file_1, 'w') as f_1:
        #video, directory
        lr_video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
        lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
        lr_video_profile = profile_video(lr_video_file)
        lr_video_name = os.path.basename(lr_video_file)
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log')
        scale = int(args.hr_resolution // args.lr_resolution)
        nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)

        cache_profile_name = '{}_{}'.format(APS_NEMO.NAME1, args.threshold)
        cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name)
        num_anchor_points = load_num_anchor_points(cache_log_dir)

        #chunk quality
        for i in range(75):
            postfix = 'chunk{:04d}'.format(i)
            cache_profile_name = '{}_{}'.format(APS_NEMO.NAME1, args.threshold)
            cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, postfix)
            nemo_chunk_quality = load_chunk_quality(cache_log_dir, -1)

            cache_profile_name = '{}_{}'.format(APS_Uniform.NAME1, args.threshold)
            cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, postfix)
            uniform_chunk_quality = load_chunk_quality(cache_log_dir, num_anchor_points[i] - 1)

            cache_profile_name = '{}_{}'.format(APS_Random.NAME1, args.threshold)
            cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, postfix)
            random_chunk_quality = load_chunk_quality(cache_log_dir, num_anchor_points[i] - 1)

            f_0.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(i, nemo_chunk_quality, uniform_chunk_quality, random_chunk_quality))

        #frame quality
        postfix = 'chunk0010'
        cache_profile_name = '{}_{}.profile'.format(APS_NEMO.NAME1, args.threshold)
        cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, postfix)
        nemo_frame_quality = load_frame_quality(cache_log_dir)

        cache_profile_name = '{}_{}'.format(APS_Uniform.NAME1, num_anchor_points[10])
        cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, postfix)
        uniform_frame_quality = load_frame_quality(cache_log_dir)

        cache_profile_name = '{}_{}.tmp'.format(APS_Random.NAME1, num_anchor_points[10])
        cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, postfix)
        random_frame_quality = load_frame_quality(cache_log_dir)

        count = 0
        for nq, uq, rq in zip(nemo_frame_quality, uniform_frame_quality, random_frame_quality):
            f_1.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(count, nq, uq, rq))
            count += 1
