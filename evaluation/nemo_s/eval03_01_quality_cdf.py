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

def load_anchor_point_quality(log_dir, index):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    cache_quality = None
    dnn_quality = None

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        if index is not None:
            assert index < len(quality_lines)
            quality_line = quality_lines[index].strip().split('\t')
            cache_quality = float(quality_line[1]) - float(quality_line[3])
            dnn_quality = float(quality_line[2]) - float(quality_line[3])
        else:
            quality_line = quality_lines[-1].strip().split('\t')
            cache_quality = float(quality_line[1]) - float(quality_line[3])
            dnn_quality = float(quality_line[2]) - float(quality_line[3])

    return cache_quality, dnn_quality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
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

    #sort
    args.content.sort(key=lambda val: content_order[val])

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval03_01_quality_cdf_{}p.txt'.format(args.lr_resolution))

    nemo_quality = []
    fast_quality = []
    per_frame_quality = []
    with open(log_file, 'w') as f:
        for content in args.content:
            #video, directory
            lr_video_dir = os.path.join(args.dataset_rootdir, content, 'video')
            lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
            lr_video_profile = profile_video(lr_video_file)
            lr_video_name = os.path.basename(lr_video_file)
            log_dir = os.path.join(args.dataset_rootdir, content, 'log')
            scale = int(args.hr_resolution // args.lr_resolution)
            nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)

            nemo_chunk_quality = []
            fast_chunk_quality = []
            per_frame_chunk_quality = []

            for i in range(0, 74):
                chunk_name = 'chunk{:04d}'.format(i)

                cache_profile_name = '{}_{}'.format(APS_NEMO.NAME1, args.threshold)
                cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, chunk_name)
                cache_quality, dnn_quality = load_anchor_point_quality(cache_log_dir, None)
                nemo_chunk_quality.append(cache_quality)

                cache_profile_name = '{}_{}'.format(APS_Uniform.NAME1, args.threshold)
                cache_log_dir = os.path.join(log_dir, lr_video_name, nemo_s.name, cache_profile_name, chunk_name)
                cache_quality, dnn_quality_ = load_anchor_point_quality(cache_log_dir, 0)
                fast_chunk_quality.append(cache_quality)

                per_frame_chunk_quality.append(dnn_quality)

            nemo_quality.append(np.average(nemo_chunk_quality))
            fast_quality.append(np.average(fast_chunk_quality))
            per_frame_quality.append(np.average(per_frame_chunk_quality))

        nemo_quality.sort()
        fast_quality.sort()
        per_frame_quality.sort()

        print(np.average(np.asarray(fast_quality) - np.asarray(per_frame_quality)))
        print(np.max(np.asarray(fast_quality) - np.asarray(per_frame_quality)))
        print(np.min(np.asarray(fast_quality) - np.asarray(per_frame_quality)))

        count = 0
        f.write('0\t0\t0\t0\n')
        for nq, fq, pfq in zip(nemo_quality, fast_quality, per_frame_quality):
            f.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(count/len(nemo_quality), nq, fq, pfq))
            f.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format((count+1)/len(nemo_quality), nq, fq, pfq))
            count += 1
