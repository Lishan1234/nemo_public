import argparse
import os
import glob
import operator

import numpy as np

from tool.video import profile_video
from dnn.model.nemo_s import NEMO_S

from evaluation.libvpx_results import *
from evaluation.cache_profile_results import *
from tool.mac import *
from tool.mobile import *

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

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

    args = parser.parse_args()

    #dnn
    scale = int(args.hr_resolution // args.lr_resolution)
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)

    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval_section3_01.txt'.format(args.lr_resolution))
    with open(log_file, 'w') as f:
        #video, directory
        lr_video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
        lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
        lr_video_profile = profile_video(lr_video_file)
        lr_video_name = os.path.basename(lr_video_file)
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log')

        impact = []
        error = []
        for i in range(75):
            postfix = 'chunk{:04d}'.format(i)

            #quality estimation error: Random_Eval_0.5
            cache_log_file = os.path.join(log_dir, lr_video_name, nemo_s.name, 'NEMO_0.5', postfix, 'quality.txt')
            with open(cache_log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('\t')
                    estimation_error = float(line[1]) - float(line[4])
                    if estimation_error >= 0:
                        error.append(estimation_error)
        print(np.average(error), np.min(error), np.max(error))
