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
from tool.mobile import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dataset
    parser.add_argument('--chunk_idx', type=int, default=120)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)

    args = parser.parse_args()

    #dnn
    scale = int(args.hr_resolution // args.lr_resolution)
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval03_02_b.txt')
    postfix = 'chunk{:04d}'.format(args.chunk_idx)

    #video
    dataset_dir = os.path.join(args.dataset_rootdir, args.content)
    lr_video_file = os.path.abspath(glob.glob(os.path.join(dataset_dir, 'video', '{}p*'.format(args.lr_resolution)))[0])
    lr_video_name = os.path.basename(lr_video_file)
    assert(os.path.exists(lr_video_file))

    #log
    with open(log_file, 'w') as f:
        #NEMO
        nemo_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(APS_NEMO.NAME1, args.threshold), postfix)
        nemo_error_frames = chunk_error_frames(nemo_log_dir)

        #Uniform
        uniform_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(APS_Uniform.NAME1, args.threshold), postfix)
        uniform_error_frames = chunk_error_frames(uniform_log_dir)

        #Random
        random_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(APS_Random.NAME1, args.threshold), postfix)
        random_error_frames = chunk_error_frames(random_log_dir)

        max_len = max([len(nemo_error_frames), len(uniform_error_frames), len(random_error_frames)])

        for i in range(max_len):
            f.write('{}'.format(i+1))
            if i < len(nemo_error_frames):
                f.write('\t{}'.format(nemo_error_frames[i]))
            else:
                f.write('\t')
            if i < len(uniform_error_frames):
                f.write('\t{}'.format(uniform_error_frames[i]))
            else:
                f.write('\t')
            if i < len(random_error_frames):
                f.write('\t{}\n'.format(random_error_frames[i]))
            else:
                f.write('\t\n')
