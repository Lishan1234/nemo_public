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
    parser.add_argument('--chunk_idx', type=int, required=True)
    parser.add_argument('--gop', type=int, default=120)

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
    log_file = os.path.join(log_dir, 'eval03_01_a.txt')

    #video
    dataset_dir = os.path.join(args.dataset_rootdir, args.content)
    print(dataset_dir)
    lr_video_file = os.path.abspath(glob.glob(os.path.join(dataset_dir, 'video', '{}p*'.format(args.lr_resolution)))[0])
    lr_video_name = os.path.basename(lr_video_file)
    assert(os.path.exists(lr_video_file))
    start_idx = args.gop * args.chunk_idx
    end_idx = args.gop * (args.chunk_idx + 1)

    #log
    with open(log_file, 'w') as f:
        #NEMO
        nemo_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}.profile'.format(APS_NEMO.NAME1, args.threshold))
        nemo_quality = libvpx_quality(nemo_log_dir)[start_idx:end_idx]

        #No motion vector
        no_mv_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}.profile'.format(APS_NEMO.NAME1, args.threshold), 'no_mv')
        no_mv_quality = libvpx_quality(no_mv_log_dir)[start_idx:end_idx]

        #No residual
        no_residual_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}.profile'.format(APS_NEMO.NAME1, args.threshold), 'no_residual')
        no_residual_quality = libvpx_quality(no_residual_log_dir)[start_idx:end_idx]

        #No motion vector & No residual
        no_mv_residual_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}.profile'.format(APS_NEMO.NAME1, args.threshold), 'no_mv_residual')
        no_mv_residual_quality = libvpx_quality(no_mv_residual_log_dir)[start_idx:end_idx]

        for i, quality in enumerate(zip(nemo_quality, no_mv_quality, no_residual_quality, no_mv_residual_quality)):
            f.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(i, quality[0], quality[1], quality[2], quality[3]))
