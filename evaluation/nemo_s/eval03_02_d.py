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
    log_file = os.path.join(log_dir, 'eval03_02_d.txt')
    postfix = 'chunk{:04d}'.format(args.chunk_idx)

    #video
    dataset_dir = os.path.join(args.dataset_rootdir, args.content)
    lr_video_file = os.path.abspath(glob.glob(os.path.join(dataset_dir, 'video', '{}p*'.format(args.lr_resolution)))[0])
    lr_video_name = os.path.basename(lr_video_file)
    assert(os.path.exists(lr_video_file))

    #log
    out_degree_log_dir =os.path.join(dataset_dir, 'log', lr_video_name, postfix)
    video_out_degree = chunk_out_degree(out_degree_log_dir, None)
    video_avg_out_degree = np.average(video_out_degree)
    video_std_out_degree = np.std(video_out_degree)
    video_out_degree_percentile =  np.percentile(video_out_degree,[0, 25, 50, 75, 100], interpolation='nearest')
    with open(log_file, 'w') as f:
        #|anchor points|
        nemo_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(APS_NEMO.NAME1, args.threshold), postfix)
        num_anchor_points = chunk_anchor_points(nemo_log_dir)

        #NEMO
        nemo_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}.profile'.format(APS_NEMO.NAME1, args.threshold), postfix)
        nemo_ap_idx = chunk_anchor_point_index(nemo_log_dir)
        nemo_out_degree = chunk_out_degree(out_degree_log_dir, nemo_ap_idx)
        nemo_avg_out_degree = np.average(nemo_out_degree)
        nemo_std_out_degree = np.std(nemo_out_degree)
        nemo_out_degree_percentile =  np.percentile(nemo_out_degree,[0, 25, 50, 75, 100], interpolation='nearest')

        #Uniform
        uniform_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(APS_Uniform.NAME1, num_anchor_points), postfix)
        uniform_ap_idx = chunk_anchor_point_index(uniform_log_dir)
        uniform_out_degree = chunk_out_degree(out_degree_log_dir, uniform_ap_idx)
        uniform_avg_out_degree = np.average(uniform_out_degree)
        uniform_std_out_degree = np.std(uniform_out_degree)
        uniform_out_degree_percentile =  np.percentile(uniform_out_degree,[0, 25, 50, 75, 100], interpolation='nearest')

        #Random
        random_log_dir = os.path.join(dataset_dir, 'log', lr_video_name, nemo_s.name, '{}_{}.tmp'.format(APS_Random.NAME1, num_anchor_points), postfix)
        random_ap_idx = chunk_anchor_point_index(random_log_dir)
        random_out_degree = chunk_out_degree(out_degree_log_dir, random_ap_idx)
        random_avg_out_degree = np.average(random_out_degree)
        random_std_out_degree = np.std(random_out_degree)
        random_out_degree_percentile =  np.percentile(random_out_degree,[0, 25, 50, 75, 100], interpolation='nearest')

        f.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(nemo_avg_out_degree, uniform_avg_out_degree, random_avg_out_degree, video_avg_out_degree))
        f.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(nemo_std_out_degree, uniform_std_out_degree, random_std_out_degree, video_std_out_degree))
