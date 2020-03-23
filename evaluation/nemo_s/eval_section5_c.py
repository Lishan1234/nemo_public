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
    parser.add_argument('--chunk_idx', type=int, required=True)

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
    log_file = os.path.join(log_dir, 'eval_section5_c.txt'.format(args.lr_resolution))
    with open(log_file, 'w') as f0:
        #video, directory
        lr_video_dir = os.path.join(args.dataset_rootdir, args.content, 'video')
        lr_video_file = os.path.abspath(glob.glob(os.path.join(lr_video_dir, '{}p*'.format(args.lr_resolution)))[0])
        lr_video_profile = profile_video(lr_video_file)
        lr_video_name = os.path.basename(lr_video_file)
        log_dir = os.path.join(args.dataset_rootdir, args.content, 'log')

        quality_log_file = os.path.join(log_dir, lr_video_name, nemo_s.name, 'Exhaustive', 'chunk{:04d}'.format(args.chunk_idx), 'quality.txt')
        print(quality_log_file)
        count = 0
        cdf_xvals = []
        with open(quality_log_file, 'r') as f1:
            lines = f1.readlines()
            print(len(lines))
            for line in lines:
                line = line.strip().split('\t')
                error = float(line[1]) - float(line[0])
                if error <= 0.5:
                    count += 1
                cdf_xvals.append(error)

        cdf_xvals = np.sort(cdf_xvals)
        cdf_yvals = np.arange(len(cdf_xvals))/float(len(cdf_xvals)-1)

        sampled_cdf_xvals = []
        sampled_cdf_yvals = []
        num_samples = len(cdf_xvals) // 10
        for i in range(num_samples):
            sampled_cdf_xvals.append(cdf_xvals[i * 10])
            sampled_cdf_yvals.append(cdf_yvals[i * 10])

        for sampled_cdf_xval, sampled_cdf_yval in zip(sampled_cdf_xvals, sampled_cdf_yvals):
            f0.write('{}\t{}\n'.format(sampled_cdf_xval, sampled_cdf_yval))

        print('Count: {}'.format(count))
