import argparse
import os
import glob

from tool.video import profile_video
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S

from evaluation.cache_profile_results import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, nargs='+', required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    args = parser.parse_args()

    #sort
    args.content.sort()

    #dnn
    scale = int(args.hr_resolution // args.lr_resolution)
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)

    #cache_profiler
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random

    #log
    log_dir = os.path.join(args.dataset_rootdir, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval01_01_a.txt')
    with open(log_file, 'w') as f:
        for content in args.content:
            cache_avg_mac = []
            cache_std_mac = []
            dnn_avg_mac = None
            dnn_std_mac = None

            for idx, threshold in enumerate(args.threshold):
                video_dir = os.path.join(args.dataset_rootdir, content, 'video')
                lr_video_file = glob.glob(os.path.join(video_dir, '{}p*'.format(args.lr_resolution)))[0]
                lr_video_name = os.path.basename(lr_video_file)
                log_dir = os.path.join(args.dataset_rootdir, content, 'log', lr_video_name, nemo_s.name, '{}_{}'.format(aps_class.NAME1, threshold))
                result_cache, result_dnn = norm_mac(log_dir)
                cache_avg_mac.append(result_cache['avg_mac'])
                cache_std_mac.append(result_cache['std_mac'])
                if idx == 0:
                    dnn_avg_mac = result_dnn['avg_mac']
                    dnn_std_mac = result_dnn['std_mac']

            f.write('{}\t{}\t{}\t{}\t{}\n'.format(content, '\t'.join(str(x) for x in cache_avg_mac), \
                    dnn_avg_mac, '\t'.join(str(x) for x in cache_std_mac), dnn_std_mac))
