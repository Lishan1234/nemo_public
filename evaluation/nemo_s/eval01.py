import argparse
import os

from tool.video import profile_video
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S

from evaluation.mac import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    args = parser.parse_args()

    #dnn
    lr_video_file = os.path.join(args.dataset_rootdir, args.content[0], 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_rootdir, args.content[0], 'video', args.hr_video_name)
    lr_video_info = profile_video(lr_video_file)
    hr_video_info = profile_video(hr_video_file)
    scale = int(hr_video_info['height'] / lr_video_info['height'])
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale)

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
    log_file = os.path.join(log_dir, 'eval01_{}p.txt'.format(lr_video_info['height']))
    with open(log_file, 'w') as f:
        for content in args.content:
            dataset_dir = os.path.join(args.dataset_rootdir, content)
            result = mac_and_quality_gain(dataset_dir, args.lr_video_name, nemo_s, aps_class, args.threshold)
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(content, \
                    result['avg_quality']['cache'], result['avg_quality']['dnn'],
                    result['avg_mac']['cache'], result['avg_mac']['dnn'],
                    result['std_mac']['cache'], result['std_mac']['dnn'],
                    result['avg_norm_mac']['cache'], result['avg_norm_mac']['dnn'],
                    result['std_norm_mac']['cache'], result['std_norm_mac']['dnn']))

