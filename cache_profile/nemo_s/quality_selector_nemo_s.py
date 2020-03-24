import argparse
import os
import math
import glob
import numpy as np
import json

import tensorflow as tf

from tool.video import profile_video, FFmpegOption
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from cache_profile.anchor_point_selector_nemo_bound import APS_NEMO_Bound
from cache_profile.anchor_point_selector_uniform_eval import APS_Uniform_Eval
from cache_profile.anchor_point_selector_random_eval import APS_Random_Eval
from dnn.model.nemo_s import NEMO_S

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_file', type=str, required=True)
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--reference_content', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--lr_resolution', type=int, required=True)
    parser.add_argument('--hr_resolution', type=int, required=True)

    #dataset
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #dnn
    parser.add_argument('--num_filters', type=int, nargs='+', required=True)
    parser.add_argument('--num_blocks', type=int, nargs='+', required=True)

    #anchor point selector
    parser.add_argument('--aps_class', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--gop', type=int, required=True)
    parser.add_argument('--bound', type=int, required=True)

    #device
    parser.add_argument('--device_id', type=str, nargs='+', required=True)

    args = parser.parse_args()

    #validation
    if args.aps_class == 'nemo_bound':
        assert(args.bound is not None)

    #scale, nhwc
    scale = int(args.hr_resolution // args.lr_resolution)

    #models
    models = []
    for num_blocks, num_filters in zip(args.num_blocks, args.num_filters):
        nemo_s = NEMO_S(num_blocks, num_filters, scale)
        models.append(nemo_s)

    #cache_profiler
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random
    elif args.aps_calss == 'nemo_bound':
        aps_class = APS_NEMO_Bound
    else:
        raise NotImplementedError

    #reference video
    latency_dict = {}
    for device_id in args.device_id:
        latency_dict[device_id] = {}

        for model in models:
            anchor_point = []
            non_anchor_frame = []

            video_file = os.path.abspath(glob.glob(os.path.join(args.dataset_rootdir, args.reference_content,  'video', '{}p*'.format(args.lr_resolution)))[0])
            video_name = os.path.basename(video_file)
            if aps_class == APS_NEMO_Bound:
                latency_log = os.path.join(args.dataset_rootdir, args.reference_content, 'log', video_name, model.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_id, 'latency.txt')
                metadata_log = os.path.join(args.dataset_rootdir, args.reference_content, 'log', video_name, model.name, '{}_{}_{}.profile'.format(aps_class.NAME1, args.bound, args.threshold), device_id, 'metadata.txt')
            else:
                latency_log = os.path.join(args.dataset_rootdir, args.reference_content, 'log', video_name, model.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_id, 'latency.txt')
                metadata_log = os.path.join(args.dataset_rootdir, args.reference_content, 'log', video_name, model.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold), device_id, 'metadata.txt')

            with open(latency_log, 'r') as f0, open(metadata_log, 'r') as f1:
                latency_lines = f0.readlines()
                metadata_lines = f1.readlines()

                for latency_line, metadata_line in zip(latency_lines, metadata_lines):
                    latency_result = latency_line.strip().split('\t')
                    metadata_result = metadata_line.strip().split('\t')

                    #anchor point
                    if int(metadata_result[2]) == 1:
                        anchor_point.append(float(latency_result[2]))
                    #non-anchor frame
                    elif int(metadata_result[2]) == 0:
                        non_anchor_frame.append(float(latency_result[2]))
                    else:
                        raise RuntimeError

                latency_dict[device_id][model.name] = {}
                latency_dict[device_id][model.name]['anchor_point'] = np.average(anchor_point)
                latency_dict[device_id][model.name]['non_anchor_frame'] = np.average(non_anchor_frame)

    selected_model = {}
    for device_id in args.device_id:
        for content in args.content:
            if content not in selected_model.keys():
                selected_model[content] = {}
            video_file = os.path.abspath(glob.glob(os.path.join(args.dataset_rootdir, content,  'video', '{}p*'.format(args.lr_resolution)))[0])
            video_name = os.path.basename(video_file)
            fps = profile_video(video_file)['frame_rate']

            real_time_model = None
            for model in models:
                is_real_time = True
                if aps_class == APS_NEMO_Bound:
                    log = os.path.join(args.dataset_rootdir, content, 'log', video_name, model.name, '{}_{}_{}'.format(aps_class.NAME1, args.bound, args.threshold), 'quality.txt')
                else:
                    log = os.path.join(args.dataset_rootdir, content, 'log', video_name, model.name, '{}_{}'.format(aps_class.NAME1, args.threshold), 'quality.txt')
                with open(log, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split('\t')
                        num_anchor_points = int(line[1])
                        max_latency = 0
                        latency = num_anchor_points * latency_dict[device_id][model.name]['anchor_point'] + (args.gop - num_anchor_points) * latency_dict[device_id][model.name]['non_anchor_frame']
                        if latency > (args.gop / fps) * 1000:
                            is_real_time = False
                            break

                    if is_real_time is True:
                        real_time_model = model

            if real_time_model is None:
                selected_model[content][device_id] = {}
                selected_model[content][device_id]['num_blocks'] = models[0].num_blocks
                selected_model[content][device_id]['num_filters'] = models[0].num_filters
                selected_model[content][device_id]['scale'] = models[0].scale
                selected_model[content][device_id]['aps_class'] = args.aps_class
                selected_model[content][device_id]['bound'] = 12
                selected_model[content][device_id]['threshold'] = args.threshold
            else:
                selected_model[content][device_id] = {}
                selected_model[content][device_id]['num_blocks'] = real_time_model.num_blocks
                selected_model[content][device_id]['num_filters'] = real_time_model.num_filters
                selected_model[content][device_id]['scale'] = real_time_model.scale
                selected_model[content][device_id]['aps_class'] = args.aps_class
                selected_model[content][device_id]['bound'] = args.bound
                selected_model[content][device_id]['threshold'] = args.threshold

    for content in args.content:
        video_file = os.path.abspath(glob.glob(os.path.join(args.dataset_rootdir, content,  'video', '{}p*'.format(args.lr_resolution)))[0])
        video_name = os.path.basename(video_file)
        json_file = os.path.join(args.dataset_rootdir, content, 'log', video_name, 'device_to_dnn.json')

        with open(json_file, 'w') as f:
            json.dump(selected_model[content], f)

    log_file = os.path.join(args.dataset_rootdir, 'evaluation', 'device_to_dnn_{}p.txt'.format(args.lr_resolution))
    with open(log_file, 'w') as f:
        f.write('Content\t{}\n'.format('\t'.join(args.device_id)))
        for content in args.content:
            f.write(content)
            for device_id in args.device_id:
                f.write('\t{}'.format(selected_model[content][device_id]['num_filters']))
            f.write('\n')
