import argparse
import os
import math
import glob
import numpy as np
import json

import tensorflow as tf

from tool.video import profile_video

content_order = {'product_review': 0, 'how_to': 1, 'vlogs': 2, 'game_play_1': 3, 'skit': 4,
                'haul': 5, 'challenge':6, 'favorite': 7, 'education': 8, 'unboxing': 9}

#TODO: test each (devices, dnns, content) and save a log at {content}/log/processor_to_quality.json
#TODO: save processor names
#TODO: test with 0th chunk

class QualitySelector():
    def __init__():
        self.dataset_dir = dataset_dir
        self.video_name = video_name
        self.gop = gop
        self.models = models
        self.processors = processors
        self.algorithms = algorithms
        self.fallback_algorithm_type = fallback_algorithm_type

    def measure_sample_video_latency():
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

    #TODO: sort a model by multiple attributes (num_blocks, num_filters)
    #TODO: sort a algorithm type by {maximum_num_anchor_points} (margin, max_num_anchor_points)
    #TODO: support multiple algorithm types

    def _algorithm_key(self):
        pass

    def _model_key(self):
        pass

    def select_quality():
        processor_to_quality = {}
        for processor in self.processors:
            video_path = os.path.join(self.dataset_dir, 'video', self.video_name)
            fps = profile_video(video_path)['frame_rate']

            selected_model = None
            for algorithm in sorted(self.algorithms, key=lambda x: self._algorith_key(x))
                for model in sorted(self.models, key=lambda x: self._model_key(x)):
                    is_real_time = True
                    log_path = os.path.join(self.dataset_dir, 'log', video_name, model.name, 'quality_{}.txt'.format(self.algorithm_type))
                    with open(log, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip().split('\t')
                            num_anchor_points = int(line[1])
                            num_frames = int(line[2]) #TODO: log num_frames
                            latency = num_anchor_points * latency_dict[processor][model.name]['anchor_point'] + (num_frames - num_anchor_points) * latency_dict[processor][model.name]['non_anchor_frame']
                            if latency > (self.gop / fps) * 1000:
                                is_real_time = False
                                break
                        if is_real_time is True:
                            selected_model = model

            if selected_model is None:
                processor_to_quality[processor] = {}
                processor_to_quality[processor]['num_blocks'] = selected_model.num_blocks
                processor_to_quality[processor]['num_filters'] = selected_model.num_filters
                processor_to_quality[processor]['scale'] = selected_model.scale
                processor_to_quality[processor]['algorithm_type'] = self.algorithm_type
            else:
                raise RuntimeError('No model is selected')

        json_path = os.path.join(self.dataset_dir, 'log', video_name, 'processor_to_quality.json')
        with open(json_path, 'w') as f:
            json.dump(processor_to_quality, f)

        log_path = os.path.join(self.dataset_dir, 'evaluation', 'device_to_dnn_{}p.txt'.format(args.lr_resolution))
        with open(log_path, 'w') as f:
            f.write('Content\t{}\n'.format('\t'.join(args.device_id)))
            for content in args.content:
                f.write(content)
                for device_id in args.device_id:
                    f.write('\t{}'.format(processor_to_quality[content][device_id]['num_filters']))
                f.write('\n')

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

    #sort
    args.content.sort(key=lambda val: content_order[val])

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
    elif args.aps_class == 'nemo_bound':
        aps_class = APS_NEMO_Bound
    else:
        raise NotImplementedError
