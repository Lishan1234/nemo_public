import argparse
import os
import math

import tensorflow as tf

from tool.video import profile_video, FFmpegOption
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from cache_profile.anchor_point_selector_uniform_eval import APS_Uniform_Eval
from cache_profile.anchor_point_selector_random_eval import APS_Random_Eval
from cache_profile.anchor_point_selector_nemo_bound import APS_NEMO_Bound
from dnn.model.nemo_s import NEMO_S

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_file', type=str, required=True)
    parser.add_argument('--dataset_rootdir', type=str, required=True)
    parser.add_argument('--content', type=str, nargs='+', required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #dataset
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--gop', type=int, required=True)
    parser.add_argument('--max_num_anchor_points', type=int, default=None)
    parser.add_argument('--chunk_idx', default=None, type=str)
    parser.add_argument('--num_decoders', default=24, type=int)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--task', choices=['profile','summary'], required=True)
    parser.add_argument('--profile_all', action='store_true')

    args = parser.parse_args()

    for content in args.content:
        dataset_dir = os.path.join(args.dataset_rootdir, content)

        #scale, nhwc
        lr_video_file = os.path.join(dataset_dir, 'video', args.lr_video_name)
        hr_video_file = os.path.join(dataset_dir, 'video', args.hr_video_name)
        lr_video_info = profile_video(lr_video_file)
        hr_video_info = profile_video(hr_video_file)
        scale = int(hr_video_info['height'] / lr_video_info['height'])
        nhwc = [1, lr_video_info['height'], lr_video_info['width'], 3]

        #model (restore)
        nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale)
        ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
        checkpoint_dir = os.path.join(dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), nemo_s.name)
        checkpoint = nemo_s.load_checkpoint(checkpoint_dir)
        checkpoint.model.scale = scale
        checkpoint.model.nhwc = nhwc

        aps = None
        if args.mode == 'uniform':
            aps = APS_Uniform(checkpoint.model, args.vpxdec_file, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold)
        elif args.mode == 'random':
            aps = APS_Random(checkpoint.model, args.vpxdec_file, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold)
        elif args.mode == 'nemo':
            aps = APS_NEMO(checkpoint.model, args.vpxdec_file, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold, args.num_decoders, args.profile_all)
        elif args.mode == 'uniform_eval':
            aps = APS_Uniform_Eval(checkpoint.model, args.vpxdec_file, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold, args.num_decoders, args.profile_all)
        elif args.mode == 'random_eval':
            aps = APS_Random_Eval(checkpoint.model, args.vpxdec_file, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold, args.num_decoders, args.profile_all)
        elif args.mode == 'nemo_bound':
            assert(args.max_num_anchor_points is not None)
            aps = APS_NEMO_Bound(checkpoint.model, args.vpxdec_file, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold, args.num_decoders, args.max_num_anchor_points, args.profile_all)
        else:
            raise NotImplementedError

        if args.task == 'profile':
            if args.chunk_idx is None:
                num_chunks = int(math.ceil(lr_video_info['duration'] / (args.gop / lr_video_info['frame_rate'])))
                for i in range(num_chunks):
                    aps.run(i)
            else:
                if ',' in args.chunk_idx:
                    start_index = int(args.chunk_idx.split(',')[0])
                    end_index = int(args.chunk_idx.split(',')[1])
                    for i in range(start_index, end_index + 1):
                        aps.run(i)
                else:
                    aps.run(int(args.chunk_idx))
        elif args.task == 'summary':
            if args.chunk_idx is None:
                num_chunks = int(math.ceil(lr_video_info['duration'] / (args.gop / lr_video_info['frame_rate'])))
                aps.summary(0, num_chunks)
            else:
                if ',' in args.chunk_idx:
                    start_index = int(args.chunk_idx.split(',')[0])
                    end_index = int(args.chunk_idx.split(',')[1])
                    aps.summary(start_index, end_index + 1)
                else:
                    aps.summary(int(args.chunk_idx), int(args.chunk_idx) + 1)
        else:
            raise NotImplementedError
