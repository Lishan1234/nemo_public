import argparse
import os

import tensorflow as tf

from tool.video import profile_video, FFmpegOption
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from dnn.model.nemo_s import NEMO_S

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_file', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
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
    parser.add_argument('--chunk_idx', default=None, type=int)
    parser.add_argument('--num_decoders', default=24, type=int)
    parser.add_argument('--mode', choices=['uniform','random','nemo'], required=True)
    parser.add_argument('--task', choices=['profile','summary'], required=True)

    args = parser.parse_args()

    #scale, nhwc
    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    lr_video_info = profile_video(lr_video_file)
    hr_video_info = profile_video(hr_video_file)
    scale = int(hr_video_info['height'] / lr_video_info['height'])
    nhwc = [1, lr_video_info['height'], lr_video_info['width'], 3]

    #model (restore)
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale)
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), nemo_s.name)
    checkpoint = nemo_s.load_checkpoint(checkpoint_dir)
    checkpoint.model.scale = scale
    checkpoint.model.nhwc = nhwc

    aps = None
    if args.mode == 'uniform':
        aps = APS_Uniform(checkpoint.model, args.vpxdec_file, args.dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold)
    elif args.mode == 'random':
        aps = APS_Random(checkpoint.model, args.vpxdec_file, args.dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold)
    elif args.mode == 'nemo':
        aps = APS_NEMO(checkpoint.model, args.vpxdec_file, args.dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, args.threshold, args.num_decoders)
    else:
        raise NotImplementedError

    if args.task == 'profile':
        if args.chunk_idx is None:
            num_chunks = int(lr_video_info['duration'] // (args.gop / lr_video_info['frame_rate']))
            for i in range(num_chunks):
                aps.run(i)
        else:
            aps.run(args.chunk_idx)
    elif args.task == 'summary':
        aps.summary()
    else:
        raise NotImplementedError
