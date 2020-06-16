import os, time, sys, time
import subprocess
import argparse
import collections
import json
import importlib

import numpy as np

import nemo.dnn.model
from nemo.tool.snpe import snpe_convert_model, snpe_convert_dataset, snpe_benchmark, snpe_benchmark_random_config
from nemo.tool.video import profile_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #path
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')

    #training
    parser.add_argument('--train_type', type=str, required=True)

    #dnn
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')

    #device
    parser.add_argument('--device_id', type=str, required=True)
    parser.add_argument('--runtime', type=str, default='GPU_FP16')

    args = parser.parse_args()

    lr_video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.data_dir, args.content, 'video', args.hr_video_name)
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']
    input_shape = [1, lr_video_profile['height'], lr_video_profile['width'], 3]
    output_shape = [1, hr_video_profile['height'], hr_video_profile['width'], 3]

    if (hr_video_profile['height'] % lr_video_profile['height'] == 0 and
            hr_video_profile['width'] % lr_video_profile['width'] == 0):
        model = nemo.dnn.model.build(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type, apply_clip=True)
    else:
        model = nemo.dnn.model.build(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type, output_shape=output_shape, apply_clip=True)

    if args.train_type == 'train_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
        log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name, model.name, 'snpe_random_benchmark')
    elif args.train_type == 'finetune_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, '{}_finetune'.format(model.name))
        log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name, '{}_finetune'.format(model.name), 'snpe_random_benchmark')
    else:
        raise ValueError('Unsupported training types')

    dlc_path = os.path.join(checkpoint_dir, '{}.dlc'.format(model.name))
    #if not os.path.exists(dlc_path):
    snpe_convert_model(model, input_shape, checkpoint_dir)

    json_path = snpe_benchmark_random_config(args.device_id, args.runtime, model.name, dlc_path, log_dir)
    snpe_benchmark(json_path)
