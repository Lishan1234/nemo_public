import os
import sys
import argparse
import shlex
import math

import numpy as np
import tensorflow as tf

from tool.video import profile_video
from tool.libvpx import Frame, CacheProfile, save_frames, setup_sr_frames, load_frames, measure_offline_cache_quality
from dnn.model.nas_s import NAS_S
from dnn.utility import raw_quality

class APS_Uniform():
    def __init__(self, model, vpxdec_file, dataset_dir, lr_video_name, hr_video_name, gop, threshold):
        self.model = model
        self.vpxdec_file = vpxdec_file
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.threshold = threshold

    def run(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, postfix)

        #setup lr, sr, hr frames
        save_frames(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.gop, chunk_idx)
        save_frames(self.vpxdec_file, self.dataset_dir, self.hr_video_name, self.gop, chunk_idx)
        #setup_sr_frames(self.vpxdec_file, self.dataset_dir, self.lr_video_name, self.gop, chunk_idx, self.model)

        #load per-frame super-resolution quality
        lr_raw_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        sr_raw_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        hr_raw_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        quality_dnn, quality_bilinear = raw_quality(lr_raw_dir, sr_raw_dir, hr_raw_dir, self.model.nhwc, self.model.scale, precision=tf.uint8)

        #load frames (index)
        frames = load_frames(self.dataset_dir, self.lr_video_name, self.gop, chunk_idx)

        #select/evaluate anchor points
        log_file = os.path.join(log_dir, 'quality_{}.txt'.format(self.__class__.__name__))
        num_anchor_points = 0
        with open(log_file, 'w') as f:
            for i in range(len(frames)):
                #select anchor points uniformly
                num_anchor_points = i + 1
                cache_profile = CacheProfile.fromframes(frames, profile_dir, '{}_{}'.format(self.__class__.__name__, num_anchor_points))
                for j in range(num_anchor_points):
                    idx = j * math.floor(len(frames) / num_anchor_points)
                    cache_profile.add_anchor_point(frames[idx])
                quality_cache = measure_offline_cache_quality(self.vpxdec_file, self.dataset_dir, self.lr_video_name, \
                                                            self.hr_video_name, postfix, self.model, cache_profile)

                #log quality, quality diff, etc:w
                quality_diff = quality_dnn - quality_cache
                quality_error =  np.percentile(np.asarray(dnn_quality) - np.asarray(quality_cache) \
                                                            ,[95, 99, 100], interpolation='nearest')
                frame_count_1 = sum(map(lambda x : x >= 0.5, quality_diff))
                frame_count_2 = sum(map(lambda x : x >= 1.0, quality_diff))
                log = '{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\n'.format(num_anchor_points,
                                        np.average(quality_cache), np.average(quality_sr), np.average(quality_bilinear),
                                        frame_count_1, frame_count_2, '\t'.join(str(np.round(x, 2)) for x in quality_error))
                f.write(log)

                print('{} anchor points: PSNR(Cache)={:.2f}, PSNR(SR)={:.2f}'.format(num_anchor_points,
                                                                np.average(quality_cache), np.average(quality_sr)))

                return


                #check quality difference
                if np.average(quality_diff) <= self.threshold:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache Erosion Analyzer')

    #directory, path
    parser.add_argument('--vpxdec_path', type=str, required=True)
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
    parser.add_argument('--chunk_idx', default=None)

    args = parser.parse_args()

    #scale, nhwc
    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    lr_video_info = profile_video(lr_video_file)
    hr_video_info = profile_video(hr_video_file)
    scale = int(hr_video_info['height'] / lr_video_info['height'])
    nhwc = [1, lr_video_info['height'], lr_video_info['width'], 3]

    #model (restore)
    nas_s = NAS_S(args.num_blocks, args.num_filters, scale, None)
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), nas_s.name)
    checkpoint = nas_s.load_checkpoint(checkpoint_dir)
    checkpoint.model.scale = scale
    checkpoint.model.nhwc = nhwc

    #aps_baseline = APS_Uniform(checkpoint.model, args.vpxdec_path, args.dataset_dir, args.lr_video_name, args.hr_video_name, rgs.gop, args.threshold)
    #aps_baseline.run()
