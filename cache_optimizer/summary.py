import os
import argparse
import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

from tool.tf import valid_raw_dataset
from tool.ffprobe import profile_video
from dnn.model.edsr_s import EDSR_S
from dnn.utility import resolve_bilinear
from cache_optimizer.anchor_point_selector_v1 import APS_v1
from cache_optimizer.anchor_point_selector_baseline import APS_Baseline

class Summary():
    def __init__(self, model, content_dir, input_video, compare_video, gop, quality_diff):
        self.model = model
        self.content_dir = content_dir
        self.input_video = input_video
        self.compare_video = compare_video
        self.gop = gop
        self.quality_diff = quality_diff

    def prepare_bilinear(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        lr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)

        input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
        input_video_info = profile_video(input_video_path)
        compare_video_path = os.path.join(self.content_dir, 'video', self.compare_video)
        compare_video_info = profile_video(compare_video_path)

        scale = compare_video_info['height'] // input_video_info['height']
        valid_raw_ds = valid_raw_dataset(lr_image_dir, hr_image_dir,  input_video_info['height'], input_video_info['width'], scale, \
                                        pattern='[0-9][0-9][0-9][0-9].raw')
        bilinear_psnr_values = []
        for idx, img in enumerate(valid_raw_ds):
            lr = img[0][0]
            hr = img[1][0]
            bilinear = resolve_bilinear(lr, compare_video_info['height'], compare_video_info['width'])
            bilinear_psnr_value = tf.image.psnr(bilinear, hr, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

        log_path = os.path.join(log_dir, postfix, 'quality_bilinear.txt')
        with open(log_path, 'w') as f:
            f.write('\n'.join(str(np.round(bilinear_psnr_value, 2)) for bilinear_psnr_value in bilinear_psnr_values))


    def summary_aps_v1(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log = None

        with open(os.path.join(log_dir, postfix, 'quality_{}.txt'.format(APS_v1.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                num_anchor_points = line[0]
                quality = line[1]
                quality_sr = line[2]
                quality_diff = line[3]
                quality_error_90 = line[4]
                quality_error_95 = line[5]
                #quality_error_100 = line[6]
                quality_error_100 = 0

                if float(quality_diff) < self.quality_diff:
                    break

        #TODO:
        bilinear_psnr_values = []
        with open(os.path.join(log_dir, postfix, 'quality_bilinear.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                bilinear_psnr_value = float(line)
                bilinear_psnr_values.append(bilinear_psnr_value)
        quality_bilinear = np.average(bilinear_psnr_values)

        log = '{}\t{}\t{}\t{}\t{}\t{}'.format(num_anchor_points, quality, quality_bilinear, quality_sr, quality_diff, quality_error_90, quality_error_95, quality_error_100)

        #TODO: mac // cache in libvpx

        with open(os.path.join(log_dir, 'latency_{}_0.txt'.format(APS_v1.__name__)), 'r') as f0, \
             open(os.path.join(log_dir, 'latency_{}_1.txt'.format(APS_v1.__name__)), 'r') as f1, \
             open(os.path.join(log_dir, 'latency_{}_2.txt'.format(APS_v1.__name__)), 'r') as f2:
            line0 = f0.readlines()[chunk_idx].strip().split('\t')
            line1 = f1.readlines()[chunk_idx].strip().split('\t')
            line2 = f2.readlines()[chunk_idx].strip().split('\t')

            elapsed_time0 = np.round(float(line2[-1]) - float(line0[-2]), 2)
            elapsed_time1 = line0[1]
            elapsed_time2 = line1[1]
            elapsed_time3 = line2[1]
            log += '\t{}\t{}\t{}\t{}\n'.format(elapsed_time0, elapsed_time1, elapsed_time2, elapsed_time3)

        return log, num_anchor_points

    def summary_baseline0(self):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log = None

        with open(os.path.join(log_dir, postfix, 'quality_{}.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                num_anchor_points = line[0]
                quality = line[1]
                quality_sr = line[2]
                quality_diff = line[3]
                quality_error_90 = line[4]
                quality_error_95 = line[5]
                quality_error_100 = line[6]

                if quality_diff < self.quality_diff:
                    break

        log = '{}\t{}\t{}\t{}\t{}\t{}'.format(num_anchor_points, quality, quality_bilinear, quality_sr, quality_diff, quality_error_90, quality_error_95, quality_error_100)
        return log

    def summary_baseline0(self, num_anchor_points):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log = None

        with open(os.path.join(log_dir, postfix, 'quality_{}.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            line = lines[num_anchor_points].strip().split('\t')
            num_anchor_points = line[0]
            quality = line[1]
            quality_sr = line[2]
            quality_diff = line[3]
            quality_error_90 = line[4]
            quality_error_95 = line[5]
            quality_error_100 = line[6]

        log = '{}\t{}\t{}\t{}\t{}\t{}'.format(num_anchor_points, quality, quality_bilinear, quality_sr, quality_diff, quality_error_90, quality_error_95, quality_error_100)
        return log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #options for libvpx
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--input_video_name', type=str, required=True)
    parser.add_argument('--compare_video_name', type=str, required=True)
    parser.add_argument('--gop', type=int, required=True)

    #options for edsr_s (DNN)
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    #options for anchor point selector
    parser.add_argument('--quality_diff', type=float, required=True)

    args = parser.parse_args()

    input_video_path = os.path.join(args.content_dir, 'video', args.input_video_name)
    compare_video_path = os.path.join(args.content_dir, 'video', args.compare_video_name)
    input_video_info = profile_video(input_video_path)
    compare_video_info = profile_video(compare_video_path)

    scale = int(compare_video_info['height'] / input_video_info['height'])
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, scale, None)

    summary = Summary(edsr_s, args.content_dir, args.input_video_name, args.compare_video_name, args.gop, args.quality_diff)
    print(summary.summary_aps_v1(0))
