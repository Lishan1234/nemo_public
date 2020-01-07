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
from tool.libvpx import count_mac_for_cache

class Logger():
    def __init__(self, model, content_dir, input_video, compare_video, gop, quality_diff):
        self.model = model
        self.content_dir = content_dir
        self.input_video = input_video
        self.compare_video = compare_video
        self.gop = gop
        self.quality_diff = quality_diff
        self.count_chunks = 1
        self.start_time = None

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


    def aps_v1(self, chunk_idx, quality_file, mac_file, latency_file):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log = None

        #cache quality
        with open(os.path.join(log_dir, postfix, 'quality_{}.txt'.format(APS_v1.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                num_anchor_points = int(line[0])
                quality = float(line[1])
                quality_sr = float(line[2])
                quality_diff = float(line[3])
                quality_error_90 = float(line[4])
                quality_error_95 = float(line[5])
                quality_error_100 = float(line[6])
                quality_estimate = float(line[7])

                if float(quality_diff) < self.quality_diff:
                    break

        #bilinear quality
        bilinear_psnr_values = []
        with open(os.path.join(log_dir, postfix, 'quality_bilinear.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                bilinear_psnr_value = float(line)
                bilinear_psnr_values.append(bilinear_psnr_value)
        quality_bilinear = np.average(bilinear_psnr_values)

        quality_log = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(chunk_idx, num_anchor_points, quality, quality_bilinear, quality_sr, quality_diff, quality_error_90, quality_error_95, quality_error_100, quality_estimate)
        quality_file.write(quality_log)
        quality_file.flush()

        #mac
        compare_video_path = os.path.join(args.content_dir, 'video', args.compare_video_name)
        compare_video_info = profile_video(compare_video_path)
        dnn_mac = self.model.count_mac()
        assert(dnn_mac is not None)
        cache_mac = count_mac_for_cache(compare_video_info['height'], compare_video_info['width'], 3)
        total_dnn_mac = num_anchor_points * dnn_mac
        total_cache_mac = (self.gop - num_anchor_points) * cache_mac
        total_mac = total_dnn_mac
        baseline_mac = dnn_mac * self.gop
        normalized_mac = (total_mac / baseline_mac)

        mac_log = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}\n'.format(chunk_idx, dnn_mac, cache_mac, total_dnn_mac, total_cache_mac, total_mac, baseline_mac, normalized_mac)
        mac_file.write(mac_log)
        mac_file.flush()

        #latency
        with open(os.path.join(log_dir, 'latency_{}_0.txt'.format(APS_v1.__name__)), 'r') as f0, \
             open(os.path.join(log_dir, 'latency_{}_1.txt'.format(APS_v1.__name__)), 'r') as f1, \
             open(os.path.join(log_dir, 'latency_{}_2.txt'.format(APS_v1.__name__)), 'r') as f2:
            line0 = f0.readlines()[chunk_idx].strip().split('\t')
            line1 = f1.readlines()[chunk_idx].strip().split('\t')
            line2 = f2.readlines()[chunk_idx].strip().split('\t')

            elapsed_time0 = float(line2[-1]) - float(line0[-2])
            if self.start_time is None:
                elapsed_time1 = elapsed_time0
                self.start_time = float(line0[-2])
            else:
                elapsed_time1 = float(line2[-1]) - self.start_time
            throughput = self.count_chunks / elapsed_time1
            elapsed_time2 = float(line0[1])
            elapsed_time3 = float(line1[1])
            elapsed_time4 = float(line2[1])

        latency_log = '{}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(chunk_idx, elapsed_time0, elapsed_time1, throughput, elapsed_time2, elapsed_time3, elapsed_time4)
        latency_file.write(latency_log)
        latency_file.flush()

        return num_anchor_points

    def aps_baseline_0(self, chunk_idx, quality_file):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log = None

        #cache quality
        with open(os.path.join(log_dir, postfix, 'quality_{}.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                num_anchor_points = int(line[0])
                quality = float(line[1])
                quality_sr = float(line[2])
                quality_diff = float(line[3])
                quality_error_90 = float(line[4])
                quality_error_95 = float(line[5])
                quality_error_100 = float(line[6])

                if float(quality_diff) < self.quality_diff:
                    break

        #bilinear quality
        bilinear_psnr_values = []
        with open(os.path.join(log_dir, postfix, 'quality_bilinear.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                bilinear_psnr_value = float(line)
                bilinear_psnr_values.append(bilinear_psnr_value)
        quality_bilinear = np.average(bilinear_psnr_values)

        quality_log = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(chunk_idx, num_anchor_points, quality, quality_bilinear, quality_sr, quality_diff, quality_error_90, quality_error_95, quality_error_100)
        quality_file.write(quality_log)
        quality_file.flush()

    def aps_baseline_1(self, chunk_idx, quality_file, num_anchor_points):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        log = None

        #cache quality
        with open(os.path.join(log_dir, postfix, 'quality_{}.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            line = lines[num_anchor_points - 1].strip().split('\t')
            num_anchor_points = int(line[0])
            quality = float(line[1])
            quality_sr = float(line[2])
            quality_diff = float(line[3])
            quality_error_90 = float(line[4])
            quality_error_95 = float(line[5])
            quality_error_100 = float(line[6])

        #bilinear quality
        bilinear_psnr_values = []
        with open(os.path.join(log_dir, postfix, 'quality_bilinear.txt'.format(APS_Baseline.__name__)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                bilinear_psnr_value = float(line)
                bilinear_psnr_values.append(bilinear_psnr_value)
        quality_bilinear = np.average(bilinear_psnr_values)

        quality_log = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(chunk_idx, num_anchor_points, quality, quality_bilinear, quality_sr, quality_diff, quality_error_90, quality_error_95, quality_error_100)
        quality_file.write(quality_log)

    def run(self, chunk_idx):
        log_dir = os.path.join(self.content_dir, 'log', self.input_video, self.model.name)
        f0 = open(os.path.join(log_dir, 'quality_{}.txt'.format(APS_v1.__name__)), 'w')
        f1 = open(os.path.join(log_dir, 'mac_{}.txt'.format(APS_v1.__name__)), 'w')
        f2 = open(os.path.join(log_dir, 'latency_{}.txt'.format(APS_v1.__name__)), 'w')
        f3 = open(os.path.join(log_dir, 'quality_{}_0.txt'.format(APS_Baseline.__name__)), 'w')
        f4 = open(os.path.join(log_dir, 'quality_{}_1.txt'.format(APS_Baseline.__name__)), 'w')

        if chunk_idx is None:
            input_video_path = os.path.join(self.content_dir, 'video', self.input_video)
            input_video_info = profile_video(input_video_path)
            num_chunks = int(input_video_info['duration'] // (self.gop / input_video_info['frame_rate']))
            for i in range(num_chunks):
                num_anchor_points = self.aps_v1(i, f0, f1, f2)
                self.aps_baseline_0(i, f3)
                self.aps_baseline_1(i, f4, num_anchor_points)
        else:
            num_anchor_points = self.aps_v1(chunk_idx, f0, f1, f2)
            self.aps_baseline_0(chunk_idx, f3)
            self.aps_baseline_1(chunk_idx, f4, num_anchor_points)

        f0.close()
        f1.close()
        f2.close()
        f3.close()
        f4.close()

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
    parser.add_argument('--chunk_idx', type=int, default=None)
    parser.add_argument('--quality_diff', type=float, required=True)

    args = parser.parse_args()

    input_video_path = os.path.join(args.content_dir, 'video', args.input_video_name)
    compare_video_path = os.path.join(args.content_dir, 'video', args.compare_video_name)
    input_video_info = profile_video(input_video_path)
    compare_video_info = profile_video(compare_video_path)

    scale = int(compare_video_info['height'] / input_video_info['height'])
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, scale, None)

    logger = Logger(edsr_s, args.content_dir, args.input_video_name, args.compare_video_name, args.gop, args.quality_diff)
    logger.run(args.chunk_idx)
