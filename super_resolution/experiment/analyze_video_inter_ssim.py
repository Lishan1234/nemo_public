import os, glob, random, sys, time, argparse
import re
import shutil

import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from skimage.measure import compare_ssim

parser = argparse.ArgumentParser(description="Video dataset")

parser.add_argument('--data_dir', type=str, default="./data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--video_len', type=int, default=60)
parser.add_argument('--video_start', type=int, required=True)
parser.add_argument('--target_resolution', type=int, default=1080) #target HR resolution
parser.add_argument('--original_resolution', default=2160) #original HR resolution - raw source
parser.add_argument('--video_format', type=str, default="webm")
parser.add_argument('--sample_fps', type=float, default=1.0)

args = parser.parse_args()

#check video exists
video_dir = os.path.join(args.data_dir, args.dataset, "video")
video_name = '{}p_lossless_{}sec_{}st'.format(args.target_resolution, args.video_len, args.video_start)
video_path = os.path.join(video_dir, '{}.webm'.format(video_name))
print(os.path.abspath(video_path))
assert os.path.isfile(video_path)

#open a video
video_cap = cv2.VideoCapture(video_path)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
assert video_cap.isOpened()
assert video_fps != 0
print('video fps: {}'.format(video_fps))

#open a logfile
log_dir = os.path.join(args.data_dir, args.dataset, 'log')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'inter_ssim_{:.2f}_{}.log'.format(args.sample_fps, video_name))
log_file = open(log_path, 'w')
assert log_file is not None

inter_ssim_list = []
interval = int(video_fps / args.sample_fps)
curr_count = 0
ssim_count = 0
prev_frame = None
prev_count = None
while (video_cap.isOpened()):
    ref, bgr_frame = video_cap.read()

    if ref==True:
        if curr_count % interval == 0:
            curr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            if prev_frame is None:
                prev_frame = curr_frame
                prev_count = curr_count
                continue

            #read a frame and measure SSIM
            print(prev_frame.shape, curr_frame.shape)
            inter_ssim = compare_ssim(prev_frame, curr_frame, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            inter_ssim_list.append(inter_ssim)
            print('[Current frame {}, Previous frame {}] inter-SSIM: {:.2f}dB'.format(curr_count, prev_count, inter_ssim))

            #save into a log file
            log = '{}\t{}\t{}\t{:.2f}\n'.format(ssim_count, prev_count, curr_count, inter_ssim)
            log_file.write(log)

            prev_frame = curr_frame
            prev_count = curr_count

            ssim_count += 1

        curr_count += 1
    else:
        break

log = 'Average\t{:.2f}dB'.format(np.average(inter_ssim_list))
log_file.write(log)
print('Average inter-SSIM: {:.2f}dB'.format(np.average(inter_ssim_list)))
log_file.close()
