import os, glob, random, sys, time, argparse
import re
import shutil

import utility as util

import tensorflow as tf
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Video dataset")

parser.add_argument('--data_dir', type=str, default="./data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--video_len', type=int, default=60)
parser.add_argument('--video_start', type=int, required=True)
parser.add_argument('--target_resolution', type=int, default=1080) #target HR resolution
parser.add_argument('--original_resolution', default=2160) #original HR resolution - raw source
parser.add_argument('--video_format', type=str, default="webm")
parser.add_argument('--scale', type=int, required=True)
parser.add_argument('--device_id', type=str, default=None)
parser.add_argument('--lq_dnn', type=str, required=True)
parser.add_argument('--hq_dnn', type=str, required=True)

args = parser.parse_args()

video_dir = os.path.join(args.data_dir, args.dataset, "video")
video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
filtered_video_files = []

hr_regex = "^{}p_lossless_{}sec_{}st".format(args.target_resolution, args.video_len, args.video_start)
hr_pattern = re.compile(hr_regex)

lr_regex = "^{}p_\d*k_{}sec_{}st".format(args.target_resolution // args.scale, args.video_len, args.video_start)
lr_pattern = re.compile(lr_regex)

lr_bicubic_regex = "^{}p_{}p_\d*k_{}sec_{}st".format(args.target_resolution, args.target_resolution // args.scale, args.video_len, args.video_start)
lr_bicubic_pattern = re.compile(lr_bicubic_regex)

lq_dnn_regex = "^{}p_{}p_{}sec_{}st_{}".format(args.target_resolution, args.target_resolution // args.scale, args.video_len, args.video_start, args.lq_dnn)
lq_dnn_pattern = re.compile(lq_dnn_regex)

hq_dnn_regex = "^{}p_{}p_{}sec_{}st_{}".format(args.target_resolution, args.target_resolution // args.scale, args.video_len, args.video_start, args.hq_dnn)
hq_dnn_pattern = re.compile(hq_dnn_regex)

#sr_regex = "^{}p_{}p_{}sec_{}st".format(args.target_resolution, args.target_resolution // args.scale, args.video_len, args.video_start)
#sr_pattern = re.compile(sr_regex)

print('video_files: {}'.format(video_files))
#filter videos
for video_file in video_files:
    if hr_pattern.match(video_file): filtered_video_files.append(video_file)
    if lr_pattern.match(video_file): filtered_video_files.append(video_file)
    if lr_bicubic_pattern.match(video_file): filtered_video_files.append(video_file)
    if hq_dnn_pattern.match(video_file): filtered_video_files.append(video_file)
    if lq_dnn_pattern.match(video_file): filtered_video_files.append(video_file)
    #if sr_pattern.match(video_file): filtered_video_files.append(video_file)
print('filtered_video_files: {}'.format(filtered_video_files))
assert len(filtered_video_files) == 5

#create a root dir if not existing
src_data_dir = os.path.join(args.data_dir, args.dataset, args.dataset)
dst_data_root = '/storage/emulated/0/Android/data/android.example.testlibvpx/files/mobinas'
dst_data_dir = os.path.join(dst_data_root, args.dataset)
dst_video_dir = os.path.join(dst_data_dir, 'video')
cmd = 'adb shell "mkdir -p {}"'.format(dst_data_root)
os.system(cmd)
cmd = 'adb shell "mkdir -p {}"'.format(dst_video_dir)
os.system(cmd)

#copy a new dataset
for video_file in filtered_video_files:
    video_path = os.path.join(video_dir, video_file)
    if args.device_id is None:
        cmd = 'adb push {} {}/'.format(video_path, dst_video_dir)
    else:
        cmd = 'adb -s {} push {} {}/'.format(args.device_id, video_path, dst_video_dir)
    os.system(cmd)

#make a video file name list txt file
video_list_dir = os.path.join(video_dir, '{}p_{}p_{}sec_{}st'.format(args.target_resolution, args.target_resolution // args.scale, args.video_len, args.video_start), args.hq_dnn, args.lq_dnn)
os.makedirs(video_list_dir, exist_ok=True)
video_list_path = os.path.join(video_list_dir, 'video_list')
f = open(video_list_path, 'w')
for video_file in filtered_video_files:
    if hr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if lr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if lr_bicubic_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if hq_dnn_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if lq_dnn_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
f.close()

#copy a video list
if args.device_id is None:
    cmd = 'adb push {} {}/'.format(video_list_path, dst_video_dir)
else:
    cmd = 'adb -s {} push {} {}/'.format(args.device_id, video_list_path, dst_video_dir)
os.system(cmd)
