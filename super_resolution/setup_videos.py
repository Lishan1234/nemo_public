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
parser.add_argument('--video_start', type=int, default=0)
parser.add_argument('--target_resolution', type=int, default=1080) #target HR resolution
parser.add_argument('--original_resolution', default=2160) #original HR resolution - raw source
parser.add_argument('--video_format', type=str, default="webm")
parser.add_argument('--scale', type=int, required=True)
parser.add_argument('--device_id', type=str, default=None)

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

sr_regex = "^{}p_{}p_{}sec_{}st".format(args.target_resolution, args.target_resolution // args.scale, args.video_len, args.video_start)
sr_pattern = re.compile(sr_regex)

#filter videos
for video_file in video_files:
    if hr_pattern.match(video_file): filtered_video_files.append(video_file)
    if lr_pattern.match(video_file): filtered_video_files.append(video_file)
    if lr_bicubic_pattern.match(video_file): filtered_video_files.append(video_file)
    if sr_pattern.match(video_file): filtered_video_files.append(video_file)
print(filtered_video_files)

#create a root dir if not existing
src_data_dir = os.path.join(args.data_dir, args.dataset, args.dataset)
dst_data_root = '/storage/emulated/0/Android/data/android.example.testlibvpx/files/mobinas'
dst_data_dir = os.path.join(dst_data_root, args.dataset)
cmd = 'adb shell "mkdir -p {}"'.format(dst_data_root)
os.system(cmd)

#remove an existing dataset in mobile
os.path.join(dst_data_dir, args.dataset)
if args.device_id is None:
    cmd = 'adb shell "rm -rf {}"'.format(dst_data_dir)
else:
    cmd = 'adb -s {} shell "rm -rf {}"'.format(args.device_id, dst_data_dir)
os.system(cmd)

#create a data dir
#cmd = 'adb shell "mkdir -p {}"'.format(dst_data_dir)
os.system(cmd)

#copy a new dataset
for video_file in filtered_video_files:
    video_path = os.path.join(video_dir, video_file)
    if args.device_id is None:
        cmd = 'adb push {} {}'.format(video_path, dst_data_dir)
    else:
        cmd = 'adb -s {} push {} {}'.format(args.device_id, video_path, dst_data_dir)
    os.system(cmd)

#make a video file name list txt file
video_list = os.path.join(video_dir, 'video_list')
f = open(video_list, 'w')
for video_file in filtered_video_files:
    if hr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if lr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if lr_bicubic_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if sr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
f.close()

if args.device_id is None:
    cmd = 'adb push {} {}'.format(video_list, dst_data_dir)
else:
    cmd = 'adb -s {} push {} {}'.format(args.device_id, video_list, dst_data_dir)
os.system(cmd)

#TODO: check whether it works correctly

"""
#remove/create a directory
data_dir = os.path.join(args.data_dir, args.dataset, args.dataset)
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir, exist_ok=True)

#copy videos
for video_file in filtered_video_files:
    origin = os.path.join(video_dir, video_file)
    copy = os.path.join(data_dir, video_file)
    shutil.copyfile(origin, copy)

"""
