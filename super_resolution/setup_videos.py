import os, glob, random, sys, time, argparse
import utility as util
import re
import shutil

import tensorflow as tf
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Video dataset")

parser.add_argument('--data_dir', type=str, default="/ssd1/data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--video_len', type=int, default=60)
parser.add_argument('--video_start', type=int, default=0)
parser.add_argument('--target_resolution', type=int, default=1080) #target HR resolution
parser.add_argument('--original_resolution', default=2160) #original HR resolution - raw source
parser.add_argument('--video_format', type=str, default="webm")
parser.add_argument('--scale', type=int, required=True)
#parser.add_argument('--lr', type=int, default=240)

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

print(hr_regex)

#filter videos
for video_file in video_files:
    #find a HR video
    if hr_pattern.match(video_file): filtered_video_files.append(video_file)
    if lr_pattern.match(video_file): filtered_video_files.append(video_file)
    if lr_bicubic_pattern.match(video_file): filtered_video_files.append(video_file)
    if sr_pattern.match(video_file): filtered_video_files.append(video_file)
   # #find a LR video
    #find a LR bicubic video
    #find a SR videos
print(filtered_video_files)

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

#make a video file name list txt file
f = open("{}/video_list".format(data_dir), 'w')
for video_file in filtered_video_files:
    if hr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if lr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if lr_bicubic_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
for video_file in filtered_video_files:
    if sr_pattern.match(video_file): f.write("{}\n".format(video_file.split(".")[0]))
f.close()

#compress videos
"""
home = os.path.expanduser('~')
compressed_path = os.path.join(home, 'MobiNAS/TestLibvpx/app/src/main/res/raw', '{}.zip'.format(args.dataset))
print(compressed_path)
cmd = 'zip -r {} {}'.format(compressed_path, data_dir)
os.system(cmd)
"""
