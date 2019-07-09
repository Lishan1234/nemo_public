import os, glob, random, sys, time, argparse
import utility as util
import re
import shutil

from config import *

import numpy as np
from scipy.misc import imsave
from PIL import Image
import ntpath

#Info: Use video_list for debugging

parser = argparse.ArgumentParser(description="Video dataset")

#data
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--data_dir', type=str, default="./data")
parser.add_argument('--target_resolution', type=int, default=1080)
parser.add_argument('--scale', type=int, required=True)
parser.add_argument('--video_len', type=int, default=60)
parser.add_argument('--video_start', type=int, required=True)
parser.add_argument('--lq_dnn', type=str, required=True)
parser.add_argument('--hq_dnn', type=str, required=True)
parser.add_argument('--device_id', type=str, default=None)

#eval
parser.add_argument('--start_idx', type=int, default=None)
parser.add_argument('--end_idx', type=int, default=None)
parser.add_argument('--cache_only', action='store_true')

args = parser.parse_args()

assert (args.start_idx is None and args.end_idx is None) or (args.start_idx is not None and args.end_idx is not None)

#0. setup
dst_debug_dir = os.path.join(args.data_dir, args.dataset, 'debug')
os.makedirs(dst_debug_dir, exist_ok=True)
video_dir = os.path.join(args.data_dir, args.dataset, 'video')
video_list_path = os.path.join(video_dir, '{}p_{}p_{}sec_{}st'.format(args.target_resolution, args.target_resolution // args.scale, args.video_len, args.video_start), args.hq_dnn, args.lq_dnn, 'video_list')
#video_list_path = os.path.join(video_dir, 'video_list')

assert os.path.isfile(video_list_path)
with open(video_list_path, 'r') as f:
    video_list = f.readlines()

lr_name = video_list[1].rstrip('\r\n')
lr_cache_name = 'lr_cache_{}'.format(video_list[3].rstrip('\r\n'))
hr_cache_name = 'hr_cache_{}'.format(video_list[3].rstrip('\r\n'))
hr_name = video_list[0].rstrip('\r\n')

lr_dir = os.path.join(dst_debug_dir, lr_name)
os.makedirs(lr_dir, exist_ok=True)
lr_cache_dir = os.path.join(dst_debug_dir, lr_cache_name)
os.system('rm -rf {}'.format(os.path.abspath(lr_cache_dir)))
os.makedirs(lr_cache_dir, exist_ok=True)
hr_cache_dir = os.path.join(dst_debug_dir, hr_cache_name)
os.system('rm -rf {}'.format(os.path.abspath(hr_cache_dir)))
os.makedirs(hr_cache_dir, exist_ok=True)
hr_dir = os.path.join(dst_debug_dir, hr_name)
os.makedirs(hr_dir, exist_ok=True)

lr_h = int(video_list[3].split('_')[1].split('p')[0])
hr_h = int(video_list[3].split('_')[0].split('p')[0])
lr_w = WIDTH[lr_h]
hr_w = WIDTH[hr_h]

if args.device_id is None:
    adb_cmd = 'adb'
else:
    adb_cmd = 'adb -s {}'.format(args.device_id)

#1. download frames
src_frame_dir = '/storage/emulated/0/Android/data/android.example.testlibvpx/files/mobinas/{}/frame'.format(args.dataset)
dst_frame_dir = os.path.join(dst_debug_dir, 'frame')
os.system('rm -rf {}'.format(dst_frame_dir))
os.makedirs(dst_frame_dir, exist_ok=True)

if args.start_idx is None and args.end_idx is None:
    if not args.cache_only:
        cmd = '{} shell find {}/ -name "*{}p_*k*" -print0 | xargs -0 -i {} pull {{}} {}'.format(adb_cmd, src_frame_dir, args.target_resolution // args.scale, adb_cmd, dst_frame_dir)
        os.system(cmd)
        cmd = '{} shell find {}/ -name "*{}p_lossless*" -print0 | xargs -0 -i {} pull {{}} {}'.format(adb_cmd, src_frame_dir, args.target_resolution, adb_cmd, dst_frame_dir)
        os.system(cmd)
    cmd = '{} shell find {}/ -name "*lr_cache*" -print0 | xargs -0 -i {} pull {{}} {}'.format(adb_cmd, src_frame_dir, adb_cmd, dst_frame_dir)
    os.system(cmd)
    cmd = '{} shell find {}/ -name "*hr_cache*" -print0 | xargs -0 -i {} pull {{}} {}'.format(adb_cmd, src_frame_dir, adb_cmd, dst_frame_dir)
    os.system(cmd)
else:
    raise NotImplementedError
    #TODO: download intermediate frames
    lr_postfix = video_list[1].rstrip('\r\n')
    for idx in range(args.start_idx, args.end_idx + 1):
        if not args.cache_only:
            cmd = '{} pull {}/{}_{}.y {}'.format(adb_cmd, src_frame_dir, idx, lr_name, dst_frame_dir)
            os.system(cmd)
            cmd = '{} pull {}/{}_{}.y {}'.format(adb_cmd, src_frame_dir, idx, hr_name, dst_frame_dir)
            os.system(cmd)
        cmd = '{} pull {}/{}_{}.y {}'.format(adb_cmd, src_frame_dir, idx, lr_cache_name, dst_frame_dir)
        os.system(cmd)
        cmd = '{} pull {}/{}_{}.y {}'.format(adb_cmd, src_frame_dir, idx, hr_cache_name, dst_frame_dir)
        os.system(cmd)

lr_image_filenames = sorted(glob.glob('{}/*_{}.y'.format(dst_frame_dir, lr_name)))
lr_cache_image_filenames = sorted(glob.glob('{}/*_{}.y'.format(dst_frame_dir, lr_cache_name)))
hr_cache_image_filenames = sorted(glob.glob('{}/*_{}.y'.format(dst_frame_dir, hr_cache_name)))
hr_image_filenames = sorted(glob.glob('{}/*_{}.y'.format(dst_frame_dir, hr_name)))
print(hr_cache_image_filenames)

print("hr size: {}, hr_cache: {}, lr cache size: {} lr size: {}".format(len(hr_image_filenames), len(hr_cache_image_filenames), len(lr_cache_image_filenames), len(lr_image_filenames)))

if not args.cache_only:
    for idx, image_filename in enumerate(lr_image_filenames):
        image_raw = np.fromfile(image_filename, dtype=np.uint8)
        image_raw.shape = (lr_h, lr_w)
        image_reshape = Image.fromarray(image_raw)
        image_reshape.save("{}/{}.png".format(lr_dir, ntpath.basename(image_filename).split('.')[0]))

    for idx, image_filename in enumerate(hr_image_filenames):
        image_raw = np.fromfile(image_filename, dtype=np.uint8)
        image_raw.shape = (hr_h, hr_w)
        image_reshape = Image.fromarray(image_raw)
        image_reshape.save("{}/{}.png".format(hr_dir, ntpath.basename(image_filename).split('.')[0]))

for idx, image_filename in enumerate(lr_cache_image_filenames):
    image_raw = np.fromfile(image_filename, dtype=np.uint8)
    image_raw.shape = (lr_h, lr_w)
    image_reshape = Image.fromarray(image_raw)
    image_reshape.save("{}/{}.png".format(lr_cache_dir, ntpath.basename(image_filename).split('.')[0]))

for idx, image_filename in enumerate(hr_cache_image_filenames):
    image_raw = np.fromfile(image_filename, dtype=np.uint8)
    print(image_filename)
    image_raw.shape = (hr_h, hr_w)
    image_reshape = Image.fromarray(image_raw)
    image_reshape.save("{}/{}.png".format(hr_cache_dir, ntpath.basename(image_filename).split('.')[0]))
