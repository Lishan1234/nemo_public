import argparse
import os
import glob
import subprocess
import re
import sys
from shutil import copyfile

parser = argparse.ArgumentParser(description='setup')
parser.add_argument('--vid_len', type=int, default=20)
parser.add_argument('--vid_format', type=str, default='webm')
parser.add_argument('--data_dir', type=str, default='../../data/starcraft1_new')
parser.add_argument('--keyframe', type=int, default=120)
parser.add_argument('--scale', type=int, default=3)
parser.add_argument('--bitrate', type=str, required=True)
args = parser.parse_args()

source_res = 2160
target_hr = 1080
target_lr_bitrate = list(map(lambda x: int(x), args.bitrate.split(',')))
frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len))

#1. create direcetories
os.makedirs(os.path.join(frame_dir, '{}p-original'.format(source_res), 'original'), exist_ok=True)
os.makedirs(os.path.join(frame_dir, '{}p'.format(source_res), 'original'), exist_ok=True)
os.makedirs(os.path.join(frame_dir, '{}p'.format(target_hr), 'original'), exist_ok=True)
for bitrate in target_lr_bitrate:
    os.makedirs(os.path.join(frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'original'), exist_ok=True)
    os.makedirs(os.path.join(frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'bicubic_{}p'.format(target_hr)), exist_ok=True)

key_frame_dir = os.path.join(args.data_dir, 'keyframe_{}sec'.format(args.vid_len))
os.makedirs(os.path.join(key_frame_dir, '{}p'.format(source_res), 'original'), exist_ok=True)
os.makedirs(os.path.join(key_frame_dir, '{}p'.format(target_hr), 'original'), exist_ok=True)
for bitrate in target_lr_bitrate:
    os.makedirs(os.path.join(key_frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'original'), exist_ok=True)
    os.makedirs(os.path.join(key_frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'bicubic_{}p'.format(target_hr)), exist_ok=True)

#2. create dataset with all frames / keyframes
#2-1. decode 2160p images
vid_path = os.path.join(args.data_dir, 'video', '{}p_{}.{}'.format(source_res, args.vid_len, args.vid_format))
assert os.path.isfile(vid_path)
os.system('ffmpeg -i {} {}/%04d.png'.format(vid_path, os.path.join(frame_dir, '{}p-original'.format(source_res), 'original')))

#2-2. encode 2160p images and decode into 2160p/1080p images
#2160p, 1080p
method = 'libx264'
container = 'mp4'
source_dir = os.path.join(frame_dir, '{}p-original'.format(source_res), 'original')
source_bitrate = 40000
output_path = os.path.join(args.data_dir, 'video', '{}p_{}k_{}.{}'.format(source_res, source_bitrate, args.vid_len, container))
os.system('''ffmpeg -i {}/%04d.png -pix_fmt yuv420p -deinterlace -vsync 1 -threads 4 -vcodec {} -r 30 -g 120 -sc_threshold 0 -b:v {}k -bufsize {}k -maxrate {}k -preset medium -profile:v main -tune film -f {} -y {}'''.format(source_dir, method, source_bitrate, source_bitrate * 2, source_bitrate, container, output_path))
os.system('ffmpeg -i {} {}/%04d.png'.format(output_path, os.path.join(frame_dir, '{}p'.format(source_res), 'original')))
os.system('ffmpeg -i {}/%04d.png -vf scale=-1:{} {}/%04d.png'.format(os.path.join(frame_dir, '{}p'.format(source_res), 'original'), target_hr, os.path.join(frame_dir, '{}p'.format(target_hr), 'original')))

#2-3. encode 2160p images into 540p/360p/270p videos and extract frames
#540p/360p/270p
method = 'libx264'
container = 'mp4'
source_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p-original'.format(source_res), 'original')
for bitrate in target_lr_bitrate:
    output_path = os.path.join(args.data_dir, 'video', '{}p_{}k_{}.{}'.format(target_hr//args.scale, bitrate, args.vid_len, container))
    os.system('''ffmpeg -i {}/%04d.png -pix_fmt yuv420p -deinterlace -vf 'scale=-1:{}' -vsync 1 -threads 4 -vcodec {} -r 30 -g 120 -sc_threshold 0 -b:v {}k -bufsize {}k -maxrate {}k -preset medium -profile:v main -tune film -f {} -y {}'''.format(source_dir, target_hr//args.scale, method, bitrate, bitrate * 2, bitrate, container, output_path))
    os.system('ffmpeg -i {} {}/%04d.png'.format(output_path, os.path.join(frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'original')))
    os.system('ffmpeg -i {}/%04d.png -vf scale=-1:{} {}/%04d.png'.format(os.path.join(frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'original'), target_hr, os.path.join(frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'bicubic_{}p'.format(target_hr))))

""" raw low-resolution images (old) --> extract from compressed video (new)
for lr in target_lr:
    os.system('ffmpeg -i {}/%04d.png -vf scale=-1:{} {}/%04d.png'.format(source_dir lr, os.path.join(frame_dir, '{}p'.format(lr), 'original')))
    os.system('ffmpeg -i {}/%04d.png -vf scale=-1:{} {}/%04d.png'.format(os.path.join(frame_dir, '{}p'.format(lr), 'original'), target_hr, os.path.join(frame_dir, '{}p'.format(lr), 'bicubic_{}p'.format(target_hr))))
"""

# 3.calculate key frame index
keyframe_idx = []
container = 'mp4'
output_path = os.path.join(args.data_dir, 'video', '{}p_{}k_{}.{}'.format(target_hr//args.scale, target_lr_bitrate[0], args.vid_len, container))

cmd = subprocess.Popen('ffprobe -show_frames {} -select_streams v | grep -E "pkt_size|pict_type"'.format(output_path), shell=True, stdout=subprocess.PIPE)
pkt_size_patt = re.compile('pkt_size=\d+')
pict_type_patt = re.compile('pict_type=\S')

for idx, line in enumerate(cmd.stdout):
    if idx % 2 == 1:
        m = pict_type_patt.search(str(line))
        pict_type = m.group().split('=')[-1]

        if pict_type == 'I':
            keyframe_idx.append(int((idx-1)/2))

# 4.copy keyframes
hr_frame_dir = os.path.join(frame_dir, '{}p'.format(target_hr), 'original')
hr_frames = sorted(glob.glob('{}/*.png'.format(hr_frame_dir)))
for idx, hr_frame in enumerate(hr_frames):
    if idx in keyframe_idx:
        copyfile(hr_frame, os.path.join(key_frame_dir, '{}p'.format(target_hr), 'original', os.path.basename(hr_frame)))

for bitrate in target_lr_bitrate:
    lr_frame_dir = os.path.join(frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'original')
    lr_frames = sorted(glob.glob('{}/*.png'.format(lr_frame_dir)))
    for idx, lr_frame in enumerate(lr_frames):
        if idx in keyframe_idx:
            copyfile(lr_frame, os.path.join(key_frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'original', os.path.basename(lr_frame)))

    lr_frame_dir = os.path.join(frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'bicubic_{}p'.format(target_hr))
    lr_frames = sorted(glob.glob('{}/*.png'.format(lr_frame_dir)))
    for idx, lr_frame in enumerate(lr_frames):
        if idx in keyframe_idx:
            copyfile(lr_frame, os.path.join(key_frame_dir, '{}p-{}k'.format(target_hr//args.scale, bitrate), 'bicubic_{}p'.format(target_hr), os.path.basename(lr_frame)))
