import argparse
import os
import glob
from shutil import copyfile

parser = argparse.ArgumentParser(description='setup')
parser.add_argument('--vid_len', type=int, default=20)
parser.add_argument('--data_root', type=str, default='../../data/starcraft1_new')
parser.add_argument('--vid_format', type=str, default='webm')
parser.add_argument('--keyframe', type=int, default=30)
args = parser.parse_args()

source_res = '2160'
target_hr = '1080'
target_lr = ['540', '360', '270']

# 1.create direcetories
frame_root = os.path.join(args.data_root, '{}sec'.format(args.vid_len))
key_frame_root = os.path.join(args.data_root, 'keyframe_{}sec'.format(args.vid_len))

os.makedirs(os.path.join(frame_root, '{}p'.format(source_res), 'original'), exist_ok=True)
os.makedirs(os.path.join(frame_root, '{}p'.format(target_hr), 'original'), exist_ok=True)
for lr in target_lr:
    os.makedirs(os.path.join(frame_root, '{}p'.format(lr), 'original'), exist_ok=True)
    os.makedirs(os.path.join(frame_root, '{}p'.format(lr), 'bicubic_{}p'.format(target_hr)), exist_ok=True)

os.makedirs(os.path.join(key_frame_root, '{}p'.format(source_res), 'original'), exist_ok=True)
os.makedirs(os.path.join(key_frame_root, '{}p'.format(target_hr), 'original'), exist_ok=True)
for lr in target_lr:
    os.makedirs(os.path.join(key_frame_root, '{}p'.format(lr), 'original'), exist_ok=True)
    os.makedirs(os.path.join(key_frame_root, '{}p'.format(lr), 'bicubic_{}p'.format(target_hr)), exist_ok=True)

# 2.create dataset with all frames
# 2-1.decode and save 2160p images
# 2-2.resize 2160p images to 1080p/540p/360/270p by bicubic interpolation (training/testing)
# 2-3.copy keyframes and create seperate dataset
vid_path = os.path.join(args.data_root, 'video', '{}p_{}.{}'.format(source_res, args.vid_len, args.vid_format))
assert os.path.isfile(vid_path)
os.system('ffmpeg -i {} {}/%04d.png'.format(vid_path, os.path.join(frame_root, '{}p'.format(source_res), 'original')))

os.system('ffmpeg -i {}/%04d.png -vf scale=-1:{} {}/%04d.png'.format(os.path.join(frame_root, '{}p'.format(source_res), 'original'), target_hr, os.path.join(frame_root, '{}p'.format(target_hr), 'original')))
for lr in target_lr:
    os.system('ffmpeg -i {}/%04d.png -vf scale=-1:{} {}/%04d.png'.format(os.path.join(frame_root, '{}p'.format(source_res), 'original'), lr, os.path.join(frame_root, '{}p'.format(lr), 'original')))
    os.system('ffmpeg -i {}/%04d.png -vf scale=-1:{} {}/%04d.png'.format(os.path.join(frame_root, '{}p'.format(lr), 'original'), target_hr, os.path.join(frame_root, '{}p'.format(lr), 'bicubic_{}p'.format(target_hr))))

hr_frame_root = os.path.join(frame_root, '{}p'.format(target_hr), 'original')
hr_frames = sorted(glob.glob('{}/*.png'.format(hr_frame_root)))
for idx, hr_frame in enumerate(hr_frames):
    if idx % args.keyframe == 0:
        copyfile(hr_frame, os.path.join(key_frame_root, '{}p'.format(target_hr), 'original', os.path.basename(hr_frame)))

for lr in target_lr:
    lr_frame_root = os.path.join(frame_root, '{}p'.format(lr), 'original')
    lr_frames = sorted(glob.glob('{}/*.png'.format(lr_frame_root)))
    for idx, lr_frame in enumerate(lr_frames):
        if idx % args.keyframe == 0:
            copyfile(lr_frame, os.path.join(key_frame_root, '{}p'.format(lr), 'original', os.path.basename(lr_frame)))

    lr_frame_root = os.path.join(frame_root, '{}p'.format(lr), 'bicubic_{}p'.format(target_hr))
    lr_frames = sorted(glob.glob('{}/*.png'.format(lr_frame_root)))
    for idx, lr_frame in enumerate(lr_frames):
        if idx % args.keyframe == 0:
            copyfile(lr_frame, os.path.join(key_frame_root, '{}p'.format(lr), 'bicubic_{}p'.format(target_hr), os.path.basename(lr_frame)))
