import argparse
import os
import glob
import sys
import math
from shutil import copyfile
from scipy.misc import imread
import numpy
import shlex
import subprocess
import json
import time
import re

#TODO1: support more codecs
#TODO2: support more quality metrics

parser = argparse.ArgumentParser(description='setup')
parser.add_argument('--vid_len', type=int, default=20)
parser.add_argument('--vid_format', type=str, default='webm')
parser.add_argument('--data_dir', type=str, default='../../data/starcraft1_new')
parser.add_argument('--keyframe', type=int, default=120)
args = parser.parse_args()

target_res = 1080

encoding_method = 'libx264'
container = 'mp4'
#encoding_bitrates = [600, 800, 1000, 1200, 1400, 1600]
encoding_bitrates = [1400]
#scales = [2,3,4]
scales = [2]
bitrates = {}
#bitrates[2] = [300, 500, 700, 900]
bitrates[2] = [300]
#bitrates[3] = [100, 200, 300, 400]
#bitrates[4] = [50, 100, 150, 200]

#1. encode raw frames with multiple bitrates
for bitrate in encoding_bitrates:
    #encode
    video_dir = os.path.join(args.data_dir, 'video')
    frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p'.format(target_res), 'original')
    output_name = '{}p_{}k_{}'.format(target_res, bitrate, args.vid_len)
    output_video = '{}.{}'.format(output_name, container)
    cmd = 'ffmpeg -framerate 30 -i {}/%04d.png -pix_fmt yuv420p -deinterlace -vsync 1 -threads 4 -vcodec {} -r 30 -g 120 -sc_threshold 0 -b:v {}k -bufsize {}k -maxrate {}k -preset medium -profile:v main -tune film -f {} -y {}'.format(frame_dir, encoding_method, bitrate, bitrate * 2, bitrate, container, os.path.join(video_dir, output_video))
    #os.system(cmd)

    #decode
    frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p-{}k'.format(target_res, bitrate), 'original')
    os.makedirs(frame_dir, exist_ok=True)
    cmd = 'ffmpeg -i {} {}/%04d.png'.format(os.path.join(video_dir, output_video), frame_dir)
    #os.system(cmd)

"""
#2. encode sr frames + raw frames (with multiple bitrates)
#iterate over (scale, bitrate)
#gather frames - encode - decode (input frames/video/output frames)
for hr_bitrate in encoding_bitrates:
    for scale in scales:
        for lr_bitrate in bitrates[scale]:
            encode_frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p'.format(target_res), 'sr-{}p-{}k'.format(target_res//scale, lr_bitrate))
            os.makedirs(encode_frame_dir, exist_ok=True)

            #copy keyframe (sr) frames
            key_frame_dir = os.path.join(args.data_dir, 'keyframe_{}sec'.format(args.vid_len), '{}p-{}k'.format(target_res//scale, lr_bitrate), 'sr_{}p'.format(target_res))
            key_frames = sorted(glob.glob('{}/*.png'.format(key_frame_dir)))
            for idx, key_frame in enumerate(key_frames):
                copyfile(key_frame, os.path.join(encode_frame_dir, '{:04d}.png'.format(1+(idx*args.keyframe))))

            #copy original frames
            hr_frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p'.format(target_res), 'original')
            hr_frames = sorted(glob.glob('{}/*.png'.format(hr_frame_dir)))
            for idx, hr_frame in enumerate(hr_frames):
                if idx % args.keyframe != 0:
                    copyfile(hr_frame, os.path.join(encode_frame_dir, os.path.basename(hr_frame)))

            #encode
            video_dir = os.path.join(args.data_dir, 'video')
            output_name = '{}p_{}k_{}_sr_{}p_{}k'.format(target_res, hr_bitrate, args.vid_len, target_res//scale, lr_bitrate)
            output_video = '{}.{}'.format(output_name, container)
            cmd = 'ffmpeg -framerate 30 -i {}/%04d.png -pix_fmt yuv420p -deinterlace -vsync 1 -threads 4 -vcodec {} -r 30 -g 120 -sc_threshold 0 -b:v {}k -bufsize {}k -maxrate {}k -preset medium -profile:v main -tune film -f {} -y {}'.format(encode_frame_dir, encoding_method, hr_bitrate, hr_bitrate * 2, hr_bitrate, container, os.path.join(video_dir, output_video))
            os.system(cmd)

            #decode
            decode_frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p-{}k'.format(target_res, hr_bitrate), 'sr-{}p-{}k'.format(target_res//scale, lr_bitrate))
            os.makedirs(decode_frame_dir, exist_ok=True)
            cmd = 'ffmpeg -i {} {}/%04d.png'.format(os.path.join(video_dir, output_video), decode_frame_dir)
            os.system(cmd)
"""

#3. measurement
def psnr(img1, img2, max_value=255.0):
    mse = numpy.mean((img1-img2)**2)
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(max_value/math.sqrt(mse))

#data structure
reference_frames_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p'.format(target_res), 'original')
reference_frames = sorted(glob.glob('{}/*.png'.format(reference_frames_dir)))
target_frames = {}
for hr_bitrate in encoding_bitrates:
    #original frames
    target_frames[hr_bitrate] = {}
    target_frames[hr_bitrate]['quality'] = []
    frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p-{}k'.format(target_res, hr_bitrate), 'original')
    target_frames[hr_bitrate]['frame'] = sorted(glob.glob('{}/*.png'.format(frame_dir)))

    #original + sr frames
    for scale in scales:
        target_frames[hr_bitrate][scale] = {}
        for lr_bitrate in bitrates[scale]:
            target_frames[hr_bitrate][scale][lr_bitrate] = {}
            target_frames[hr_bitrate][scale][lr_bitrate]['quality'] = []
            frame_dir = os.path.join(args.data_dir, '{}sec'.format(args.vid_len), '{}p-{}k'.format(target_res, hr_bitrate), 'sr-{}p-{}k'.format(target_res//scale, lr_bitrate))
            target_frames[hr_bitrate][scale][lr_bitrate]['frame'] = sorted(glob.glob('{}/*.png'.format(frame_dir)))

#measure quality
#for idx in range(len(reference_frames)):
for idx in range(5): #for time constraint
    start_time = time.time()
    reference_frame = imread(reference_frames[idx])
    for hr_bitrate in encoding_bitrates:
        #original frames
        target_frame = imread(target_frames[hr_bitrate]['frame'][idx])
        quality = psnr(reference_frame, target_frame)
        target_frames[hr_bitrate]['quality'].append(quality)
        #print(quality)

        #original + sr frames
        for scale in scales:
            for lr_bitrate in bitrates[scale]:
                target_frame = imread(target_frames[hr_bitrate][scale][lr_bitrate]['frame'][idx])
                quality = psnr(reference_frame, target_frame)
                target_frames[hr_bitrate][scale][lr_bitrate]['quality'].append(quality)
                #print(quality)

    #print('{}/{} done: {}sec'.format(idx, len(reference_frames), time.time()-start_time))
    print('{}/{} done: {}sec'.format(idx, 5, time.time()-start_time))

#measure bitrate
def get_video_bitrate_duration(video_path): #unit: kb/s, s
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(video_path)
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    return float(ffprobeOutput['streams'][0]['bit_rate'])/1000, float(ffprobeOutput['streams'][0]['duration'])

def get_I_frame_size(video_path): #unit: kb
    cmd = subprocess.Popen('ffprobe -show_frames {} -select_streams v | grep -E "pkt_size|pict_type"'.format(video_path), shell=True, stdout=subprocess.PIPE)

    pkt_size_patt = re.compile('pkt_size=\d+')
    pict_type_patt = re.compile('pict_type=\S')

    class Video:
        def __init__(self):
            self.frames = []

        def get_frame_info(self):
            I_frames = []
            P_frames = []
            B_frames = []

            for idx, frame in enumerate(self.frames):
                if frame.pict_type == 'I':
                    I_frames.append(frame.pkt_size)
                elif frame.pict_type == 'P':
                    P_frames.append(frame.pkt_size)
                elif frame.pict_type == 'B':
                    B_frames.append(frame.pkt_size)

            return {'I': I_frames, 'P': P_frames, 'B': B_frames}

    class Frame:
        def __init__(self, pkt_size, pict_type):
            self.pkt_size = pkt_size
            self.pict_type = pict_type

    video = Video()

    for idx, line in enumerate(cmd.stdout):
        if idx % 2 == 0:
            m = pkt_size_patt.search(str(line))
            pkt_size = int(m.group().split('=')[-1]) / 1000 * 8 #kbps
        else:
            m = pict_type_patt.search(str(line))
            pict_type = m.group().split('=')[-1]

            video.frames.append(Frame(pkt_size, pict_type))

    result = video.get_frame_info()
    I_frames, P_frames, B_frames = result['I'], result['P'], result['B']
    return I_frames

for hr_bitrate in encoding_bitrates:
    #original frames
    hr_video_path = os.path.join(args.data_dir, 'video', '{}p_{}k_{}.{}'.format(target_res, hr_bitrate, args.vid_len, container))
    target_frames[hr_bitrate]['bitrate'], duration = get_video_bitrate_duration(hr_video_path)
    hr_I_frame_size = numpy.sum(get_I_frame_size(hr_video_path))
    #print('I-frame portion: {}'.format(hr_I_frame_size / target_frames[hr_bitrate]['bitrate'] / duration * 100))

    #original + sr frames
    for scale in scales:
        for lr_bitrate in bitrates[scale]:
            hr_video_path = os.path.join(args.data_dir, 'video', '{}p_{}k_{}_sr_{}p_{}k.{}'.format(target_res, hr_bitrate, args.vid_len, target_res//scale, lr_bitrate, container))
            lr_video_path = os.path.join(args.data_dir, 'video', '{}p_{}k_{}.{}'.format(target_res//scale, lr_bitrate, args.vid_len, container))
            hr_bitrate_, duration = get_video_bitrate_duration(hr_video_path)
            lr_bitrate_, _ = get_video_bitrate_duration(lr_video_path)
            hr_I_frame_size = numpy.sum(get_I_frame_size(hr_video_path))
            lr_I_frame_size = numpy.sum(get_I_frame_size(lr_video_path))

            target_frames[hr_bitrate][scale][lr_bitrate]['bitrate'] = ((hr_bitrate_ * duration) - hr_I_frame_size + lr_I_frame_size) / duration

            #print('I-frame portion: {}'.format(lr_I_frame_size / target_frames[hr_bitrate][scale][lr_bitrate]['bitrate'] / duration * 100))

for hr_bitrate in encoding_bitrates:
    print('original-{}k: {:.4f}dB, {}kbps'.format(hr_bitrate, numpy.mean(target_frames[hr_bitrate]['quality']), target_frames[hr_bitrate]['bitrate']))
    for scale in scales:
        for lr_bitrate in bitrates[scale]:
            print('sr-{}k-{}p-{}k: {:.4f}dB, {}kbps'.format(hr_bitrate, target_res//scale, lr_bitrate, numpy.mean(target_frames[hr_bitrate][scale][lr_bitrate]['quality']), target_frames[hr_bitrate][scale][lr_bitrate]['bitrate']))

#TODO: log
