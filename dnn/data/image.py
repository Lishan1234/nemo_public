import os, glob, random, sys, time, argparse
import shutil
import logging
import subprocess
import shlex
import json
import math

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Source: https://gist.githubusercontent.com/oldo/dc7ee7f28851922cca09/raw/3238ad3ad64eeacfcafe7c18e7e57d28b73cb007/video-metada-finder.py
def findVideoMetadata(video_path):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(video_path)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    # prints all the metadata available:
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(ffprobeOutput)

    # for example, find height and width
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']

    return width, height

class ImageData():
    def __init__(self, args):
        self.video_dir = args.video_dir
        self.image_dir = args.image_dir
        self.ffmpeg_path = args.ffmpeg_path
        self.video_fmt = args.video_fmt

        self.filter_frames = args.filter_frames
        self.filter_fps = args.filter_fps
        self.upsample = args.upsample

        #check ffmpeg
        if self.ffmpeg_path is None:
            logging.error('youtube-dl does not exist')
            sys.exit()

        #check ffmpeg support for libvpx
        ffmpeg_cmd = []
        ffmpeg_cmd.append(self.ffmpeg_path)
        ffmpeg_cmd.append('-buildconf')
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE)
        configuration = result.stdout.decode().split()
        map(lambda x: x.strip(), configuration)
        if '--enable-libvpx' not in configuration:
            logging.error('ffmpeg does not support libvpx')
            sys.exit()

    def ffmpeg_filter_options(self):
        if self.filter_frames == 'keyframes':
            return '-vf "select=eq(pict_type\,I)" -vsync vfr'
        elif self.filter_frames == 'uniform':
            return '-vf fps={}'.format(self.filter_fps)

    def ffmpeg_filter_rescale_option(self, width, height):
        if self.filter_frames == 'keyframes':
            return '-vf "select=eq(pict_type\,I)",scale={}:{} -vsync vfr -sws_flags {}'.format(width, height, self.upsample)
        elif self.filter_frames == 'uniform':
            return '-vf fps={},scale={}:{} -sws_flags {}'.format(self.filter_fps, width, height, self.upsample)

    def get_ffmpeg_filter(self):
        if self.filter_frames == 'keyframes':
            return '-vf "select=eq(pict_type\,I)" -vsync vfr'
        elif self.filter_frames == 'uniform':
            return '-vf fps={}'.format(self.filter_fps)

    def get_ffmpeg_filter_rescale(self, width, height):
        if self.filter_frames == 'keyframes':
            return '-vf "select=eq(pict_type\,I)",scale={}:{} -vsync vfr -sws_flags {}'.format(width, height, self.upsample)
        elif self.filter_frames == 'uniform':
            return '-vf fps={},scale={}:{} -sws_flags {}'.format(self.filter_fps, width, height, self.upsample)

    def get_postfix(self):
        if self.filter_frames == 'keyframes':
            return 'keyframes'
        elif self.filter_frames == 'uniform':
            return '{0:.2f}fps'.format(self.filter_fps)

    def execute_ffmpeg(self, video_path, image_dir, ffmpeg_filter):
        if not os.path.exists(video_path):
            logging.error('{} does not exist'.format(lr_video_path))
            sys.exit()

        if os.path.exists(image_dir):
            logging.info('{} already exists'.format(image_dir))
            return
        else:
            os.makedirs(image_dir)
            cmd = '{} -i {} {} {}/%04d.png'.format(self.ffmpeg_path, video_path, ffmpeg_filter, image_dir)
            os.system(cmd)

    def save_lr_images(self, lr_video_name):
        video_path = os.path.join(self.video_dir, '{}.{}'.format(lr_video_name, self.video_fmt))
        image_dir = os.path.join(self.image_dir, '{}_{}'.format(lr_video_name, self.get_postfix()))
        self.execute_ffmpeg(video_path, image_dir, self.get_ffmpeg_filter())

    def save_rescaled_lr_images(self, lr_video_name, hr_video_name):
        lr_video_path = os.path.join(self.video_dir, '{}.{}'.format(lr_video_name, self.video_fmt))
        lr_width, lr_height = findVideoMetadata(lr_video_path)
        hr_video_path = os.path.join(self.video_dir, '{}.{}'.format(hr_video_name, self.video_fmt))
        hr_width, hr_height = findVideoMetadata(hr_video_path)

        if hr_height % lr_height == 0 and hr_width % hr_width == 0:
            logging.info('does not require rescaled lr images')
            return
        else:
            scale = math.floor(hr_height / lr_height)
            if scale == 1:
                logging.error('scale is 1')
                return
            elif hr_width % scale != 0 or hr_height % scale != 0:
                logging.error('unsupported high resolution')
                return

        target_width = int(hr_width / scale)
        target_height = int(hr_height / scale)

        video_path = os.path.join(self.video_dir, '{}.{}'.format(lr_video_name, self.video_fmt))
        target_video_name = lr_video_name.replace(str(lr_height), str(target_height))
        image_dir = os.path.join(self.image_dir, '{}_{}'.format(target_video_name, self.get_postfix()))
        self.execute_ffmpeg(video_path, image_dir, self.get_ffmpeg_filter_rescale(target_width, target_height))

    def save_hr_images(self, hr_video_name):
        video_path = os.path.join(self.video_dir, '{}.{}'.format(hr_video_name, self.video_fmt))
        image_dir = os.path.join(self.image_dir, '{}_{}'.format(hr_video_name, self.get_postfix()))
        self.execute_ffmpeg(video_path, image_dir, self.get_ffmpeg_filter())

    def save_all(self, lr_video_name, hr_video_name):
        self.save_lr_images(lr_video_name)
        self.save_hr_images(hr_video_name)
        self.save_rescaled_lr_images(lr_video_name, hr_video_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--video_fmt', type=str, required=True)

    parser.add_argument('--filter_frames', type=str, choices=['uniform', 'keyframes',], required=True)
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')
    args = parser.parse_args()

    imagedata = ImageData(args)
    imagedata.save_all(args.lr_video_name, args.hr_video_name)
