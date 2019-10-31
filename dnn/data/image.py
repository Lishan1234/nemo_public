import os, glob, random, sys, time, argparse
import shutil
import logging
import subprocess
import shlex
import json
import math

#path: [dataset_dir]-[content_name]-[dataset]-[image]-[240p-{}]-[%04d.png]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--content_name', type=str,  required=True)
parser.add_argument('--start_time', type=int, required=True)
parser.add_argument('--duration', type=int, required=True)
parser.add_argument('--resolution_pairs', nargs='+', type=str, required=True)
parser.add_argument('--video_fmt', type=str, required=True)
parser.add_argument('--ffmpeg_path', type=str, required=True)
parser.add_argument('--filter_frames', type=str, choices=['uniform', 'keyframes',], required=True)
parser.add_argument('--filter_fps', type=float, default=1.0)
parser.add_argument('--upsample', type=str, default='bilinear')
args = parser.parse_args()

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

class SRConfig():
    def __init__(self, input_width, input_height, output_width, output_height):
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height

    def __str__(self):
        return "input_width: {}, input_height: {}, output_width: {}, output_height{}".format(self.input_width, self.input_height, self.output_width, self.output_height)

    def require_preprocess(self):
        if self.output_height % self.input_height != 0:
            return True
        else:
            return False

class ImageData():
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.content_name = args.content_name
        self.video_dir = os.path.join(self.dataset_dir, self.content_name, 'video')
        self.ffmpeg_path = args.ffmpeg_path
        self.start_time = args.start_time
        self.duration = args.duration
        self.video_fmt = args.video_fmt
        self.filter_frames = args.filter_frames
        self.filter_fps = args.filter_fps
        self.upsample = args.upsample

        #check videos & find metadata (width, height)
        self.sr_configs = []
        for resolution_pair in args.resolution_pairs:
            input_resolution = int(resolution_pair.split(',')[0])
            output_resolution = int(resolution_pair.split(',')[1])

            input_video_path = os.path.join(self.video_dir, '{}p_s{}_d{}_encoded.{}'.format(input_resolution, self.start_time, self.duration, self.video_fmt))
            if not os.path.exists(input_video_path):
                logging.error('{} does not exist'.format(video_path))
                sys.exit()

            output_video_path = os.path.join(self.video_dir, '{}p_s{}_d{}_encoded.{}'.format(output_resolution, self.start_time, self.duration, self.video_fmt))
            if not os.path.exists(output_video_path):
                logging.error('{} does not exist'.format(video_path))
                sys.exit()

            input_width, input_height = findVideoMetadata(input_video_path)
            output_width, output_height = findVideoMetadata(output_video_path)
            sr_config = SRConfig(input_width, input_height, output_width, output_height)
            self.sr_configs.append(sr_config)
            logging.info(sr_config)

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
            return 'eyframes'
        elif self.filter_frames == 'uniform':
            return '{0:.2f}fps'.format(self.filter_fps)

    def execute_ffmpeg(self, video_path, image_dir, ffmpeg_filter):
        if os.path.exists(image_dir):
            logging.info('{} already exists'.format(image_dir))
            return
        else:
            os.makedirs(image_dir)
            cmd = '{} -i {} {} {}/%04d.png'.format(self.ffmpeg_path, video_path, ffmpeg_filter, image_dir)
            os.system(cmd)

    def save(self):
        for sr_config in self.sr_configs:
            #input (encoded) frames
            video_name = '{}p_s{}_d{}_encoded'.format(sr_config.input_height, self.start_time, self.duration)
            video_path = os.path.join(self.video_dir, '{}.{}'.format(video_name, self.video_fmt))
            image_dir = os.path.join(self.dataset_dir, self.content_name, 'image', '{}_{}'.format(video_name, self.get_postfix()))
            self.execute_ffmpeg(video_path, image_dir, self.get_ffmpeg_filter())

            #output (raw) frames
            video_name = '{}p_s{}_d{}'.format(sr_config.output_height, self.start_time, self.duration)
            video_path = os.path.join(self.video_dir, '{}.{}'.format(video_name, self.video_fmt))
            image_dir = os.path.join(self.dataset_dir, self.content_name, 'image', '{}_{}'.format(video_name, self.get_postfix()))
            self.execute_ffmpeg(video_path, image_dir, self.get_ffmpeg_filter())

            #preprocess (encoded) frames
            if sr_config.require_preprocess():
                scale = math.floor(sr_config.output_height / sr_config.input_height)
                assert sr_config.output_height % scale == 0 and sr_config.output_height % scale == 0 and scale != 1
                target_width = int(sr_config.output_width / scale)
                target_height = int(sr_config.output_height / scale)
                video_name = '{}p_s{}_d{}_encoded'.format(sr_config.input_height, self.start_time, self.duration)
                video_path = os.path.join(self.video_dir, '{}.{}'.format(video_name, self.video_fmt))
                image_dir = os.path.join(self.dataset_dir, self.content_name, 'image', '{}p_s{}_d{}_{}'.format(target_height, self.start_time, self.duration, self.get_postfix()))
                self.execute_ffmpeg(video_path, image_dir, self.get_ffmpeg_filter_rescale(target_width, target_height))

if __name__ == '__main__':
    imagedata = ImageData(args)
    imagedata.save()
