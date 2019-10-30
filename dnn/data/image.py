import os, glob, random, sys, time, argparse
import shutil
import logging
import subprocess

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--content_name', type=str,  required=True)
parser.add_argument('--ffmpeg_path', type=str, required=True)
parser.add_argument('--start_time', type=int, required=True)
parser.add_argument('--duration', type=int, required=True)
parser.add_argument('--resolutions', nargs='+', type=int, required=True)
parser.add_argument('--video_fmt', type=str, required=True)
parser.add_argument('--filter_frames', type=str, choices=['uniform', 'keyframes',], required=True)
parser.add_argument('--filter_fps', type=float, default=1.0)
args = parser.parse_args()

#path: [dataset_dir]-[content_name]-[dataset]-[image]-[240p-{}]-[%04d.png]

class ImageData():
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.content_name = args.content_name
        self.video_dir = os.path.join(self.dataset_dir, self.content_name, 'video')
        self.ffmpeg_path = args.ffmpeg_path
        self.start_time = args.start_time
        self.duration = args.duration
        self.resolutions = args.resolutions
        self.video_fmt = args.video_fmt
        self.filter_frames = args.filter_frames
        self.filter_fps = args.filter_fps

        #check videos
        for resolution in self.resolutions:
            video_path = os.path.join(self.video_dir, '{}p_s{}_d{}_encoded.{}'.format(resolution, self.start_time, self.duration, self.video_fmt))
            if not os.path.exists(video_path):
                logging.error('{} does not exist'.format(video_path))
                sys.exit()

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

    def save_keyframes(self):
        for resolution in self.resolutions:
            name = 'keyframes'
            image_dir = os.path.join(self.dataset_dir, self.content_name, 'image', '{}p-{}'.format(resolution, name))
            if os.path.exists(image_dir):
                logging.info('{} already exists'.format(image_dir))
                continue
            else:
                os.makedirs(image_dir)
            video_path = os.path.join(self.video_dir, '{}p_s{}_d{}_encoded.{}'.format(resolution, self.start_time, self.duration, self.video_fmt))
            cmd = '{} -i {} -vf "select=eq(pict_type\,I)" -vsync vfr {}/%04d.png'.format(self.ffmpeg_path, video_path, image_dir)
            os.system(cmd)

    def save_uniform(self):
        for resolution in self.resolutions:
            name = '{0:.2f}fps'.format(self.filter_fps)
            image_dir = os.path.join(self.dataset_dir, self.content_name, 'image', '{}p-{}'.format(resolution, name))
            if os.path.exists(image_dir):
                logging.info('{} already exists'.format(image_dir))
                continue
            else:
                os.makedirs(image_dir)
            video_path = os.path.join(self.video_dir, '{}p_s{}_d{}_encoded.{}'.format(resolution, self.start_time, self.duration, self.video_fmt))
            cmd = '{} -i {} -vf fps={} {}/%04d.png'.format(self.ffmpeg_path, video_path, self.filter_fps, image_dir)
            os.system(cmd)

    def save_images(self):
        print(self.filter_frames)
        if self.filter_frames == 'keyframes':
            self.save_keyframes()
        elif self.filter_frames == 'uniform':
            self.save_uniform()

if __name__ == '__main__':
    imagedata = ImageData(args)
    imagedata.save_images()
