import sys
import os
import shutil
import logging
import subprocess
import re
import argparse
import time
import shlex

#TODO: need to test on mov format

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, required=True)
parser.add_argument('--start_time', type=int, required=True)
parser.add_argument('--num_threads', type=int, default=4)
parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')
parser.add_argument('--gop', type=str, default=120)
parser.add_argument('--duration', type=int, default='300')
parser.add_argument('--video_fmt', type=str, default='webm')
parser.add_argument('--raw_video_fmt', type=str, default='webm')

args = parser.parse_args()

class Encoder():
    def __init__(self, args):
        self.video_dir = args.video_dir
        self.ffmpeg_path = args.ffmpeg_path

        self.gop = args.gop
        self.num_threads = args.num_threads
        self.start_time = args.start_time
        self.duration= args.duration
        self.raw_video_fmt = args.raw_video_fmt
        self.video_fmt = args.video_fmt

        self.ffmpeg_cut_options = ""
        if self.start_time is not None:
            self.ffmpeg_cut_options += "-ss {}".format(self.start_time)
        if self.duration is not None:
            self.ffmpeg_cut_options += " -t {}".format(self.duration)

        #check a raw video
        raw_video_path = os.path.join(self.video_dir, "2160p.{}".format(self.raw_video_fmt))
        if not os.path.exists(raw_video_path):
            logging.error("raw video does not exist")
            sys.exit()

        #check ffmpeg
        if self.ffmpeg_path is None:
            logging.error("youtube-dl does not exist")
            sys.exit()

        #check ffmpeg support for libvpx
        ffmpeg_cmd = []
        ffmpeg_cmd.append(self.ffmpeg_path)
        ffmpeg_cmd.append("-buildconf")
        result = subprocess.check_output(ffmpeg_cmd).decode('utf-8')
        configuration = result.split()
        map(lambda x: x.strip(), configuration)
        if "--enable-libvpx" not in configuration:
            logging.error("ffmpeg does not support libvpx")
            sys.exit()

    def cut_2160p(self):
        input_video_path = os.path.join(self.video_dir, "2160p.{}".format(self.raw_video_fmt))
        output_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} {} -threads {} -c:v libvpx-vp9 -c copy {}".format(self.ffmpeg_path, input_video_path, self.ffmpeg_cut_options, self.num_threads * 8, output_video_path)
        os.system(cmd)

    def encode_1080p_lossless(self):
        input_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        output_video_path = os.path.join(self.video_dir, "1080p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        #output_video_path = os.path.join(self.video_dir, "1080p_s{}_d{}.mp4".format(self.start_time, self.duration))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            #return
        cmd = "{} -i {} -vf scale=1920x1080 -threads {} -c:v libvpx-vp9 -lossless 1 -g {} -c:a libopus {}".format(self.ffmpeg_path, input_video_path, self.num_threads * 4, self.gop, output_video_path)
        #cmd = "{} -i {} -vf scale=1920x1080 -threads {} -c:v libx265 -crf 0 -preset ultrafast -c:a libopus {}".format(self.ffmpeg_path, input_video_path, self.num_threads * 4, output_video_path)
        os.system(cmd)

    def encode_960p_v1_lossless(self):
        input_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        output_video_path = os.path.join(self.video_dir, "960p_240p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        #output_video_path = os.path.join(self.video_dir, "960p_240p_s{}_d{}.mp4".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=1704x960 -threads {} -c:v libvpx-vp9 -lossless 1 -g {} -c:a libopus {}".format(self.ffmpeg_path, input_video_path, self.num_threads * 4, self.gop, output_video_path)
        #cmd = "{} -i {} -vf scale=1704x960 -threads {} -c:v libx265 -crf 0 -preset ultrafast -c:a libopus {}".format(self.ffmpeg_path, input_video_path, self.num_threads * 4, output_video_path)
        os.system(cmd)

    def encode_960p_v2_lossless(self):
        input_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        output_video_path = os.path.join(self.video_dir, "960p_480p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        #output_video_path = os.path.join(self.video_dir, "960p_480p_s{}_d{}.mp4".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=1708x960 -threads {} -c:v libvpx-vp9 -lossless 1 -g {} -c:a libopus {}".format(self.ffmpeg_path, input_video_path, self.num_threads * 4, self.gop, output_video_path)
        #cmd = "{} -i {} -vf scale=1708x960 -threads {} -c:v libx265 -crf 0 -preset ultrafast -c:a libopus {}".format(self.ffmpeg_path, input_video_path, self.num_threads * 4, output_video_path)
        os.system(cmd)

    def encode_240p(self):
        input_video_path = os.path.join(self.video_dir, "1080p_s{}_d{}.mp4".format(self.start_time, self.duration))
        output_video_path = os.path.join(self.video_dir, "240p_s{}_d{}_encoded.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=426x240 -b:v 150k \
        -minrate 75k -maxrate 218k -tile-columns 0  -keyint_min {} -g {} -threads {} \
        -quality good -crf 37 -c:v libvpx-vp9 -c:a libopus \
        -pass 1 -speed 4 {} && \
        {} -i {} -vf scale=426x240 -b:v 150k \
        -minrate 75k -maxrate 218k -tile-columns 0  -keyint_min {} -g {} -threads {} \
        -quality good -crf 37 -c:v libvpx-vp9 -c:a libopus \
        -pass 2 -speed 1 -y {}".format(self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads, output_video_path, self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads, output_video_path)
        os.system(cmd)

    def encode_360p(self):
        input_video_path = os.path.join(self.video_dir, "1080p_s{}_d{}.mp4".format(self.start_time, self.duration))
        output_video_path = os.path.join(self.video_dir, "360p_s{}_d{}_encoded.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=640x360 -b:v 276k \
        -minrate 138k -maxrate 400k -tile-columns 1  -keyint_min {} -g {} -threads {} \
        -quality good -crf 36 -c:v libvpx-vp9 -c:a libopus \
        -pass 1 -speed 4 {} && \
        {} -i {} -vf scale=640x360 -b:v 276k \
        -minrate 138k -maxrate 400k -tile-columns 1  -keyint_min {} -g {} -threads {}\
        -quality good -crf 36 -c:v libvpx-vp9 -c:a libopus \
        -pass 2 -speed 4 -y {}".format(self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 2, output_video_path, self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads* 2, output_video_path)
        os.system(cmd)

    def encode_480p(self):
        input_video_path = os.path.join(self.video_dir, "1080p_s{}_d{}.mp4".format(self.start_time, self.duration))
        output_video_path = os.path.join(self.video_dir, "480p_s{}_d{}_encoded.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=854x480 -b:v 750k \
        -minrate 375k -maxrate 1088k -tile-columns 1  -keyint_min {} -g {} -threads {} \
        -quality good -crf 33 -c:v libvpx-vp9 -c:a libopus \
        -pass 1 -speed 4 {} && \
        {} -i {} -vf scale=854x480 -b:v 750k \
        -minrate 375k -maxrate 1088k -tile-columns 1  -keyint_min {} -g {} -threads {} \
        -quality good -crf 33 -c:v libvpx-vp9 -c:a libopus \
        -pass 2 -speed 4 -y {}".format(self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 2, output_video_path, self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 2, output_video_path)
        os.system(cmd)

    def encode_720p(self):
        input_video_path = os.path.join(self.video_dir, "1080p_s{}_d{}.mp4".format(self.start_time, self.duration))
        output_video_path = os.path.join(self.video_dir, "720p_s{}_d{}_encoded.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=1280x720 -b:v 1024k \
        -minrate 512k -maxrate 1485k -tile-columns 2  -keyint_min {} -g {} -threads {} \
        -quality good -crf 32 -c:v libvpx-vp9 -c:a libopus \
        -pass 1 -speed 4 {} && \
        {} -i {} -vf scale=1280x720 -b:v 1024k \
        -minrate 512k -maxrate 1485k -tile-columns 2  -keyint_min {} -g {} -threads {} \
        -quality good -crf 32 -c:v libvpx-vp9 -c:a libopus \
        -pass 2 -speed 4 -y {}".format(self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 4, output_video_path, self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 4, output_video_path)
        os.system(cmd)

    def encode_1080p(self):
        input_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        output_video_path = os.path.join(self.video_dir, "1080p_s{}_d{}_encoded.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=1920x1080 -b:v 1800k \
        -minrate 900k -maxrate 2610k -tile-columns 2  -keyint_min {} -g {} -threads {} \
        -quality good -crf 31 -c:v libvpx-vp9 -c:a libopus \
        -pass 1 -speed 4 {} && \
        {} -i {} -vf scale=1920x1080 -b:v 1800k \
        -minrate 900k -maxrate 2610k -tile-columns 2  -keyint_min {} -g {} -threads {} \
        -quality good -crf 31 -c:v libvpx-vp9 -c:a libopus \
         -pass 2 -speed 4 -y {}".format(self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 4, output_video_path, self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 4, output_video_path)
        os.system(cmd)

    def encode_1440p(self):
        input_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        output_video_path = os.path.join(self.video_dir, "1440p_s{}_d{}_encoded.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=2560x1440 -b:v 6000k \
        -minrate 3000k -maxrate 8700k -tile-columns 3  -keyint_min {} -g {} -threads {} \
        -quality good -crf 24 -c:v libvpx-vp9 -c:a libopus \
        -pass 1 -speed 4 {} && \
        {} -i {} -vf scale=2560x1440 -b:v 6000k \
        -minrate 3000k -maxrate 8700k -tile-columns 3  -keyint_min {} -g {} -threads {} \
        -quality good -crf 24 -c:v libvpx-vp9 -c:a libopus \
        -pass 2 -speed 4 -y {}".format(self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 8, output_video_path, self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 8, output_video_path)
        os.system(cmd)

    def encode_2160p(self):
        input_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}.{}".format(self.start_time, self.duration, self.video_fmt))
        output_video_path = os.path.join(self.video_dir, "2160p_s{}_d{}_encoded.{}".format(self.start_time, self.duration, self.video_fmt))

        if os.path.exists(output_video_path):
            logging.info("{} already exists".format(output_video_path))
            return

        cmd = "{} -i {} -vf scale=3840x2160 -b:v 12000k \
        -minrate 6000k -maxrate 17400k -tile-columns 3  -keyint_min {} -g {} -threads {} \
        -quality good -crf 15 -c:v libvpx-vp9 -c:a libopus \
        -pass 1 -speed 4 {} && \
        {} -i {} -vf scale=3840x2160 -b:v 12000k \
        -minrate 6000k -maxrate 17400k -tile-columns 3  -keyint_min {} -g {} -threads {} \
        -quality good -crf 15 -c:v libvpx-vp9 -c:a libopus \
        -pass 2 -speed 4 -y {}".format(self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 8, output_video_path, self.ffmpeg_path, input_video_path, self.gop, self.gop, self.num_threads * 8, output_video_path)
        os.system(cmd)

    def encode_all(self):
        start_time = time.time()
        self.encode_1080p_lossless()
        """
        self.cut_2160p()
        print('elaspsed_time (1/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_1080p_lossless()
        print('elaspsed_time (2/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_960p_v1_lossless()
        print('elaspsed_time (3/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_960p_v2_lossless()
        print('elaspsed_time (4/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_240p()
        print('elaspsed_time (5/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_360p()
        print('elaspsed_time (6/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_480p()
        print('elaspsed_time (7/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_720p()
        print('elaspsed_time (8/9): {}sec'.format(time.time() - start_time))
        start_time = time.time()
        self.encode_1080p()
        print('elaspsed_time (9/9): {}sec'.format(time.time() - start_time))
        #self.encode_1440p()
        #print('elaspsed_time (10/11): {}sec'.format(time.time() - start_time))
        #self.encode_2160p()
        #print('elaspsed_time (11/11): {}sec'.format(time.time() - start_time))
        """

    def prepare_manifest(self):
        #TODO
        pass

if __name__ == '__main__':
    encoder = Encoder(args)
    encoder.encode_all()
