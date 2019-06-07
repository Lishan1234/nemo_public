import os, glob, random, sys, time, argparse
import utility as util
from config import *

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
parser.add_argument('--fps', type=float, default=5.0)
parser.add_argument('--mode', type=str,  required=True)
parser.add_argument('--enable_debug', action='store_true')
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument('--num_patch', type=int, default=10000)
#parser.add_argument('--lr', type=int, default=240)
#parser.add_argument('--scale', type=int, required=True)

args = parser.parse_args()
#scale = [2, 3, 4]
scales = [1, 4]

BITRATE = {240: 512, 270: 512, 1080: 4400} #WOWZA recommendation
RESOLUTION={240: (240, 426), 360: (360, 480), 480: (480, 858), 720: (720, 1280), 1080: (1080, 1920)}
VP9_DASH_PARAMS="-tile-columns 4 -frame-parallel 1"

video_dir = os.path.join(args.data_dir, args.dataset, "video")
frame_dir = os.path.join(args.data_dir, args.dataset, "{}_{}_{}_{:.2f}".format(args.target_resolution, args.video_len, args.video_start, args.fps))
sr_dir = os.path.join(args.data_dir, args.dataset, "{}_{}_{}".format(args.target_resolution, args.video_len, args.video_start))
tfrecord_dir = os.path.join(args.data_dir, args.dataset, "{}_{}_{}_{:.2f}".format(args.target_resolution, args.video_len, args.video_start, args.fps))
os.makedirs(video_dir, exist_ok=True)
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(sr_dir, exist_ok=True)
os.makedirs(tfrecord_dir, exist_ok=True)
original_video_path = os.path.join(video_dir, "{}p.{}".format(args.original_resolution, args.video_format))

len_hour = "{:02d}".format(int(args.video_len / 3600))
remainder = args.video_len - int(args.video_len / 3600) * 3600
len_min = "{:02d}".format(int(remainder / 60))
len_sec = "{:02d}".format(args.video_len % 60)

start_hour = "{:02d}".format(int(args.video_start / 3600))
remainder = args.video_start - int(args.video_start / 3600) * 3600
start_min = "{:02d}".format(int(remainder / 60))
start_sec = "{:02d}".format(args.video_start % 60)

print(original_video_path)
assert os.path.exists(original_video_path)

#TODO: prepare bicubic upsampled images
def setup_video():
    for scale in scales:
        #Cut&Transcode videos
        resized_resolution = args.target_resolution//scale
        resize_bitrate = BITRATE[resized_resolution]
        resized_video_path = os.path.join(video_dir, "{}p_{}k_{}sec_{}st.{}".format(resized_resolution, resize_bitrate, args.video_len, args.video_start, args.video_format))

        if not os.path.exists(resized_video_path):
            """ Temporally use GOP size as 30 """
            #cmd = "ffmpeg -i {} -ss {}:{}:{} -t {}:{}:{} -c:v libvpx-vp9 -vf scale=-1:{} -b:v {}k -keyint_min 120 -g 120 -threads 4 -speed 4 {} -an -f webm -dash 1 {}".format(original_video_path, start_hour, start_min, start_sec, len_hour, len_min, len_sec, resized_resolution, resize_bitrate, VP9_DASH_PARAMS, resized_video_path)
            cmd = "ffmpeg -i {} -ss {}:{}:{} -t {}:{}:{} -c:v libvpx-vp9 -vf scale=-1:{} -b:v {}k -keyint_min {} -g {} -threads 4 -speed 4 {} -an -f webm -dash 1 {}".format(original_video_path, start_hour, start_min, start_sec, len_hour, len_min, len_sec, resized_resolution, resize_bitrate, KEY_INTERVAL, KEY_INTERVAL, VP9_DASH_PARAMS, resized_video_path)
            os.system(cmd)

        #Generate images by sampling
        sampled_frame_dir = os.path.join(frame_dir, "{}p".format(resized_resolution))
        os.makedirs(sampled_frame_dir, exist_ok=True)
        cmd = "ffmpeg -i {} -vf fps={} {}/%04d.png".format(resized_video_path, args.fps, sampled_frame_dir)
        os.system(cmd)

        if scale != 1:
            #Transcode bicubic interpolated videos
            upsample_bitrate = BITRATE[args.target_resolution]
            upsample_video_path = os.path.join(video_dir, "{}p_{}p_{}k_{}sec_{}st.{}".format(args.target_resolution, resized_resolution, upsample_bitrate, args.video_len, args.video_start, args.video_format))
            if not os.path.exists(upsample_video_path):
                """ Temporally use GOP size as 30 """
                #cmd = "ffmpeg -i {} -c:v libvpx-vp9 -vf scale=-1:{} -b:v {}k -keyint_min 120 -g 120 -threads 4 -speed 4 {} -an -f webm -dash 1 {}".format(resized_video_path, args.target_resolution, upsample_bitrate, VP9_DASH_PARAMS, upsample_video_path)
                cmd = "ffmpeg -i {} -c:v libvpx-vp9 -vf scale=-1:{} -b:v {}k -keyint_min {} -g {} -threads 4 -speed 4 {} -an -f webm -dash 1 {}".format(resized_video_path, args.target_resolution, upsample_bitrate, KEY_INTERVAL, KEY_INTERVAL, VP9_DASH_PARAMS, upsample_video_path)
                os.system(cmd)

            #Generate images by sampling
            sampled_frame_dir = os.path.join(frame_dir, "{}p_{}p_bicubic".format(args.target_resolution, resized_resolution))
            os.makedirs(sampled_frame_dir, exist_ok=True)
            cmd = "ffmpeg -i {} -vf fps={} {}/%04d.png".format(upsample_video_path, args.fps, sampled_frame_dir)
            os.system(cmd)

    #Lossless encoding for target resolution (training data)
    resized_video_path =os.path.join(video_dir, "{}p_lossless_{}sec_{}st.{}".format(args.target_resolution, args.video_len, args.video_start, args.video_format))
    if not os.path.exists(resized_video_path):
        cmd = "ffmpeg -i {} -ss {}:{}:{} -t {}:{}:{} -c:v libvpx-vp9 -vf scale=-1:{} -lossless 1 {}".format(original_video_path, start_hour, start_min, start_sec, len_hour, len_min, len_sec, args.target_resolution, resized_video_path)
        os.system(cmd)

    #Generate images by sampling
    sampled_frame_dir = os.path.join(frame_dir, "{}p_lossless".format(args.target_resolution))
    os.makedirs(sampled_frame_dir, exist_ok=True)
    cmd = "ffmpeg -i {} -vf fps={} {}/%04d.png".format(resized_video_path, args.fps, sampled_frame_dir)
    os.system(cmd)

def setup_sr():
    for scale in scales:
        if (scale == 1):
            continue
        resized_resolution = args.target_resolution // scale
        resize_bitrate = BITRATE[resized_resolution]
        resized_video_path = os.path.join(video_dir, "{}p_{}k_{}sec_{}st.{}".format(resized_resolution, resize_bitrate, args.video_len, args.video_start, args.video_format))

        assert (os.path.exists(resized_video_path))
        input_dir = os.path.join(sr_dir, "{}p".format(resized_resolution))
        os.makedirs(input_dir, exist_ok=True)
        cmd = "ffmpeg -i {} {}/%04d.png".format(resized_video_path, input_dir)
        os.system(cmd)

def setup_tfrecord():
    tf.enable_eager_execution()

    for scale in scales:
        if scale == 1: #skip scale==1
            continue
        resized_resolution = args.target_resolution//scale

        train_tfrecord_path = os.path.join(tfrecord_dir, "train_{}p.tfrecords".format(resized_resolution))
        train_writer = tf.io.TFRecordWriter(train_tfrecord_path)

        hr_dir = os.path.join(frame_dir, "{}p_lossless".format(args.target_resolution))
        lr_dir = os.path.join(frame_dir, "{}p".format(resized_resolution))
        lr_bicubic_dir = os.path.join(frame_dir, "{}p_{}p_bicubic".format(args.target_resolution, resized_resolution))

        hr_filenames = sorted(glob.glob("{}/*.png".format(hr_dir)))
        lr_filenames = sorted(glob.glob("{}/*.png".format(lr_dir)))
        lr_bicubic_filenames = sorted(glob.glob("{}/*.png".format(lr_bicubic_dir)))
        hr_images = []
        lr_images = []
        lr_bicubic_images = []

        for hr_filename, lr_filename, lr_bicubic_filename in zip(hr_filenames, lr_filenames, lr_bicubic_filenames):
            with tf.device('cpu:0'):
                hr_images.append(util.load_image(hr_filename))
                lr_images.append(util.load_image(lr_filename))
                lr_bicubic_images.append(util.load_image(lr_bicubic_filename))

        assert len(lr_images) == len(hr_images) == len(lr_bicubic_images)
        assert len(lr_images) != 0
        print('dataset length: {}'.format(len(lr_images)))

        count = 0
        while count < args.num_patch:
            rand_idx = random.randint(0, len(lr_images) - 1)
            height, width, channel = lr_images[rand_idx].get_shape().as_list()

            if height < (args.patch_size + 1) or width < (args.patch_size + 1):
                continue
            else:
                count += 1

            if count == 1:
                start_time = time.time()
            elif count % 1000 == 0:
                print('Train TFRecord Process status: [{}/{}] / Take {} seconds'.format(count, args.num_patch, time.time() - start_time))
                start_time = time.time()

            hr_image, lr_image = util.crop_augment_image(hr_images[rand_idx], lr_images[rand_idx], scale, args.patch_size)

            #Debug
            if args.enable_debug:
                if hr_image.get_shape().as_list()[-1] == 1:
                    Image.fromarray(np.uint8(tf.squeeze(hr_image).numpy()*255), mode='L').save('hr.png')
                    Image.fromarray(np.uint8(tf.squeeze(lr_image).numpy()*255), mode='L').save('lr.png')
                else:
                    Image.fromarray(np.uint8(hr_image.numpy()*255)).save('hr.png')
                    Image.fromarray(np.uint8(lr_image.numpy()*255)).save('lr.png')
                sys.exit()

            hr_binary_image = hr_image.numpy().tostring()
            lr_binary_image = lr_image.numpy().tostring()

            feature = {
                'hr_image_raw': util._bytes_feature(hr_binary_image),
                'lr_image_raw': util._bytes_feature(lr_binary_image),
                'patch_size': util._int64_feature(args.patch_size),
                'scale': util._int64_feature(scale),
                'channel': util._int64_feature(hr_image.get_shape().as_list()[-1]),
                }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            train_writer.write(tf_example.SerializeToString())
        train_writer.close()

        valid_tfrecord_path = os.path.join(tfrecord_dir, "valid_{}p.tfrecords".format(resized_resolution))
        valid_writer = tf.io.TFRecordWriter(valid_tfrecord_path)

        for i in range(len(hr_images)):
            hr_image, lr_image, lr_bicubic_image = hr_images[i], lr_images[i], lr_bicubic_images[i]

            hr_binary_image = hr_image.numpy().tostring()
            lr_binary_image = lr_image.numpy().tostring()
            lr_bicubic_binary_image = lr_bicubic_image.numpy().tostring()
            lr_image_shape = lr_image.get_shape().as_list()

            feature = {
                'hr_image_raw': util._bytes_feature(hr_binary_image),
                'lr_image_raw': util._bytes_feature(lr_binary_image),
                'lr_bicubic_image_raw': util._bytes_feature(lr_bicubic_binary_image),
                'height': util._int64_feature(lr_image_shape[0]),
                'width': util._int64_feature(lr_image_shape[1]),
                'channel': util._int64_feature(hr_image.get_shape().as_list()[-1]),
                'scale': util._int64_feature(scale),
                }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            valid_writer.write(tf_example.SerializeToString())
        valid_writer.close()

if __name__ == "__main__":
    if args.mode == "video":
        setup_video()
    elif args.mode == "tfrecord":
        setup_tfrecord()
    elif args.mode == "sr":
        setup_sr()
    elif args.mode == "all":
        setup_video()
        setup_tfrecord()
    else:
        raise NotImplementedError
