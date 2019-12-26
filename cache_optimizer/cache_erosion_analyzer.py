import os
import sys
import argparse
import glob

import scipy.misc
import numpy as np
import tensorflow as tf

from tool.ffprobe import profile_video
from tool.tensorflow import valid_image_dataset

from dnn.model.edsr_s import EDSR_S

#Note: load a model outside of CRA
class CRA():
    def __init__(self, model, vpxdec_path, content_dir, input_video, compare_video, num_threads, gop):
        self.model = model
        self.vpxdec_path = vpxdec_path
        self.content_dir = content_dir
        self.input_video = input_video
        self.compare_video = compare_video
        self.num_threads = num_threads
        self.gop = gop

    def _prepare_hr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        #check exists
        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        if os.path.exists(hr_image_dir):
            return

        #video metdata
        video_path = os.path.join(self.content_dir, 'video', self.compare_video)
        result = profile_video(video_path)

        #vpxdec (hr, save_frame)
        os.makedirs(hr_image_dir)
        cmd = '{} --codec=vp9 --noblit --threads={} --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame'.format(self.vpxdec_path, self.num_threads, \
                start_idx, end_idx - start_idx, self.content_dir, self.compare_video, postfix)
        os.system(cmd)

        #convert raw to png
        images = glob.glob(os.path.join(hr_image_dir, '*.raw'))
        for idx, image in enumerate(images):
            arr = np.fromfile(image, dtype=np.uint8)
            arr = np.reshape(arr, (result['height'], result['width'], 3))
            name = os.path.splitext(os.path.basename(image))[0]
            name += '.png'
            scipy.misc.imsave(os.path.join(hr_image_dir, name), arr)

    def _prepare_sr_frames(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        #check exists
        hr_image_dir = os.path.join(self.content_dir, 'image', self.compare_video, postfix)
        lr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        #TODO
        #sr_image_dir = os.path.join(self.content_dir, 'image', self.input_video, postfix)
        #if os.path.exists(sr_image_dir):
        #    return

        #video metdata
        video_path = os.path.join(self.content_dir, 'video', self.input_video)
        result = profile_video(video_path)

        #vpxdec (lr, save_frame)
        #os.makedirs(lr_image_dir) TODO
        cmd = '{} --codec=vp9 --noblit --threads={} --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-frame'.format(self.vpxdec_path, self.num_threads, \
                start_idx, end_idx - start_idx, self.content_dir, self.input_video, postfix)
        os.system(cmd)

        #convert raw to png
        """
        images = glob.glob(os.path.join(lr_image_dir, '*.raw'))
        for idx, image in enumerate(images):
            arr = np.fromfile(image, dtype=np.uint8)
            arr = np.reshape(arr, (result['height'], result['width'], 3))
            name = os.path.splitext(os.path.basename(image))[0]
            name += '.png'
            scipy.misc.imsave(os.path.join(lr_image_dir, name), arr)
        """

        #quality
        """
        valid_image_ds = valid_image_dataset(lr_image_dir, hr_image_dir)
        sr_psnr_values = []
        bilinear_psnr_values = []
        for idx, imgs in enumerate(valid_image_ds):
            now = time.perf_counter()
            lr = imgs[0][0]
            hr = imgs[1][0]

            lr = tf.cast(lr, tf.float32)
            sr = self.checkpoint.model(lr)

            #measure sr quality
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            #sr_image = tf.image.encode_png(tf.squeeze(sr))
            #tf.io.write_file(os.path.join(self.decode_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)
            save_name = os.path.basename(imgs[0][1].numpy()[0].decode())
            save_path =
            tf.io.write_file(os.path.join(self.decode_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

            duration = time.perf_counter() - now
            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        #log
        quality_log_path = os.path.join(self.log_dir, 'quality.txt')
        with open(quality_log_path, 'w') as f:
            f.write('Average\t{:.2f}\t{:.2f}\n'.format(np.average(sr_psnr_values), np.average(bilinear_psnr_values)))
            for idx, psnr_values in enumerate(list(zip(sr_psnr_values, bilinear_psnr_values))):
                f.write('{}\t{:.2f}\t{:.2f}\n'.format(idx, psnr_values[0], psnr_values[1]))

        #convert to raw (or save as raw)
        #TODO: save as raw and check it by png
        """

    def _prepare_queue1(self):
        pass

    def _prepare_queue2(self):
        pass

    def prepare(self, chunk_idx):
        #self._prepare_hr_frames(chunk_idx)
        self._prepare_sr_frames(chunk_idx)

    def _analyze_queue1(self):
        #OFFLINE_DNN
        pass

    def _analyze_queue2(self):
        pass

    def analyze(self):
        pass

#Cache Erosion Analyzer (CRA)
"""
class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

class CRA():
    def __init__(self, args):
        self.frame_list = None
        self.content_dir = args.content_dir
        self.input_video_name = args.input_video_name
        self.dnn_video_name = args.dnn_video_name
        self.compare_video_name = args.compare_video_name
        self.num_cores = args.cra_num_cores

        self.result_dir = os.path.join(self.content_dir, "result", self.input_video_name)
        self.log_dir = os.path.join(self.result_dir, "log")
        self.profile_dir = os.path.join(self.result_dir, "profile")

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.profile_dir, exist_ok=True)

        self.base_cmd = "{} --codec=vp9 --summary --noblit --threads={} --frame-buffers=50  --content-dir={} --input-video={} --dnn-video={} --compare-video={}".format(args.vpxdec_path, args.cra_num_threads, self.content_dir, self.input_video_name, self.dnn_video_name, self.compare_video_name)
        if args.num_frames is not None:
            self.base_cmd = "{} --limit={}".format(self.base_cmd, args.num_frames)

    @classmethod
    def get_profile_name(cls, video_index, super_index):
        return "cra_v{}_s{}".format(video_index, super_index)

    def prepare_frame_index(self):
        metadata_log_path = os.path.join(self.log_dir, "metadata_thread01")
        video_index = 0
        super_index = 0

        #run a decoder to generate a log of frame index information
        cmd = "{} --decode-mode=0 --save-metadata".format(self.base_cmd)
        os.system(cmd)

        #load the log and read frame index information
        self.frame_list = []
        with open(metadata_log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                metadata = line.split('\t')
                video_index = int(metadata[0])
                super_index = int(metadata[1])
                frame = Frame(video_index, super_index)
                self.frame_list.append(frame)

    #generate cache profiles for CRA: 1 of |GOP| frames is selceted as an anchor point
    def save_cache_profile(self, video_index, super_index):
        cache_profile_path = os.path.join(self.profile_dir, CRA.get_profile_name(video_index, super_index))
        with open(cache_profile_path, "wb") as f:
            byte_value = 0
            for i, frame in enumerate(self.frame_list):
                if frame.video_index == video_index and frame.super_index == super_index:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frame_list) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def prepare_cache_profiles(self):
        for frame in self.frame_list:
            self.save_cache_profile(frame.video_index, frame.super_index)

    #TODO: use Process and Queue for parallel decoding on multi-cores
    #run the cache profiles & measure the quality
    def run_cache_profiles(self):
        for frame in self.frame_list:
            cache_profile_name = CRA.get_profile_name(frame.video_index, frame.super_index)
            cache_profile_path = os.path.join(self.profile_dir, cache_profile_name)
            cmd = "{} --decode-mode=2 --dnn-mode=2 --cache-policy=1 --cache-profile={} --save-quality --save-metadata".format(self.base_cmd, cache_profile_path)
            os.system(cmd)
"""

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Cache Erosion Analyzer')

    #options for libvpx
    parser.add_argument('--vpxdec_path', type=str, required=True)
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--input_video_name', type=str, required=True)
    parser.add_argument('--compare_video_name', type=str, required=True)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--gop', type=int, required=True)

    #options for edsr_s (DNN)
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    args = parser.parse_args()

    input_video_path = os.path.join(args.content_dir, 'video', args.input_video_name)
    compare_video_path = os.path.join(args.content_dir, 'video', args.compare_video_name)
    input_video_info = profile_video(input_video_path)
    compare_video_info = profile_video(compare_video_path)

    scale = int(compare_video_info['height'] / input_video_info['height'])
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, scale, None)
    checkpoint_dir = os.path.join(args.checkpoint_dir, edsr_s.name)
    print(checkpoint_dir)
    checkpoint = edsr_s.load_checkpoint(checkpoint_dir)

    #cra = CRA(checkpoint.model, args.vpxdec_path, args.content_dir, args.input_video_name, args.compare_video_name, args.num_threads, args.gop)
    #cra.prepare(0)
