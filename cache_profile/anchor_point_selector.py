import os
import sys
import argparse
import shlex
import math
import time
import multiprocessing as mp
import shutil

import numpy as np
import tensorflow as tf

from nemo.tool.video import profile_video
from nemo.tool.libvpx import *
from nemo.tool.mac import count_mac_for_dnn, count_mac_for_cache
import nemo.dnn.model

class AnchorPointSelector():
    NAME1 = "NEMO_BOUND"

    def __init__(self, model, vpxdec_path, dataset_dir, lr_video_name, hr_video_name, gop, output_width, output_height, \
                 quality_margin, num_decoders, max_num_anchor_points):
        self.model = model
        self.vpxdec_path = vpxdec_path
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.output_width = output_width
        self.output_height = output_height
        self.quality_margin = quality_margin
        self.num_decoders = num_decoders
        self.max_num_anchor_points = max_num_anchor_points

    def _select_anchor_point(self, current_anchor_points, anchor_point_candidates):
        max_estimated_quality = None
        max_avg_estimated_quality = None
        idx = None

        for i, new_anchor_point in enumerate(anchor_point_candidates):
            estimated_quality = self._estimate_quality(current_anchor_points, new_anchor_point)
            avg_estimated_quality = np.average(estimated_quality)
            if max_avg_estimated_quality is None or avg_estimated_quality > max_avg_estimated_quality:
                max_avg_estimated_quality = avg_estimated_quality
                max_estimated_quality = estimated_quality
                idx = i

        return idx, max_estimated_quality

    def _estimate_quality(self, currnet_anchor_points, new_anchor_point):
        if currnet_anchor_points is not None:
            return np.maximum(currnet_anchor_points.estimated_quality, new_anchor_point.measured_quality)
        else:
            return new_anchor_point.measured_quality

    def _select_anchor_point_set_nemo(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)
        if self.max_num_anchor_points is not None:
            algorithm_type = 'nemo_{}'.format(self.max_num_anchor_points)
        else:
            algorithm_type = 'nemo'

        ###########step 1: analyze anchor points##########
        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx_save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx_save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx_setup_sr_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx_bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx_offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))


        #create multiple processes for parallel quality measurements
        start_time = time.time()
        q0 = mp.Queue()
        q1 = mp.Queue()
        decoders = [mp.Process(target=libvpx_offline_cache_quality_mt, args=(q0, q1, self.vpxdec_path, self.dataset_dir, \
                                    self.lr_video_name, self.hr_video_name, self.model.name, self.output_width, self.output_height)) for i in range(self.num_decoders)]
        for decoder in decoders:
            decoder.start()

        #select a single anchor point and measure the resulting quality
        single_anchor_point_sets = []
        frames = libvpx_load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        for idx, frame in enumerate(frames):
            anchor_point_set = AnchorPointSet.create(frames, profile_dir, '{}.profile'.format(frame.name))
            anchor_point_set.add_anchor_point(frame)
            anchor_point_set.save_cache_profile()
            q0.put((anchor_point_set.get_cache_profile_name(), num_skipped_frames, num_decoded_frames, postfix, idx))
            single_anchor_point_sets.append(anchor_point_set)
        for frame in frames:
            item = q1.get()
            idx = item[0]
            quality = item[1]
            single_anchor_point_sets[idx].set_measured_quality(quality)
            single_anchor_point_sets[idx].remove_cache_profile()

        #remove multiple processes
        for decoder in decoders:
            q0.put('end')
        for decoder in decoders:
            decoder.join()

        end_time = time.time()
        print('{} video chunk: (Step1-profile anchor point quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: order anchor points##########
        start_time = time.time()
        multile_anchor_point_sets = []
        anchor_point_set = None
        while len(single_anchor_point_sets) > 0:
            anchor_point_idx, estimated_quality = self._select_anchor_point(anchor_point_set, single_anchor_point_sets)
            selected_anchor_point = single_anchor_point_sets.pop(anchor_point_idx)
            if len(multile_anchor_point_sets) == 0:
                anchor_point_set = AnchorPointSet.load(selected_anchor_point, profile_dir, '{}_{}.profile'.format(algorithm_type, len(selected_anchor_point.anchor_points)))
                anchor_point_set.set_estimated_quality(selected_anchor_point.measured_quality)
            else:
                anchor_point_set = AnchorPointSet.load(multile_anchor_point_sets[-1], profile_dir, '{}_{}.profile'.format(algorithm_type, len(multile_anchor_point_sets[-1].anchor_points) + 1))
                anchor_point_set.add_anchor_point(selected_anchor_point.anchor_points[0])
                anchor_point_set.set_estimated_quality(estimated_quality)
            multile_anchor_point_sets.append(anchor_point_set)
        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 3: select anchor points##########
        start_time = time.time()
        log_path = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        with open(log_path, 'w') as f:
            for anchor_point_set in multile_anchor_point_sets:
                #log quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx_offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                                    num_skipped_frames, num_decoded_frames, postfix)
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(len(anchor_point_set.anchor_points), np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear), np.average(anchor_point_set.estimated_quality))
                f.write(quality_log)

                print('{} video chunk, {} anchor points: PSNR(Cache)={:.4f}, PSNR(SR)={:.4f}, PSNR(Bilinear)={:.4f}'.format( \
                                        chunk_idx, len(anchor_point_set.anchor_points), np.average(quality_cache), np.average(quality_dnn), \
                                        np.average(quality_bilinear)))

                #terminate
                if np.average(quality_diff) <= self.quality_margin or anchor_point_set.num_anchor_points() == self.max_num_anchor_points:
                    anchor_point_set.set_cache_profile_name('{}.profile'.format(algorithm_type))
                    anchor_point_set.save_cache_profile()
                    libvpx_offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)
                    break

        end_time = time.time()
        print('{} video chunk: (Step3) {}sec'.format(chunk_idx, end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    def _select_anchor_point_set_uniform(self, chunk_idx=None):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)
        algorithm_type = 'uniform'

        ###########step 1: measure bilinear, dnn quality##########
        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = profile_video(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx_save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx_save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx_setup_sr_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx_bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx_offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: select anchor points##########
        start_time = time.time()
        frames = libvpx_load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        log_path = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        with open(log_path, 'w') as f:
            for i in range(len(frames)):
                #select anchor point uniformly
                num_anchor_points = i + 1
                anchor_point_set = AnchorPointSet.create(frames, profile_dir, '{}_{}.profile'.format(algorithm_type, num_anchor_points))
                for j in range(num_anchor_points):
                    idx = j * math.floor(len(frames) / num_anchor_points)
                    anchor_point_set.add_anchor_point(frames[idx])

                #measure the quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx_offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                        self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                        num_skipped_frames, num_decoded_frames, postfix)
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(len(anchor_point_set.anchor_points), np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                f.write(quality_log)
                print('{} video chunk, {} anchor points: PSNR(Cache)={:.4f}, PSNR(SR)={:.4f}, PSNR(Bilinear)={:.4f}'.format( \
                                        chunk_idx, len(anchor_point_set.anchor_points), np.average(quality_cache), np.average(quality_dnn), \
                                        np.average(quality_bilinear)))
                #terminate
                if np.average(quality_diff) <= self.quality_margin:
                    anchor_point_set.set_cache_profile_name('{}.profile'.format(algorithm_type))
                    anchor_point_set.save_cache_profile()
                    libvpx_offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)
                    break

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    def _select_anchor_point_set_random(self, chunk_idx=None):
        pass

    def _select_anchor_point_set_exhaustive(self, chunk_idx=None):
        pass

    def select_anchor_point_set(self, algorithm_type, chunk_idx=None):
        if chunk_idx is not None:
            if algorithm_type == 'nemo':
                self._select_anchor_point_set_nemo(chunk_idx)
            elif algorithm_type == 'uniform':
                self._select_anchor_point_set_uniform(chunk_idx)
            elif algorithm_type == 'random':
                self._select_anchor_point_set_random(chunk_idx)
            elif algorithm_type == 'exhaustive':
                self._select_anchor_point_set_exhaustive(chunk_idx)
        else:
            lr_video_path = os.path.join(self.dataset_dir, 'video', args.lr_video_name)
            lr_video_profile = profile_video(lr_video_path)
            num_chunks = int(math.ceil(lr_video_profile['duration'] / (args.gop / lr_video_profile['frame_rate'])))
            for i in range(num_chunks):
                if algorithm_type == 'nemo':
                    self._select_anchor_point_set_nemo(chunk_idx)
                elif algorithm_type == 'uniform':
                    self._select_anchor_point_set_uniform(chunk_idx)
                elif algorithm_type == 'random':
                    self._select_anchor_point_set_random(chunk_idx)
                elif algorithm_type == 'exhaustive':
                    self._select_anchor_point_set_exhaustive(chunk_idx)

    def summary(self, start_idx, end_idx):
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, '{}_{}_{}'.format(self.NAME1, self.max_num_anchor_points, self.quality_margin))
        profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name)

        #log
        quality_summary_file = os.path.join(log_dir, 'quality.txt')
        with open(quality_summary_file, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                chunk_log_dir = os.path.join(log_dir, 'chunk{:04d}'.format(chunk_idx))
                if not os.path.exists(chunk_log_dir):
                    break
                else:
                    quality_log_path = os.path.join(chunk_log_dir, 'quality.txt')
                    with open(quality_log_path, 'r') as f1:
                        q_lines = f1.readlines()
                        f0.write('{}\t{}\n'.format(chunk_idx, q_lines[-1].strip()))

        #cache profile
        cache_profile = os.path.join(profile_dir, '{}_{}_{}.profile'.format(self.NAME1, self.max_num_anchor_points, self.quality_margin))
        cache_data = b''
        with open(cache_profile, 'wb') as f0:
            for chunk_idx in range(start_idx, end_idx):
                chunk_cache_profile = os.path.join(profile_dir, 'chunk{:04d}'.format(chunk_idx), '{}_{}_{}.profile'.format(self.NAME1, self.max_num_anchor_points, self.quality_margin))
                with open(chunk_cache_profile, 'rb') as f1:
                    f0.write(f1.read())

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--vpxdec_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)

    #dnn
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')
    parser.add_argument('--scale', type=int, default=None)
    parser.add_argument('--train_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--quality_margin', type=float, default=0.5)
    parser.add_argument('--gop', type=int, default=120)
    parser.add_argument('--max_num_anchor_points', type=int, default=24)
    parser.add_argument('--chunk_idx', type=str, default=None) #None: profile/summary all chunks, [start index],[end index]: profile/sumamry partial chunks
    parser.add_argument('--num_decoders', default=8, type=int)
    parser.add_argument('--task', choices=['profile','summary', 'all'], default='all')
    parser.add_argument('--algorithm', choices=['nemo','uniform', 'random', 'exhaustive'], required=True)

    args = parser.parse_args()

    if args.vpxdec_path is None:
        args.vpxdec_path = os.path.join(os.environ['NEMO_ROOT'], 'third_party', 'libvpx', 'vpxdec_nemo_ver2')
        assert(os.path.exists(args.vpxdec_path))

    #profile videos
    dataset_dir = os.path.join(args.data_dir, args.content)
    lr_video_path = os.path.join(dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(dataset_dir, 'video', args.hr_video_name)
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = args.output_height // lr_video_profile['height']
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]

    #load a dnn
    model = nemo.dnn.model.build(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type)
    if args.train_type == 'train_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
    elif args.train_type == 'finetune_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, '{}_finetune'.format(model.name))
    elif args.train_type == 'train_div2k':
        checkpoint_dir = os.path.join(args.data_dir, 'DIV2K', 'checkpoint', 'DIV2K_X{}'.format(scale), model.name)
    else:
        raise ValueError('Unsupported training types')
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    ckpt.restore(ckpt_path)

    #run aps
    aps = AnchorPointSelector(ckpt.model, args.vpxdec_path, dataset_dir, args.lr_video_name, args.hr_video_name, args.gop, \
                              args.output_width, args.output_height, args.quality_margin, args.num_decoders, args.max_num_anchor_points)

    if args.task == 'profile':
        #aps.select_anchor_point_set(args.algorithm, args.chunk_idx)
        pass
    elif args.task == 'summary':
        pass
    elif args.task == 'all':
        #aps.select_anchor_point_set(args.algorithm, args.chunk_idx)
        aps.select_anchor_point_set(args.algorithm, 0)
        pass
    else:
        raise NotImplementedError
