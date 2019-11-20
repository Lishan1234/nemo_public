from importlib import import_module
import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from utility import resolve, resolve_bilinear

class Tester:
    def __init__(self, model, checkpoint_dir, log_dir, image_dir):
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                psnr=tf.Variable(-1.0),
                                                optimizer=Adam(0),
                                                model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                                directory=checkpoint_dir, max_to_keep=3)
        self.image_dir = image_dir
        self.log_dir = log_dir
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def test(self, valid_dataset, save_image=False):
        sr_psnr_values = []
        bilinear_psnr_values = []

        for idx, imgs in enumerate(valid_dataset):
            self.now = time.perf_counter()

            lr = imgs[0]
            hr = imgs[1]

            #measure height, width
            if idx == 0:
                hr_shape = tf.shape(hr)[1:3]
                height = hr_shape[0].numpy()
                width = hr_shape[1].numpy()

            #meausre sr quality
            sr = resolve(self.checkpoint.model, lr)
            sr_psnr_value = tf.image.psnr(hr, sr, max_val=255)[0].numpy()
            sr_psnr_values.append(sr_psnr_value)

            #measure bilinear quality
            bilinear = resolve_bilinear(lr, height, width)
            bilinear_psnr_value = tf.image.psnr(hr, bilinear, max_val=255)[0].numpy()
            bilinear_psnr_values.append(bilinear_psnr_value)

            if save_image:
                #save sr images
                sr_image = tf.image.encode_png(tf.squeeze(sr))
                tf.io.write_file(os.path.join(self.image_dir, '{0:04d}.png'.format(idx)), sr_image)

            duration = time.perf_counter() - self.now
            print(f'PSNR(SR) = {sr_psnr_value:.3f}, PSNR(Bilinear) = {bilinear_psnr_value:3f} ({duration:.2f}s)')
        print(f'Summary: PSNR(SR) = {np.average(sr_psnr_values):.3f}, PSNR(Bilinear) = {np.average(bilinear_psnr_values):3f}')

        log_path = os.path.join(self.log_dir, 'quality.txt')
        with open(log_path, 'w') as f:
            for psnr_values in list(zip(sr_psnr_values, bilinear_psnr_values)):
                f.write('{:.2f}\t{:.2f}\n'.format(psnr_values[0], psnr_values[1]))

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str, choices=['uniform', 'keyframes',], default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')

    #dataset
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--enable_normalization', action='store_true')

    #log
    parser.add_argument('--custom_tag', type=str, default=None)

    args = parser.parse_args()

    #0. setting
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))

    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #1. dnn
    scale = upscale_factor(lr_video_path, hr_video_path)
    if args.enable_normalization:
        #TODO: rgb mean
        #normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
        pass
    else:
        normalize_config = None
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, \
                            scale, normalize_config)
    model = edsr_s.build_model()

    #1. dataset
    train_ds = train_image_dataset(lr_image_dir, hr_image_dir, args.batch_size, args.patch_size, scale, args.load_on_memory)
    valid_ds = valid_image_dataset(lr_image_dir, hr_image_dir)

    #2. create a trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), model.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), model.name)
    image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name), model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    tester = Tester(model, checkpoint_dir, log_dir, image_dir)
    tester.test(valid_ds, False)
