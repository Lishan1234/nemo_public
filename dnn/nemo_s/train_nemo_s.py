import time
import argparse
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from dnn.dataset import train_image_dataset, valid_image_dataset, setup_images
from dnn.model.nemo_s import NEMO_S
from dnn.train import SingleTrainerV1
from tool.video import profile_video, FFmpegOption

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')

    #video metadata
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--num_steps', type=int, default=300000)
    parser.add_argument('--div2k_checkpoint_dir', type=str, default=None)

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #log
    parser.add_argument('--custom_tag', type=str, default=None)

    args = parser.parse_args()

    #scale
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']

    #image
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dnn
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)
    model = nemo_s.build_model()

    #dataset
    train_ds = train_image_dataset(lr_image_dir, hr_image_dir, args.batch_size, args.patch_size, scale, args.load_on_memory)
    valid_ds = valid_image_dataset(lr_image_dir, hr_image_dir)

    #trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), model.name)
    #log_dir = os.path.join(args.dataset_dir, 'tensorflow', ffmpeg_option.summary(args.lr_video_name), model.name)
    log_dir = os.path.join('.', ffmpeg_option.summary(args.lr_video_name), model.name)
    #log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.log', ffmpeg_option.summary(args.lr_video_name), model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    trainer = SingleTrainerV1(model, checkpoint_dir, log_dir)
    if args.div2k_checkpoint_dir is not None:
        div2k_checkpoint_dir = os.path.join(args.div2k_checkpoint_dir, 'x{}'.format(scale), 'checkpoint', model.name)
        trainer.restore(div2k_checkpoint_dir)
    trainer.train(train_ds, valid_ds, steps=args.num_steps)
