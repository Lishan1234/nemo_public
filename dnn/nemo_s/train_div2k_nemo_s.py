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
    parser.add_argument('--lr_image_dir', type=str, required=True)
    parser.add_argument('--hr_image_dir', type=str, required=True)

    #video metadata
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--num_steps', type=int, default=300000)

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)

    #log
    parser.add_argument('--custom_tag', type=str, default=None)

    args = parser.parse_args()

    #dnn
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, args.scale, args.upsample_type)
    model = nemo_s.build_model()

    #dataset
    train_ds = train_image_dataset(args.lr_image_dir, args.hr_image_dir, args.batch_size, args.patch_size, args.scale, args.load_on_memory)
    valid_ds = valid_image_dataset(args.lr_image_dir, args.hr_image_dir)

    #trainer
    checkpoint_dir = os.path.join(args.lr_image_dir, 'checkpoint', model.name)
    log_dir = os.path.join(args.lr_image_dir, 'log', model.name)
    #log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.log', ffmpeg_option.summary(args.lr_video_name), model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    trainer = SingleTrainerV1(model, checkpoint_dir, log_dir)
    trainer.train(train_ds, valid_ds, steps=args.num_steps)
