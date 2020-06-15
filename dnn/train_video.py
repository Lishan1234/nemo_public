import time
import argparse
import os
import sys

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from nemo.dnn.dataset import train_image_dataset, test_image_dataset, sample_and_save_images
import nemo.dnn.model
from nemo.dnn.trainer import NEMOTrainer
from nemo.tool.video import profile_video

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--sample_fps', type=float, default=1.0)

    #training & testing
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--pretrained_checkpoint_dir', type=str, default=None)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--num_samples', type=int, default=10)

    #dnn
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, default='deconv')

    #tool
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')

    args = parser.parse_args()

    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']
    lr_image_shape = [lr_video_profile['height'], lr_video_profile['width'], 3]
    hr_image_shape = [lr_video_profile['height'] * scale, lr_video_profile['width'] * scale, 3]

    lr_image_dir = os.path.join(args.dataset_dir, 'image', args.lr_video_name, '{}fps'.format(args.sample_fps))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', args.hr_video_name, '{}fps'.format(args.sample_fps))
    sample_and_save_images(lr_video_path, lr_image_dir, args.sample_fps, args.ffmpeg_path)
    sample_and_save_images(hr_video_path, hr_image_dir, args.sample_fps, args.ffmpeg_path)

    train_ds = train_image_dataset(lr_image_dir, hr_image_dir, lr_image_shape, hr_image_shape, args.batch_size, args.patch_size, args.load_on_memory)
    test_ds = test_image_dataset(lr_image_dir, hr_image_dir, lr_image_shape, hr_image_shape, args.num_samples, args.load_on_memory)

    """
    check patches are generated correctly
    for idx, imgs in enumerate(train_ds.take(3)):
        lr_img = tf.image.encode_png(tf.cast(tf.squeeze(imgs[0]), tf.uint8))
        hr_img = tf.image.encode_png(tf.cast(tf.squeeze(imgs[1]), tf.uint8))
        tf.io.write_file('train_lr_{}.png'.format(idx), lr_img)
        tf.io.write_file('train_hr_{}.png'.format(idx), hr_img)
    for idx, imgs in enumerate(test_ds.take(3)):
        lr_img = tf.image.encode_png(tf.cast(tf.squeeze(imgs[0]), tf.uint8))
        hr_img = tf.image.encode_png(tf.cast(tf.squeeze(imgs[1]), tf.uint8))
        tf.io.write_file('test_lr_{}.png'.format(idx), lr_img)
        tf.io.write_file('test_hr_{}.png'.format(idx), hr_img)
    """

    if args.pretrained_checkpoint_dir is None:
        model = nemo.dnn.model.build(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type)
    else:
        #TODO: validate with DIV2K models
        model = tf.keras.models.load_model(args.pretrained_checkpoint_dir) #used for fine-tuning div2k-learned models

    #TODO: validate
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', args.lr_video_name, model.name)
    log_dir = os.path.join(args.dataset_dir, 'log', args.hr_video_name,  model.name)
    NEMOTrainer(model, checkpoint_dir, log_dir).train(train_ds, test_ds, args.num_epochs, args.num_steps_per_epoch)

    #TODO: check GPU utilization, time per epoch, quality improvement over 10 epoch
