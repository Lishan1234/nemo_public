import time
import argparse
import os

from utility import FFmpegOption, upscale_factor
from dataset import train_image_dataset, valid_image_dataset, setup_images
from model.nas_s import NAS_S

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    args = parser.parse_args()

    #dnn
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    scale = upscale_factor(lr_video_path, hr_video_path)
    nas_s = NAS_S(args.num_blocks, args.num_filters, scale)

    with tf.Graph().as_default(), tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model = nas_s.build_model()
        log_dir = os.path.join('.log', model.name)
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
