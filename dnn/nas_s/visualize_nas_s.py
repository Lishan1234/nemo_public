import time
import argparse
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from dnn.dataset import train_image_dataset, valid_image_dataset, setup_images
from dnn.model.nas_s import NAS_S
from tool.video import profile_video, FFmpegOption

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

    #scale, dnn
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']
    nas_s = NAS_S(args.num_blocks, args.num_filters, scale)

    with tf.Graph().as_default(), tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model = nas_s.build_model()

        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.log')
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
