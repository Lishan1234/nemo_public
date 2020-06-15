import time
import argparse
import os

import tensorflow as tf

import nemo.dnn.model
from nemo.tool.video import profile_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #architecture
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, default='deconv')

    args = parser.parse_args()

    #scale, dnn
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    lr_video_profile = profile_video(lr_video_path)
    hr_video_profile = profile_video(hr_video_path)
    scale = hr_video_profile['height'] // lr_video_profile['height']

    with tf.Graph().as_default(), tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model = nemo.dnn.model.build(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type)

        log_dir = os.path.join(args.dataset_dir, 'log', args.hr_video_name,  model.name)
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
