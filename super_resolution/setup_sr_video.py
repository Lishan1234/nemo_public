import os, time, sys, time
from importlib import import_module

from config import *
from dataset import InferenceDataset
from option import *
import utility as util

import scipy.misc
import numpy as np
import tensorflow as tf
from PIL import Image

class Tester():
    def __init__(self, args, model_builder):
        self.args = args
        self.model = model_builder.build()
        self.model_name = model_builder.get_name()
        self.output_image_dir = os.path.join(args.data_dir, args.valid_data, args.valid_datatype, '{}p_{}'.format(args.target_resolution, self.model_name))
        self.root = tf.train.Checkpoint(model=self.model)
        self.checkpoint_dir = os.path.join(args.data_dir, args.train_data, args.train_datatype, args.checkpoint_dir, model_builder.get_name())
        os.makedirs(self.output_image_dir, exist_ok=True)

    #load a model
    def load_model(self, checkpoint_dir=None):
        print(self.checkpoint_dir)
        assert tf.train.latest_checkpoint(self.checkpoint_dir) is not None
        self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    #apply a DNN
    def process(self):
        start_time = 0
        end_time = 0
        image_dataset = InferenceDataset(self.args)
        image_dataloader = image_dataset.create_dataset()
        image_dataset_len = image_dataset.get_length()
        with tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for idx, images in enumerate(image_dataloader):
                start_time = time.time()
                input_image = images[0]
                input_image = tf.expand_dims(input_image, axis=0)
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)
                output_image = tf.round(output_image * 255.0)

                #save image
                output_image = tf.squeeze(output_image).numpy().astype(np.uint8)
                output_image.tofile("{}/{:04d}.raw".format(self.output_image_dir, idx+1))
                util.print_progress(idx+1, image_dataset_len, 'Test Progress:', 'Complete', 1, 50)

    #TODO: refactor to encode in memory
    #def process_and_encode(self):

    #encode a video
    def encode(self):
        assert os.path.exists(self.output_image_dir)

        valid_datatype = self.args.valid_datatype.split('_')
        length = int(valid_datatype[1])
        start = int(valid_datatype[2])
        original_video_path = os.path.join(self.args.data_dir, self.args.valid_data, 'video', '{}p.webm'.format(self.args.original_resolution))
        sr_video_path = os.path.join(self.args.data_dir, self.args.valid_data, 'video', '{}p_{}p_{}sec_{}st_{}.webm'.format(self.args.target_resolution, self.args.target_resolution // self.args.scale, length, start, self.model_name))
        fps = util.findFPS(original_video_path)
        height = args.target_resolution // args.scale
        width = WIDTH[height]
        print(fps, height, width)

        #TODO: temporally set key interval to 30
        cmd = '/usr/bin/ffmpeg -framerate {} -s 1920x1080 -pix_fmt rgb24 -i {}/%04d.raw -vcodec libvpx-vp9 -threads 4 -speed 4 -pix_fmt yuv420p -lossless 1 -keyint_min {} -g {} -y {}'.format(fps, self.output_image_dir, KEY_INTERVAL, KEY_INTERVAL, sr_video_path)
        os.system(cmd)

        cmd = 'rm {}/*.raw'.format(self.output_image_dir)
        os.system(cmd)

if __name__ == '__main__':
    tf.enable_eager_execution()

    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)

    tester = Tester(args, model_builder)
    tester.load_model()
    tester.process()
    tester.encode()
