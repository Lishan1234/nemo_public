import os, time, sys, time
from importlib import import_module

from config import *
from dataset import TrainingDataset
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
        self.root = tf.train.Checkpoint(model=self.model)
        self.checkpoint_dir = os.path.join(args.data_dir, args.train_data, args.train_datatype, args.checkpoint_dir, model_builder.get_name())
        self.result_dir= os.path.join(args.data_dir, args.train_data, args.train_datatype, args.result_dir, model_builder.get_name())
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    #load a model
    def load_model(self, checkpoint_dir=None):
        assert tf.train.latest_checkpoint(self.checkpoint_dir) is not None
        self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    #apply a DNN
    def process(self):
        sr_psnr = []
        bicubic_psnr = []
        log_path = os.path.join(self.result_dir, 'quality.log')
        image_dataset = TrainingDataset(self.args)
        image_dataloader = image_dataset.create_dataset()
        image_dataset_len = image_dataset.get_length()

        with tf.device('gpu:{}'.format(self.args.gpu_idx)), open(log_path, 'w') as f:
            for idx, images in enumerate(image_dataloader):
                #apply super-resolution
                start_time = time.time()
                input_image = images[0]
                output_image = self.model(input_image)
                output_image - tf.squeeze(output_image, axis=0)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)
                output_image = tf.round(output_image * 255.0)
                output_image = tf.cast(output_image, tf.uint8)

                #measure quality (PSNR)
                target_image = tf.round(images[1] * 255.0)
                target_image = tf.cast(target_image, tf.uint8)
                compare_image = tf.round(images[2] * 255.0)
                compare_image = tf.cast(compare_image, tf.uint8)

                sr_psnr.append(tf.image.psnr(output_image, target_image, max_val=255.0)[0])
                bicubic_psnr.append(tf.image.psnr(compare_image, target_image, max_val=255.0)[0])

                #logging
                f.write('{}\t{:.2f}\t{:.2f}\n'.format(idx, sr_psnr[idx], bicubic_psnr[idx]))

                util.print_progress(idx+1, image_dataset_len, 'Test Progress:', 'Complete', 1, 50)

            #logging
            f.write('average\t{}\t{}'.format(np.average(sr_psnr), np.average(bicubic_psnr)))
            print('[test_quality.py finish] sr_psnr: {:.2f}dB, bicubic_psnr: {:.2f}dB'.format(np.average(sr_psnr), np.average(bicubic_psnr)))

if __name__ == '__main__':
    tf.enable_eager_execution()

    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)

    tester = Tester(args, model_builder)
    tester.load_model()
    tester.process()
