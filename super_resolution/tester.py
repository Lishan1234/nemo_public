import tensorflow as tf
import os, time
from importlib import import_module
import scipy.misc
import numpy as np

import utility as util

tfe = tf.contrib.eager

def loss_func(loss_type):
    assert loss_type in ['l1', 'l2']

    if loss_type == 'l1':
        return tf.losses.absolute_difference
    elif loss_type == 'l2':
        return tf.losses.mean_squared_error

class Tester():
    def __init__(self, args, model_builder, dataset):
        self.args = args
        self.model = model_builder.build()
        self.loss = loss_func(args.loss_type)
        self.image_dir = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}p'.format(args.lr), 'sr_{}p'.format(args.hr))

        #Checkpoint
        self.root = tf.train.Checkpoint(model=self.model)
        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.train_data, model_builder.get_name())

        #Dataset
        self.len_dataset = dataset.get_length()
        self.dataset = dataset.create_dataset()

    #TODO: save model for .pb, .h5 with input shape
    def load_model(self, checkpoint_dir=None):
        self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def validate(self):
        output_psnr_list = []
        baseline_psnr_list = []
        with tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for idx, images in enumerate(self.dataset):
                input_image, target_image, baseline_image = images[0], images[1], images[2]
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)

                output_psnr_value = tf.image.psnr(output_image, target_image, max_val=1.0)
                baseline_psnr_value = tf.image.psnr(baseline_image, target_image, max_val=1.0)

                output_psnr_list.append(output_psnr_value.numpy())
                baseline_psnr_list.append(baseline_psnr_value.numpy())

                #save image
                output_image = tf.squeeze(output_image).numpy()
                scipy.misc.imsave("{}/{:04d}.png".format(self.image_dir, idx+1), output_image)
                util.print_progress(idx+1, self.len_dataset, 'Test Progress:', 'Complete', 1, 50)

            #print result
            print('Average PSNR: {}dB (SR), {}dB (Bicubic)'.format(np.mean(output_psnr_list), np.mean(baseline_psnr_list)))
            print(output_psnr_list) #for checking temporal variation
