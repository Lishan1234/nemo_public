import tensorflow as tf
import os, time
from importlib import import_module
import scipy.misc
import numpy as np
import sys

import utility as util
from option import args
from dataset import FeatureDataset, TFRecordDataset, ImageDataset

def loss_func(loss_type):
    assert loss_type in ['l1', 'l2']

    if loss_type == 'l1':
        return tf.losses.absolute_difference
    elif loss_type == 'l2':
        return tf.losses.mean_squared_error

class Tester():
    def __init__(self, args, model_builder):
        self.args = args
        self.model = model_builder.build()
        self.model_name = model_builder.get_name()
        self.loss = loss_func(args.loss_type)

        if args.bitrate is None:
            self.sr_image_dir = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}p/sr'.format(args.hr), model_builder.get_name())
            self.feature_image_dir = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}p/feature'.format(args.hr//args.scale), model_builder.get_name())
        else:
            self.sr_image_dir = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}p-{}k/sr'.format(args.hr, args.bitrate), model_builder.get_name())
            self.feature_image_dir = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}p-{}k/feature'.format(args.hr//args.scale, args.bitrate), model_builder.get_name())
        os.makedirs(self.sr_image_dir, exist_ok=True)
        os.makedirs(self.feature_image_dir, exist_ok=True)

        #Checkpoint
        self.root = tf.train.Checkpoint(model=self.model)
        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.train_data, model_builder.get_name())

        #TFRecord
        log_dir = os.path.join(self.args.log_dir, self.args.train_data, model_builder.get_name())
        self.writer = tf.contrib.summary.create_file_writer(log_dir)

    def save_as_h5(self, checkpoint_dir=None):
        assert tf.train.latest_checkpoint(self.checkpoint_dir) is not None
        self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.model.save(os.path.join(self.checkpoint_dir, 'final.h5'), include_optimizer=False)

    #TODO: save model for .pb, .h5 with input shape
    def load_model(self, checkpoint_dir=None):
        assert tf.train.latest_checkpoint(self.checkpoint_dir) is not None
        self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        #filepath = os.path.join(self.checkpoint_dir, 'final.h5')
        #self.model.load_weights(filepath, True)

    def save_feature(self):
        #image_dataloader = TFRecordDataset(self.args).create_valid_dataset()
        image_dataset = ImageDataset(self.args)
        image_dataloader = image_dataset.create_dataset()
        image_dataset_len = image_dataset.get_length()
        with tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for idx, images in enumerate(image_dataloader):
                input_image, _, _= images[0], images[1], images[2]
                output_image = self.model(input_image)
                output_image = tf.squeeze(output_image).numpy()
                scipy.misc.imsave("{}/{:04d}.png".format(self.feature_image_dir, idx+1), output_image)
                np.save("{}/{:04d}.npy".format(self.feature_image_dir, idx+1), output_image)
                util.print_progress(idx+1, image_dataset_len, 'Test Progress:', 'Complete', 1, 50)

    def validate_feature(self):
        feature_dataset = FeatureDataset(self.args, self.model_name)
        feature_dataloader = feature_dataset.create_dataset() #TODO: move to function
        feature_dataset_len = feature_dataset.get_length()
        output_psnr_list = []
        baseline_psnr_list = []
        with tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for idx, images in enumerate(feature_dataloader):
                input_image, target_image, baseline_image = images[0], images[1], images[2]

                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)

                output_psnr_value = tf.image.psnr(output_image, target_image, max_val=1.0)
                baseline_psnr_value = tf.image.psnr(baseline_image, target_image, max_val=1.0)

                output_psnr_list.append(output_psnr_value.numpy())
                baseline_psnr_list.append(baseline_psnr_value.numpy())

                """
                output_image = tf.squeeze(output_image).numpy()
                input_image = tf.squeeze(input_image).numpy()
                target_image = tf.squeeze(target_image).numpy()
                baseline_image = tf.squeeze(baseline_image).numpy()
                scipy.misc.imsave("{}/{:04d}.png".format(self.sr_image_dir, idx+1), output_image)
                scipy.misc.imsave("{}/input_{:04d}.png".format(self.sr_image_dir, idx+1), input_image)
                scipy.misc.imsave("{}/target_{:04d}.png".format(self.sr_image_dir, idx+1), target_image)
                scipy.misc.imsave("{}/bicubic_{:04d}.png".format(self.sr_image_dir, idx+1), baseline_image)
                util.print_progress(idx+1, feature_dataset_len, 'Test Progress:', 'Complete', 1, 50)
                """

            #print result
            print('Average PSNR: {}dB (SR), {}dB (Bicubic)'.format(np.mean(output_psnr_list), np.mean(baseline_psnr_list)))
            print(output_psnr_list) #for checking temporal variation

    def validate(self):
        image_dataset = ImageDataset(self.args)
        image_dataloader = image_dataset.create_dataset()
        image_dataset_len = image_dataset.get_length()
        #image_dataloader = TFRecordDataset(self.args).create_valid_dataset()
        output_psnr_list = []
        baseline_psnr_list = []
        with tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for idx, images in enumerate(image_dataloader):
                input_image, target_image, baseline_image = images[0], images[1], images[2]
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)

                output_psnr_value = tf.image.psnr(output_image, target_image, max_val=1.0)
                baseline_psnr_value = tf.image.psnr(baseline_image, target_image, max_val=1.0)
                #print(baseline_psnr_value)

                output_psnr_list.append(output_psnr_value.numpy())
                baseline_psnr_list.append(baseline_psnr_value.numpy())

                #save image
                output_image = tf.squeeze(output_image).numpy()
                #scipy.misc.imsave("{}/{:04d}.png".format(self.sr_image_dir, idx+1), output_image)
                #scipy.misc.imsave("{}/{}".format(self.sr_image_dir, name), output_image)
                #util.print_progress(idx+1, image_dataset_len, 'Test Progress:', 'Complete', 1, 50)

                #print(baseline_image)
                #print(target_image)

            #print result
            print('Average PSNR: {}dB (SR), {}dB (Bicubic)'.format(np.mean(output_psnr_list), np.mean(baseline_psnr_list)))
            print(output_psnr_list) #for checking temporal variation

            #TODO: tmp
            with open("log_final.txt", "a") as log:
                #log.write("{} {} {} {} {0:.2f}".format(self.args.num_blocks, self.args.num_filters, self.args.upsample_type, self.args.mode, np.mean(output_psnr_list)))
                log.write("{} {} {:.2f}\n".format(self.args.upsample_type, self.args.mode, np.mean(output_psnr_list)))

if __name__ == '__main__':
    tf.enable_eager_execution()

    args.mode = 0
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    tester = Tester(args, model_builder)
    tester.load_model()
    tester.save_feature()

    """
    args.mode = 3
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    tester = Tester(args, model_builder)
    tester.save_as_h5()
    """
    """
    args.mode = 2
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    tester = Tester(args, model_builder)
    tester.load_model()
    tester.validate()

    args.mode = 3
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    tester = Tester(args, model_builder)
    tester.load_model()
    tester.validate()
    """

    """
    args.mode = 0
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    tester = Tester(args, model_builder)
    tester.load_model()
    tester.save_feature()

    args.mode = 1
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    tester_ = Tester(args, model_builder)
    tester_.load_model()
    tester_.validate_feature()

    args.mode = 2
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    tester__ = Tester(args, model_builder)
    tester__.load_model()
    tester__.validate()
    """

    #tester.validate_feature()
    #tester.save_feature()
