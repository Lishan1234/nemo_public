import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import cv2
import os, glob

from mnasdataset import MnasDataset
from ops import load_image

#for testing
import sys
sys.path.append('..')
from option import args

class BigbuckbunnyV0(MnasDataset):
    def __init__(self, args, scale):
        self.load_on_memory = args.load_on_memory
        self.patch_size = args.patch_size
        self.num_batch = args.num_batch
        self.scale = scale

        lr_image_path = os.path.join(args.data_dir, '{}fps'.format(str(args.fps)), 'lr_x{}'.format(scale))
        hr_image_path = os.path.join(args.data_dir, '{}fps'.format(str(args.fps)), 'hr')
        self.lr_image_filenames = glob.glob('{}/*.png'.format(lr_image_path))
        self.hr_image_filenames = glob.glob('{}/*.png'.format(hr_image_path))

        if self.load_on_memory:
            self.lr_images = []
            self.hr_images = []

            for lr_filename, hr_filename in zip(self.lr_image_filenames, self.hr_image_filenames):
                self.lr_images.append(load_image(lr_filename))
                self.hr_images.append(load_image(hr_filename))

    def _crop_image(self, lr_image, hr_image):
            #crop
            lr_image_cropped = tf.image.random_crop(lr_image, [self.patch_size, self.patch_size, 3])
            hr_image_cropped = tf.image.random_crop(hr_image, [self.patch_size * self.scale, self.patch_size * self.scale, 3])

            return lr_image_cropped, hr_image_cropped

    def _load_decode_image(self, lr_filename, hr_filename):
            #load
            lr_image_string = tf.io.read_file(lr_filename)
            hr_image_string = tf.io.read_file(hr_filename)

            #decode
            lr_image_decoded = tf.image.decode_image(lr_filename)
            lr_image_decoded = tf.image.convert_image_dtype(lr_image_decoded, tf.float32)
            hr_image_decoded = tf.image.decode_image(hr_filename)
            hr_image_decoded = tf.image.convert_image_dtype(hr_image_decoded, tf.float32)

            return lr_image_decoded, hr_image_decoded

    def _crop_image(self, lr_image, hr_image):
            #crop
            lr_image_cropped = tf.image.random_crop(lr_image, [self.patch_size, self.patch_size, 3])
            hr_image_cropped = tf.image.random_crop(hr_image, [self.patch_size * self.scale, self.patch_size * self.scale, 3])

            return lr_image_cropped, hr_image_cropped

    def _load_decode_crop_image(self, lr_filename, hr_filename):
            #load
            lr_image_string = tf.io.read_file(lr_filename)
            hr_image_string = tf.io.read_file(hr_filename)

            #decode
            lr_image_decoded = tf.image.decode_image(lr_filename)
            lr_image_decoded = tf.image.convert_image_dtype(lr_image_decoded, tf.float32)
            hr_image_decoded = tf.image.decode_image(hr_filename)
            hr_image_decoded = tf.image.convert_image_dtype(hr_image_decoded, tf.float32)

            #crop
            lr_image_cropped = tf.image.random_crop(lr_image_decoded, [self.patch_size, self.patch_size, 3])
            hr_image_cropped = tf.image.random_crop(hr_image_decoded, [self.patch_size * self.scale, self.patch_size * self.scale, 3])

            return lr_image_cropped, hr_image_cropped

    def create_train_dataset(self):
        if self.load_on_memory:
            dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_images))
            dataset = dataset.shuffle(10000)
            dataset = dataset.repeat(None)
            dataset = dataset.map(self._crop_image, num_parallel_calls=4)
            dataset = dataset.batch(self.num_batch)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.lr_image_filenames, self.hr_image_filenames))
            #dataset = dataset.map(self._load_decode_image, num_parallel_calls=4)
            dataset = dataset.shuffle(10000)
            dataset = dataset.repeat(None)
            dataset = dataset.map(self._load_decode_crop_image, num_parallel_calls=4)
            dataset = dataset.batch(self.num_batch)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_valid_dataset(self):
        if self.load_on_memory:
            dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_images))
            dataset = dataset.repeat(1)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.lr_image_filenames, self.hr_image_filenames))
            dataset = dataset.map(self._load_decode_image, num_parallel_calls=4)
            dataset = dataset.repeat(1)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_test_dataset(self):
        return self.create_valid_dataset()

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/cpu:0'):
        dataset = BigbuckbunnyV0(args, 3)
        train_dataset = dataset.create_train_dataset()
        valid_dataset = dataset.create_train_dataset()
        print(train_dataset.take(1))
        print(valid_dataset.take(1))
