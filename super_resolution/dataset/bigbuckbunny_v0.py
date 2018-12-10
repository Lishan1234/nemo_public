import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import os, glob
from dataset import ops
from dataset import mnasdataset

"""for testing
import ops
import mnasdataset
import sys
sys.path.append('..')
from option import args
"""

def make_dataset(args, scale):
    return BigbuckbunnyV0_tfrecord(args, scale)

class BigbuckbunnyV0_tfrecord(mnasdataset.MnasDataset):
    def __init__(self, args, scale):
        self.load_on_memory = args.load_on_memory
        self.num_batch = args.num_batch
        self.train_tfrecord_path = os.path.join(args.data_dir, args.data_name, '{}_train.tfrecords'.format(args.data_name))
        self.test_tfrecord_path= os.path.join(args.data_dir, args.data_name, '{}_test.tfrecords'.format(args.data_name))
        """
        self.train_tfrecord_path = os.path.join('data', args.data_name, '{}_train.tfrecords'.format(args.data_name))
        self.test_tfrecord_path= os.path.join('data', args.data_name, '{}_test.tfrecords'.format(args.data_name))
        """
        assert os.path.isfile(self.train_tfrecord_path)
        assert os.path.isfile(self.test_tfrecord_path)

    def _train_parse_function(self, example_proto):
        features = {'lr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'hr_image_raw': tf.FixedLenFeature((), tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)

        lr_image = tf.decode_raw(parsed_features['lr_image_raw'], tf.float32)
        hr_image = tf.decode_raw(parsed_features['hr_image_raw'], tf.float32)

        lr_image = tf.reshape(lr_image, [48, 48, 3]) #TODO: replace by read shape
        hr_image = tf.reshape(hr_image, [144, 144, 3])

        return lr_image, hr_image

    def _test_parse_function(self, example_proto):
        features = {'lr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'hr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'hr_bicubic_image_raw': tf.FixedLenFeature((), tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)

        lr_image = tf.decode_raw(parsed_features['lr_image_raw'], tf.float32)
        hr_bicubic_image = tf.decode_raw(parsed_features['hr_image_raw'], tf.float32)
        hr_image = tf.decode_raw(parsed_features['hr_bicubic_image_raw'], tf.float32)

        lr_image = tf.reshape(lr_image, [48, 48, 3]) #TODO: replace by read shape
        hr_bicubic_image = tf.reshape(hr_bicubic_image, [144, 144, 3])
        hr_image = tf.reshape(hr_image, [144, 144, 3])

        return lr_image, hr_image, hr_bicubic_image

    def create_train_dataset(self):
        dataset = tf.data.TFRecordDataset(self.train_tfrecord_path, num_parallel_reads=4)
        dataset = dataset.map(self._train_parse_function, num_parallel_calls=4)

        if self.load_on_memory:
            dataset = dataset.cache()

        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(None)
        dataset = dataset.batch(self.num_batch)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_test_dataset(self):
        dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
        dataset = dataset.map(self._test_parse_function, num_parallel_calls=4)

        if self.load_on_memory:
            dataset = dataset.cache()

        dataset = dataset.repeat(None)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset

class BigbuckbunnyV0_preprocess(mnasdataset.MnasDataset):
    def __init__(self, args, scale):
        self.patch_size = args.patch_size
        self.num_batch = args.num_batch
        self.scale = scale

        lr_image_path = os.path.join(args.data_dir, args.data_name, '{}fps'.format(str(args.fps)), 'lr_x{}_train'.format(scale))
        hr_image_path = os.path.join(args.data_dir, args.data_name, '{}fps'.format(str(args.fps)), 'hr_train')
        """for testing
        lr_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'lr_x{}_train'.format(scale))
        hr_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'hr_train')
        """
        self.lr_image_filenames = glob.glob('{}/*.png'.format(lr_image_path))
        self.hr_image_filenames = glob.glob('{}/*.png'.format(hr_image_path))

        assert len(self.lr_image_filenames) == len(self.hr_image_filenames)
        assert len(self.lr_image_filenames) != 0

        self.lr_images = []
        self.hr_images = []

        for lr_filename, hr_filename in zip(self.lr_image_filenames, self.hr_image_filenames):
            self.lr_images.append(ops.load_image_float(lr_filename))
            self.hr_images.append(ops.load_image_float(hr_filename))

    def create_train_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_images))
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(None)
        dataset = dataset.batch(self.num_batch)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_test_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_images))
        dataset = dataset.repeat(None)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

class BigbuckbunnyV0(mnasdataset.MnasDataset):
    def __init__(self, args, scale):
        self.load_on_memory = args.load_on_memory
        self.patch_size = args.patch_size
        self.num_batch = args.num_batch
        self.scale = scale

        lr_image_path = os.path.join(args.data_dir, args.data_name, '{}fps'.format(str(args.fps)), 'lr_x{}'.format(scale))
        hr_image_path = os.path.join(args.data_dir, args.data_name, '{}fps'.format(str(args.fps)), 'hr')
        self.lr_image_filenames = glob.glob('{}/*.png'.format(lr_image_path))
        self.hr_image_filenames = glob.glob('{}/*.png'.format(hr_image_path))

        assert len(self.lr_image_filenames) == len(self.hr_image_filenames)
        assert len(self.lr_image_filenames) != 0

        if self.load_on_memory:
            self.lr_images = []
            self.hr_images = []

            for lr_filename, hr_filename in zip(self.lr_image_filenames, self.hr_image_filenames):
                self.lr_images.append(ops.load_image(lr_filename))
                self.hr_images.append(ops.load_image(hr_filename))

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
            lr_image_decoded = tf.image.decode_image(lr_image_string)
            lr_image_decoded = tf.image.convert_image_dtype(lr_image_decoded, tf.float32)
            hr_image_decoded = tf.image.decode_image(hr_image_string)
            hr_image_decoded = tf.image.convert_image_dtype(hr_image_decoded, tf.float32)

            return lr_image_decoded, hr_image_decoded

    def _load_decode_crop_image(self, lr_filename, hr_filename):
            #load
            lr_image_string = tf.io.read_file(lr_filename)
            hr_image_string = tf.io.read_file(hr_filename)

            #decode
            lr_image_decoded = tf.image.decode_image(lr_image_string)
            lr_image_decoded = tf.image.convert_image_dtype(lr_image_decoded, tf.float32)
            hr_image_decoded = tf.image.decode_image(hr_image_string)
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
            dataset = dataset.shuffle(10000)
            dataset = dataset.repeat(None)
            dataset = dataset.map(self._load_decode_crop_image, num_parallel_calls=4)
            dataset = dataset.batch(self.num_batch)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_test_dataset(self):
        if self.load_on_memory:
            dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_images))
            dataset = dataset.repeat(None)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.lr_image_filenames, self.hr_image_filenames))
            dataset = dataset.map(self._load_decode_image, num_parallel_calls=4)
            dataset = dataset.repeat(None)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/cpu:0'):
        dataset = BigbuckbunnyV0_preprocess(args, 3)
        train_dataset = dataset.create_train_dataset()
        valid_dataset = dataset.create_test_dataset()

        for batch in train_dataset.take(1):
            print(tf.shape(batch[0]))
            print(tf.shape(batch[1]))
