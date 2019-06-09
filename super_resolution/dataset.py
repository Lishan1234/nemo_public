import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import os, glob, sys, time
import numpy as np

from option import args

def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def make_dataset(args, scale):
    return TFRecordDataset(args)

class TFRecordDataset():
    def __init__(self, args):
        self.num_batch = args.num_batch
        self.num_batch_per_epoch = args.num_batch_per_epoch
        self.train_tfrecord_path = os.path.join(args.data_dir, args.train_data, args.train_datatype, 'train_{}p.tfrecords'.format(args.target_resolution//args.scale))
        self.valid_tfrecord_path = os.path.join(args.data_dir, args.valid_data, args.valid_datatype, 'valid_{}p.tfrecords'.format(args.target_resolution//args.scale))

        """
        if args.bitrate is None:
            self.train_tfrecord_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}_{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch, args.scale))
            self.valid_tfrecord_path = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}_{}_valid.tfrecords'.format(args.valid_data, args.scale))
        else:
            self.train_tfrecord_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}_{}_{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch, args.scale, args.bitrate))
            self.valid_tfrecord_path = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}_{}_{}_valid.tfrecords'.format(args.valid_data, args.scale, args.bitrate))
        """

        print(self.train_tfrecord_path, self.valid_tfrecord_path)
        assert os.path.isfile(self.train_tfrecord_path)
        assert os.path.isfile(self.valid_tfrecord_path)

    def _train_parse_function(self, example_proto):
        features = {'hr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'lr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                #'lr_bicubic_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'patch_size': tf.FixedLenFeature((), tf.int64),
                'scale': tf.FixedLenFeature((), tf.int64),
                'channel': tf.FixedLenFeature((), tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)

        hr_image = tf.decode_raw(parsed_features['hr_image_raw'], tf.float32)
        lr_image = tf.decode_raw(parsed_features['lr_image_raw'], tf.float32)
        #lr_bicubic_image = tf.decode_raw(parsed_features['lr_bicubic_image_raw'], tf.float32)

        hr_image = tf.reshape(hr_image, [parsed_features['patch_size'] * parsed_features['scale'], parsed_features['patch_size'] * parsed_features['scale'], parsed_features['channel']])
        lr_image = tf.reshape(lr_image, [parsed_features['patch_size'], parsed_features['patch_size'], parsed_features['channel']])
        #lr_bicubic_image = tf.reshape(lr_bicubic_image, [parsed_features['patch_size'], parsed_features['patch_size'], parsed_features['channel'])

        return lr_image, hr_image

    def _valid_parse_function(self, example_proto):
        features = {'hr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'lr_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'lr_bicubic_image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'height': tf.FixedLenFeature((), tf.int64),
                'width': tf.FixedLenFeature((), tf.int64),
                'channel': tf.FixedLenFeature((), tf.int64),
                'scale': tf.FixedLenFeature((), tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)

        hr_image = tf.decode_raw(parsed_features['hr_image_raw'], tf.float32)
        lr_image = tf.decode_raw(parsed_features['lr_image_raw'], tf.float32)
        lr_bicubic_image = tf.decode_raw(parsed_features['lr_bicubic_image_raw'], tf.float32)

        hr_image = tf.reshape(hr_image, [parsed_features['height'] * parsed_features['scale'], parsed_features['width'] * parsed_features['scale'], parsed_features['channel']])
        lr_image = tf.reshape(lr_image, [parsed_features['height'], parsed_features['width'], parsed_features['channel']])
        lr_bicubic_image = tf.reshape(lr_bicubic_image, [parsed_features['height'] * parsed_features['scale'], parsed_features['width'] * parsed_features['scale'], parsed_features['channel']])

        return lr_image, hr_image, lr_bicubic_image

    def create_train_dataset(self):
        dataset = tf.data.TFRecordDataset(self.train_tfrecord_path, num_parallel_reads=1)
        dataset = dataset.cache() #temporally used for small tfrecord file
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
        dataset = dataset.map(self._train_parse_function, num_parallel_calls=2)
        dataset = dataset.batch(self.num_batch)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_valid_dataset(self, num_sample=None):
        dataset = tf.data.TFRecordDataset(self.valid_tfrecord_path)
        dataset = dataset.repeat(1)
        dataset = dataset.map(self._valid_parse_function)
        dataset = dataset.batch(1)

        return dataset

def _parse_feature_func(lr_image_decoded, hr_filename, lr_bicubic_filename):
    hr_image_decoded =  tf.io.read_file(hr_filename)
    hr_image_decoded = tf.image.decode_image(hr_image_decoded)
    hr_image_decoded = tf.image.convert_image_dtype(hr_image_decoded, tf.float32)

    lr_bicubic_image_decoded =  tf.io.read_file(lr_bicubic_filename)
    lr_bicubic_image_decoded = tf.image.decode_image(lr_bicubic_image_decoded)
    lr_bicubic_image_decoded = tf.image.convert_image_dtype(lr_bicubic_image_decoded, tf.float32)

    return lr_image_decoded, hr_image_decoded, lr_bicubic_image_decoded

#TODO: args.hr, args.lr are deprecated, args.data_type
class FeatureDataset():
    def __init__(self, args, model_name):
        #TODO: option for lazy loading approach
        hr_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/original'.format(args.target_resolution))
        if args.bitrate is None:
            lr_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/feature'.format(args.target_resolution//args.scale), model_name)
            lr_bicubic_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/bicubic_{}p'.format(args.target_resolution//args.scale, args.target_resolution))
        else:
            lr_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p-{}k/feature'.format(args.target_resolution//args.scale, args.bitrate))
            lr_bicubic_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p-{}k/bicubic_{}p'.format(args.target_resolution//args.scale, args.bitrate, args.target_resolution))

        self.hr_image_filenames = sorted(glob.glob('{}/*.png'.format(hr_image_path)))
        self.lr_image_filenames = sorted(glob.glob('{}/*.npy'.format(lr_image_path)))
        self.lr_bicubic_image_filenames = sorted(glob.glob('{}/*.png'.format(lr_bicubic_image_path)))

        self.lr_images = []
        for lr_image_filename in self.lr_image_filenames:
            self.lr_images.append(np.load(lr_image_filename))

        #print(self.lr_images[0])
        #print(self.hr_image_filenames)
        #print(self.lr_image_filenames)
        #print(self.lr_bicubic_image_filenames)

        assert len(self.hr_image_filenames) != 0
        assert len(self.lr_image_filenames) != 0
        assert len(self.lr_bicubic_image_filenames) != 0

        """ @deprecated: memory explosion problem occured
        self.hr_images = []
        self.lr_images = []
        self.lr_bicubic_images = []

        with tf.device('cpu:0'):
            for lr_filename, lr_bicubic_filename, hr_filename in zip(lr_image_filenames, lr_bicubic_image_filenames, hr_image_filenames):
                self.hr_images.append(load_image(hr_filename))
                self.lr_images.append(load_image(lr_filename))
                self.lr_bicubic_images.append(load_image(lr_bicubic_filename))
        """

    def get_length(self):
        #return len(hr_images)
        return len(self.hr_image_filenames)

    def create_dataset(self, num_sample=None):
        #dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_images, self.lr_bicubic_images))

        dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_image_filenames, self.lr_bicubic_image_filenames))
        dataset = dataset.repeat(1)
        dataset = dataset.map(_parse_feature_func, num_parallel_calls=2)
        dataset = dataset.batch(1)

        return dataset

def _single_parse_func(filename):
    image_decoded = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_decoded)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image_decoded

class InferenceDataset():
    def __init__(self, args):
        image_path = os.path.join(args.data_dir, args.valid_data, args.valid_datatype, '{}p'.format(args.target_resolution//args.scale))
        self.image_filenames = sorted(glob.glob('{}/*.png'.format(image_path)))
        #print(image_path)
        #print(self.image_filenames)
        #print(len(self.image_filenames))
        assert len(self.image_filenames) != 0

    def get_length(self):
        return len(self.image_filenames)

    def create_dataset(self, num_sample=None):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_filenames))
        dataset = dataset.repeat(1)
        dataset = dataset.map(_single_parse_func, num_parallel_calls=4)
        dataset = dataset.batch(1)
        return dataset

def _multiple_parse_func(lr_filename, hr_filename, lr_bicubic_filename):
    lr_image_decoded = tf.read_file(lr_filename)
    lr_image_decoded = tf.image.decode_image(lr_image_decoded)
    lr_image_decoded = tf.image.convert_image_dtype(lr_image_decoded, tf.float32)

    hr_image_decoded =  tf.io.read_file(hr_filename)
    hr_image_decoded = tf.image.decode_image(hr_image_decoded)
    hr_image_decoded = tf.image.convert_image_dtype(hr_image_decoded, tf.float32)

    lr_bicubic_image_decoded =  tf.io.read_file(lr_bicubic_filename)
    lr_bicubic_image_decoded = tf.image.decode_image(lr_bicubic_image_decoded)
    lr_bicubic_image_decoded = tf.image.convert_image_dtype(lr_bicubic_image_decoded, tf.float32)

    return lr_image_decoded, hr_image_decoded, lr_bicubic_image_decoded

class TrainingDataset():
    def __init__(self, args):
        hr_image_path = os.path.join(args.data_dir, args.valid_data, args.valid_datatype, '{}p_lossless'.format(args.target_resolution))
        lr_image_path = os.path.join(args.data_dir, args.valid_data, args.valid_datatype, '{}p'.format(args.target_resolution//args.scale))
        lr_bicubic_image_path = os.path.join(args.data_dir, args.valid_data, args.valid_datatype, '{}p_{}p_bicubic'.format(args.target_resolution, args.target_resolution//args.scale))
        self.hr_image_filenames = sorted(glob.glob('{}/*.png'.format(hr_image_path)))
        self.lr_image_filenames = sorted(glob.glob('{}/*.png'.format(lr_image_path)))
        self.lr_bicubic_image_filenames = sorted(glob.glob('{}/*.png'.format(lr_bicubic_image_path)))

        assert len(self.hr_image_filenames) != 0
        assert len(self.hr_image_filenames) == len(self.lr_image_filenames) == len(self.lr_bicubic_image_filenames)

    def get_length(self):
        return len(self.hr_image_filenames)

    def create_dataset(self, num_sample=None):
        dataset = tf.data.Dataset.from_tensor_slices((self.lr_image_filenames, self.hr_image_filenames, self.lr_bicubic_image_filenames))
        dataset = dataset.repeat(1)
        dataset = dataset.map(_multiple_parse_func, num_parallel_calls=4)
        dataset = dataset.batch(1)
        return dataset

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/cpu:0'):
        dataset = TFRecordDataset(args)
        train_dataset = dataset.create_train_dataset()
        valid_dataset = dataset.create_valid_dataset()

        for batch in train_dataset.take(1):
            print(tf.shape(batch[0]))
            print(tf.shape(batch[1]))
        for batch in valid_dataset.take(1):
            print(tf.shape(batch[0]))
            print(tf.shape(batch[1]))
            print(tf.shape(batch[2]))
