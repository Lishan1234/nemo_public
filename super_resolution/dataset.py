import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import os, glob, sys

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
        self.train_tfrecord_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}_{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch, args.scale))
        self.valid_tfrecord_path = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}_{}_valid.tfrecords'.format(args.valid_data, args.scale))
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

class ImageDataset():
    def __init__(self, args):
        #TODO: option for lazy loading approach
        hr_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/original'.format(args.hr))
        lr_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/original'.format(args.lr))
        lr_bicubic_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/bicubic_{}p'.format(args.lr, args.hr))
        print(hr_image_path)

        hr_image_filenames = glob.glob('{}/*.png'.format(hr_image_path))
        lr_image_filenames = glob.glob('{}/*.png'.format(lr_image_path))
        lr_bicubic_image_filenames = glob.glob('{}/*.png'.format(lr_bicubic_image_path))

        self.hr_images = []
        self.lr_images = []
        self.lr_bicubic_images = []

        with tf.device('cpu:0'):
            for lr_filename, lr_bicubic_filename, hr_filename in zip(lr_image_filenames, lr_bicubic_image_filenames, hr_image_filenames):
                self.hr_images.append(load_image(hr_filename))
                self.lr_images.append(load_image(lr_filename))
                self.lr_bicubic_images.append(load_image(lr_bicubic_filename))

        assert len(self.hr_images) != 0
        assert len(self.lr_images) != 0
        assert len(self.lr_bicubic_images) != 0

    def get_length(self):
        return len(self.hr_images)

    def create_dataset(self, num_sample=None):
        dataset = tf.data.Dataset.from_tensor_slices((self.lr_images, self.hr_images, self.lr_bicubic_images))
        dataset = dataset.repeat(1)
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
