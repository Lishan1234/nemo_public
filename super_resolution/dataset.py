import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import os, glob

from option import args

def make_dataset(args, scale):
    return TFRecordDataset(args)

class TFRecordDataset():
    def __init__(self, args):
        self.num_batch = args.num_batch
        self.num_batch_per_epoch = args.num_batch_per_epoch
        self.train_tfrecord_path = os.path.join(args.data_dir, args.train_data, '{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch))
        self.valid_tfrecord_path = os.path.join(args.data_dir, args.valid_data, '{}_valid.tfrecords'.format(args.valid_data))
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
        dataset = tf.data.TFRecordDataset(self.train_tfrecord_path, num_parallel_reads=4)
        dataset = dataset.map(self._train_parse_function, num_parallel_calls=4)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(self.num_batch_per_epoch)
        dataset = dataset.batch(self.num_batch)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def create_valid_dataset(self, num_sample=None):
        dataset = tf.data.TFRecordDataset(self.valid_tfrecord_path)
        dataset = dataset.map(self._valid_parse_function, num_parallel_calls=4)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

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
