import tensorflow as tf
import os, glob, random, sys
from PIL import Image
import numpy as np

import ops
sys.path.append('..')
from option import args

tf.enable_eager_execution()

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_image(lr_image, hr_image, hr_bicubic_image, scale):
    height, width, channel = lr_image.get_shape().as_list()
    rand_height = random.randint(0, height - args.patch_size - 1)
    rand_width = random.randint(0, width - args.patch_size - 1)
    lr_image_cropped = tf.image.crop_to_bounding_box(lr_image,
                                                    rand_height,
                                                    rand_width,
                                                    args.patch_size,
                                                    args.patch_size)
    hr_image_cropped = tf.image.crop_to_bounding_box(hr_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                    args.patch_size * scale,
                                                    args.patch_size * scale)
    hr_bicubic_image_cropped = tf.image.crop_to_bounding_box(hr_bicubic_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                    args.patch_size * scale,
                                                    args.patch_size * scale)
    return lr_image_cropped, hr_image_cropped, hr_bicubic_image_cropped

train_tf_records_filename = os.path.join('data', args.data_name, '{}_train.tfrecords'.format(args.data_name))
test_tf_records_filename = os.path.join('data', args.data_name, '{}_test.tfrecords'.format(args.data_name))
train_writer = tf.io.TFRecordWriter(train_tf_records_filename)
test_writer = tf.io.TFRecordWriter(test_tf_records_filename)

lr_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), '240p_bicubic')
hr_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), '1080p')
hr_bicubic_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), '1080p_bicubic')

lr_image_filenames = glob.glob('{}/*.png'.format(lr_image_path))
hr_image_filenames = glob.glob('{}/*.png'.format(hr_image_path))
hr_bicubic_image_filenames = glob.glob('{}/*.png'.format(hr_bicubic_image_path))

lr_images = []
hr_images = []
hr_bicubic_images= []

for lr_filename, hr_filename, hr_bicubic_filename in zip(lr_image_filenames, hr_image_filenames, hr_bicubic_image_filenames):
    lr_images.append(ops.load_image(lr_filename))
    hr_images.append(ops.load_image(hr_filename))
    hr_bicubic_images.append(ops.load_image(hr_bicubic_filename))

assert len(lr_images) == len(hr_images) == len(hr_bicubic_images)
assert len(lr_images) != 0

print('dataset length: {}'.format(len(lr_images)))

#Generate training tfrecord
for i in range(args.num_patch):
    if i % 1000 == 0:
        print('Train TFRecord Process status: [{}/{}]'.format(i, args.num_patch))

    rand_idx = random.randint(0, len(lr_images) - 1)
    lr_image, hr_image, hr_bicubic_image = crop_image(lr_images[rand_idx], hr_images[rand_idx], hr_bicubic_images[rand_idx], 4)
    """for testing
    Image.fromarray(np.uint8(lr_image.numpy()*255)).save('lr.png')
    Image.fromarray(np.uint8(hr_image.numpy()*255)).save('hr.png')
    Image.fromarray(np.uint8(hr_bicubic_image.numpy()*255)).save('hr_bicubic.png')
    sys.exit()
    """
    lr_image_shape = tf.shape(lr_image)
    hr_image_shape = tf.shape(hr_image)
    lr_binary_image = lr_image.numpy().tostring()
    hr_binary_image = hr_image.numpy().tostring()

    feature = {
        'lr_image_raw': _bytes_feature(lr_binary_image),
        'lr_height': _int64_feature(lr_image_shape[0]),
        'lr_width': _int64_feature(lr_image_shape[1]),
        'hr_image_raw': _bytes_feature(hr_binary_image),
        'hr_height': _int64_feature(hr_image_shape[0]),
        'hr_width': _int64_feature(hr_image_shape[1]),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    train_writer.write(tf_example.SerializeToString())
train_writer.close()

#Generate testing tfrecord
for i in range(len(lr_images)):
    if i % 10 == 0:
        print('Test TFRecord Process status: [{}/{}]'.format(i, len(lr_images)))

    lr_image = lr_images[i]
    hr_image = hr_images[i]
    hr_bicubic_image = hr_bicubic_images[i]

    lr_image_shape = tf.shape(lr_image)
    hr_image_shape = tf.shape(hr_image)
    lr_binary_image = lr_image.numpy().tostring()
    hr_binary_image = hr_image.numpy().tostring()
    hr_bicubic_binary_image = hr_bicubic_image.numpy().tostring()

    feature = {
        'lr_image_raw': _bytes_feature(lr_binary_image),
        'lr_height': _int64_feature(lr_image_shape[0]),
        'lr_width': _int64_feature(lr_image_shape[1]),
        'hr_image_raw': _bytes_feature(hr_binary_image),
        'hr_height': _int64_feature(hr_image_shape[0]),
        'hr_width': _int64_feature(hr_image_shape[1]),
        'hr_bicubic_image_raw': _bytes_feature(hr_bicubic_binary_image),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    test_writer.write(tf_example.SerializeToString())
test_writer.close()
