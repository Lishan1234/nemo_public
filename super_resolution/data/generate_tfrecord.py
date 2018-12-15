import os, glob, random, sys, time
import tensorflow as tf
from PIL import Image
import numpy as np

from option import args
import common

tf.enable_eager_execution()

#Training dataset
train_tf_records_filename = os.path.join('process', args.train_data, '{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patches))
train_writer = tf.io.TFRecordWriter(train_tf_records_filename)

train_hr_image_path = os.path.join('process', args.train_data, 'hr_x{}'.format(args.scale))
train_lr_image_path = os.path.join('process', args.train_data, 'lr_x{}'.format(args.scale))
train_lr_bicubic_image_path = os.path.join('process', args.train_data, 'lr_bicubic_x{}'.format(args.scale))

train_hr_image_filenames = glob.glob('{}/*.png'.format(train_hr_image_path))
train_lr_image_filenames = glob.glob('{}/*.png'.format(train_lr_image_path))
train_lr_bicubic_image_filenames = glob.glob('{}/*.png'.format(train_lr_bicubic_image_path))

train_hr_images = []
train_lr_images = []
train_lr_bicubic_images = []

for lr_filename, lr_bicubic_filename, hr_filename in zip(train_lr_image_filenames, train_lr_bicubic_image_filenames, train_hr_image_filenames):
    with tf.device('cpu:0'):
        train_hr_images.append(common.load_image(hr_filename))
        train_lr_images.append(common.load_image(lr_filename))
        train_lr_bicubic_images.append(common.load_image(lr_bicubic_filename))

assert len(train_lr_images) == len(train_hr_images) == len(train_lr_bicubic_images)
assert len(train_lr_images) != 0
print('dataset length: {}'.format(len(train_lr_images)))

count = 0
while count < args.num_patches:
    rand_idx = random.randint(0, len(train_lr_images) - 1)
    height, width, channel = train_lr_images[rand_idx].get_shape().as_list()
    if height < (args.patch_size + 1) or width < (args.patch_size + 1):
        continue
    else:
        count += 1

    if count == 1:
        start_time = time.time()
    elif count % 1000 == 0:
        print('Train TFRecord Process status: [{}/{}] / Take {} seconds'.format(count, args.num_patches, time.time() - start_time))
        start_time = time.time()

    hr_image, lr_image, lr_bicubic_image = common.crop_augment_image(train_hr_images[rand_idx], train_lr_images[rand_idx], train_lr_bicubic_images[rand_idx], args.scale, args.patch_size)

    if args.enable_debug:
        if hr_image.get_shape().as_list()[-1] == 1:
            Image.fromarray(np.uint8(tf.squeeze(hr_image).numpy()*255), mode='L').save('hr.png')
            Image.fromarray(np.uint8(tf.squeeze(lr_image).numpy()*255), mode='L').save('lr.png')
            Image.fromarray(np.uint8(tf.squeeze(lr_bicubic_image).numpy()*255), mode='L').save('lr_bicubic.png')
        else:
            Image.fromarray(np.uint8(hr_image.numpy()*255)).save('hr.png')
            Image.fromarray(np.uint8(lr_image.numpy()*255)).save('lr.png')
            Image.fromarray(np.uint8(lr_bicubic_image.numpy()*255)).save('lr_bicubic.png')
        sys.exit()

    hr_binary_image = hr_image.numpy().tostring()
    lr_binary_image = lr_image.numpy().tostring()
    lr_bicubic_binary_image = lr_bicubic_image.numpy().tostring()

    feature = {
        'hr_image_raw': common._bytes_feature(hr_binary_image),
        'lr_image_raw': common._bytes_feature(lr_binary_image),
        'lr_bicubic_image_raw': common._bytes_feature(lr_bicubic_binary_image),
        'patch_size': common._int64_feature(args.patch_size),
        'scale': common._int64_feature(args.scale),
        'channel': common._int64_feature(hr_image.get_shape().as_list()[-1]),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    train_writer.write(tf_example.SerializeToString())
train_writer.close()

#Validation dataset
valid_tf_records_filename = os.path.join('process', args.valid_data, '{}_{}_{}_valid.tfrecords'.format(args.train_data, args.patch_size, args.num_patches))
valid_writer = tf.io.TFRecordWriter(valid_tf_records_filename)

valid_hr_image_path = os.path.join('process', args.valid_data, 'hr_x{}'.format(args.scale))
valid_lr_image_path = os.path.join('process', args.valid_data, 'lr_x{}'.format(args.scale))
valid_lr_bicubic_image_path = os.path.join('process', args.valid_data, 'lr_bicubic_x{}'.format(args.scale))

valid_hr_image_filenames = glob.glob('{}/*.png'.format(valid_hr_image_path))
valid_lr_image_filenames = glob.glob('{}/*.png'.format(valid_lr_image_path))
valid_lr_bicubic_image_filenames = glob.glob('{}/*.png'.format(valid_lr_bicubic_image_path))

valid_hr_images = []
valid_lr_images = []
valid_lr_bicubic_images = []

for lr_filename, lr_bicubic_filename, hr_filename in zip(valid_lr_image_filenames, valid_lr_bicubic_image_filenames, valid_hr_image_filenames):
    with tf.device('cpu:0'):
        valid_hr_images.append(common.load_image(hr_filename))
        valid_lr_images.append(common.load_image(lr_filename))
        valid_lr_bicubic_images.append(common.load_image(lr_bicubic_filename))

assert len(valid_lr_images) == len(valid_hr_images) == len(valid_lr_bicubic_images)
assert len(valid_lr_images) != 0
print('dataset length: {}'.format(len(valid_lr_images)))

for i in range(len(valid_lr_images)):
    if i % 1 == 0:
        print('Valid TFRecord Process status: [{}/{}]'.format(i+1, len(valid_lr_images)))

    hr_image, lr_image, lr_bicubic_image = valid_hr_images[i], valid_lr_images[i], valid_lr_bicubic_images[i]

    hr_binary_image = hr_image.numpy().tostring()
    lr_binary_image = lr_image.numpy().tostring()
    lr_bicubic_binary_image = lr_bicubic_image.numpy().tostring()
    lr_image_shape = lr_image.get_shape().as_list()

    feature = {
        'hr_image_raw': common._bytes_feature(hr_binary_image),
        'lr_image_raw': common._bytes_feature(lr_binary_image),
        'lr_bicubic_image_raw': common._bytes_feature(lr_bicubic_binary_image),
        'height': common._int64_feature(lr_image_shape[0]),
        'width': common._int64_feature(lr_image_shape[1]),
        'channel': common._int64_feature(hr_image.get_shape().as_list()[-1]),
        'scale': common._int64_feature(args.scale),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    valid_writer.write(tf_example.SerializeToString())
valid_writer.close()
