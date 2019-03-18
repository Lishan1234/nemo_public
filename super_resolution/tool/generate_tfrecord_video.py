import os, glob, random, sys, time, argparse
import tensorflow as tf
from PIL import Image
import numpy as np
import common

parser = argparse.ArgumentParser(description='MnasNet')

parser.add_argument('--train_data', type=str, default='news')
parser.add_argument('--valid_data', type=str, default='news')
parser.add_argument('--data_root', type=str, default='../data')
parser.add_argument('--data_name', type=str, default='60_0.5')
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument('--num_patch', type=int, default=10000)
parser.add_argument('--enable_debug', action='store_true')
parser.add_argument('--scale', type=int, default=4) #for image based dataset
parser.add_argument('--hr', type=int, default=960) #for video based dataset
parser.add_argument('--bitrate', type=int, default=None) #for video based dataset

args = parser.parse_args()

tf.enable_eager_execution()

"""dataset for single HR-LR video pair"""

#Training dataset
if args.bitrate is None:
    train_tf_records_filename = os.path.join(args.data_root, args.train_data, args.data_name, '{}_{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch, args.scale))
    train_hr_image_path = os.path.join(args.data_root, args.train_data, args.data_name, '{}p/original'.format(args.hr))
    train_lr_image_path = os.path.join(args.data_root, args.train_data, args.data_name, '{}p/original'.format(args.hr//args.scale))
    train_lr_bicubic_image_path = os.path.join(args.data_root, args.train_data, args.data_name, '{}p/bicubic_{}p'.format(args.hr//args.scale, args.hr))
else:
    train_tf_records_filename = os.path.join(args.data_root, args.train_data, args.data_name, '{}_{}_{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch, args.scale, args.bitrate))
    train_hr_image_path = os.path.join(args.data_root, args.train_data, args.data_name, '{}p/original'.format(args.hr))
    train_lr_image_path = os.path.join(args.data_root, args.train_data, args.data_name, '{}p-{}k/original'.format(args.hr//args.scale, args.bitrate))
    train_lr_bicubic_image_path = os.path.join(args.data_root, args.train_data, args.data_name, '{}p-{}k/bicubic_{}p'.format(args.hr//args.scale, args.bitrate, args.hr))
train_writer = tf.io.TFRecordWriter(train_tf_records_filename)

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
while count < args.num_patch:
    rand_idx = random.randint(0, len(train_lr_images) - 1)
    height, width, channel = train_lr_images[rand_idx].get_shape().as_list()

    if height < (args.patch_size + 1) or width < (args.patch_size + 1):
        continue
    else:
        count += 1

    if count == 1:
        start_time = time.time()
    elif count % 1000 == 0:
        print('Train TFRecord Process status: [{}/{}] / Take {} seconds'.format(count, args.num_patch, time.time() - start_time))
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
if args.bitrate is None:
    valid_tf_records_filename = os.path.join(args.data_root, args.valid_data, args.data_name, '{}_{}_valid.tfrecords'.format(args.valid_data, args.scale))
else:
    valid_tf_records_filename = os.path.join(args.data_root, args.valid_data, args.data_name, '{}_{}_{}_valid.tfrecords'.format(args.valid_data, args.scale, args.bitrate))
valid_writer = tf.io.TFRecordWriter(valid_tf_records_filename)

for i in range(len(train_hr_images)):
    if i % 1 == 0:
        print('Valid TFRecord Process status: [{}/{}]'.format(i+1, len(train_hr_images)))

    hr_image, lr_image, lr_bicubic_image = train_hr_images[i], train_lr_images[i], train_lr_bicubic_images[i]

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
