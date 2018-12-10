import tensorflow as tf
import os, glob, random, sys
from PIL import Image

import ops
sys.path.append('..')
from option import args

tf.enable_eager_execution()

# Serialize images, together with labels, to TF records
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_image(lr_image, hr_image):
    #crop
    lr_image_cropped = tf.image.random_crop(lr_image, [args.patch_size, args.patch_size, 3])
    hr_image_cropped = tf.image.random_crop(hr_image, [args.patch_size * 3, args.patch_size * 3, 3])

    return lr_image_cropped, hr_image_cropped

lr_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'lr_x{}'.format(3))
hr_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'hr')
lr_image_filenames = glob.glob('{}/*.png'.format(lr_image_path))
hr_image_filenames = glob.glob('{}/*.png'.format(hr_image_path))

hr_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'hr')
lr_train_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'lr_x{}_train'.format(3))
hr_train_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'hr_train')
lr_test_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'lr_x{}_test'.format(3))
hr_test_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'hr_test')
hr_bicubic_test_image_path = os.path.join('data', args.data_name, '{}fps'.format(str(args.fps)), 'hr_biubic_test')
os.makedirs(lr_train_image_path, exist_ok=True)
os.makedirs(hr_train_image_path, exist_ok=True)
os.makedirs(lr_test_image_path, exist_ok=True)
os.makedirs(hr_test_image_path, exist_ok=True)
os.makedirs(hr_bicubic_test_image_path, exist_ok=True)

lr_images = []
hr_images = []

for lr_filename, hr_filename in zip(lr_image_filenames, hr_image_filenames):
    lr_images.append(ops.load_image(lr_filename))
    hr_images.append(ops.load_image(hr_filename))

assert len(lr_images) == len(hr_images)
assert len(lr_images) != 0

print('dataset length: {}'.format(len(lr_images)))

#Generate training tfrecord
for i in range(args.num_patch):
    if i % 1000 == 0:
        print('Train Process status: [{}/{}]'.format(i, args.num_patch))

    rand_idx = random.randint(0, len(lr_images) - 1)
    lr_image, hr_image = crop_image(lr_images[rand_idx], hr_images[rand_idx])

    lr_image = lr_image.numpy()
    hr_image = hr_image.numpy()

    Image.fromarray(lr_image).save('{}/{}.png'.format(lr_train_image_path, i))
    Image.fromarray(hr_image).save('{}/{}.png'.format(hr_train_image_path, i))

#Generate testing tfrecord
for i in range(len(lr_images)):
    if i % 10 == 0:
        print('Test Process status: [{}/{}]'.format(i, len(lr_images)))

    lr_image = lr_images[i]
    hr_image = hr_images[i]
    hr_bicubic_image = tf.image.resize_bicubic(tf.expand_dims(lr_image, 0), (tf.shape(hr_image)[0], tf.shape(hr_image)[1]))
    hr_bicubic_image = tf.squeeze(hr_bicubic_image)

    lr_encode_image = tf.image.encode_png(lr_image)
    hr_encode_image = tf.image.encode_png(hr_image)
    hr_bicubic_encode_image = tf.image.encode_png(hr_image)

    with open('{}/{}.png'.format(lr_test_image_path, i), 'w') as f:
        f.write(lr_encode_image)

    with open('{}/{}.png'.format(hr_image_path, i), 'w') as f:
        f.write(hr_encode_image)

    with open('{}/{}.png'.format(hr_bicubic_image_path, i), 'w') as f:
        f.write(hr_bicubic_encode_image)
