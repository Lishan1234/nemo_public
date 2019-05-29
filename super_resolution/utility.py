import sys
import tensorflow as tf
import random

def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_image(hr_image, lr_image, lr_bicubic_image, scale, patch_size):
    height, width, channel = lr_image.get_shape().as_list()
    #print(height, width)
    rand_height = random.randint(0, height - patch_size - 1)
    rand_width = random.randint(0, width - patch_size - 1)
    hr_image_cropped = tf.image.crop_to_bounding_box(hr_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                   patch_size * scale,
                                                   patch_size * scale)
    lr_image_cropped = tf.image.crop_to_bounding_box(lr_image,
                                                    rand_height,
                                                    rand_width,
                                                   patch_size,
                                                   patch_size)
    lr_bicubic_image_cropped = tf.image.crop_to_bounding_box(lr_bicubic_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                   patch_size * scale,
                                                   patch_size * scale)
    return hr_image_cropped, lr_image_cropped, lr_bicubic_image_cropped

def crop_augment_image(hr_image, lr_image, scale, patch_size):
    #Crop
    height, width, channel = lr_image.get_shape().as_list()
    rand_height = random.randint(0, height - patch_size - 1)
    rand_width = random.randint(0, width - patch_size - 1)
    hr_image_cropped = tf.image.crop_to_bounding_box(hr_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                   patch_size * scale,
                                                   patch_size * scale)
    lr_image_cropped = tf.image.crop_to_bounding_box(lr_image,
                                                    rand_height,
                                                    rand_width,
                                                   patch_size,
                                                   patch_size)

    #Augement
    if random.randint(0,1):
        hr_image_cropped = tf.image.flip_left_right(hr_image_cropped)
        lr_image_cropped = tf.image.flip_left_right(lr_image_cropped)
    if random.randint(0,1):
        hr_image_cropped = tf.image.flip_up_down(hr_image_cropped)
        lr_image_cropped = tf.image.flip_up_down(lr_image_cropped)
    for _ in range(random.randint(0,3)):
        hr_image_cropped = tf.image.rot90(hr_image_cropped)
        lr_image_cropped = tf.image.rot90(lr_image_cropped)

    return hr_image_cropped, lr_image_cropped
