import cv2
import tensorflow as tf

def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    #img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def load_image_float(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

if __name__ == "__main__":
    tf.enable_eager_execution()

    img = load_image('1.png')
    print(img.dtype)
    print(img)
    print(img.numpy())
