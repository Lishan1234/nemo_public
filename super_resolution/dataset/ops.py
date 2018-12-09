import cv2
import tensorflow as tf

def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    return img

if __name__ == "__main__":
    tf.enable_eager_execution()

    img = load_image_('1.png')
    print(img.dtype)
    print(img)
