import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from common import NormalizeConfig

NETWORK_NAME = 'EDSR_S'

def residual_block(x_in, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x_in)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.Add()([x_in, x])
    return x

def model(num_blocks, num_filters, scale, normalize_config=None):
    #name
    name = NETWORK_NAME
    name += '_B{}'.format(num_blocks)
    name += '_F{}'.format(num_filters)
    name += '_S{}'.format(scale)

    #model
    x_in = layers.Input(shape=(None, None, 3))
    if normalize_config : x = layers.Lambda(normalize_config.normalize)(x_in)
    else: x = x_in

    x = b = layers.Conv2D(num_filters, 3, padding='same')(x_in)
    for i in range(num_blocks):
        b = residual_block(b, num_filters)
    b = layers.Conv2D(num_filters, 3, padding='same')(b)
    x = layers.Add()([x, b])

    x = layers.Conv2DTranspose(3, 5, scale, padding='same')(x)

    if normalize_config : x = layers.Lambda(normalize_config.denormalize)(x)

    model = Model(inputs=x_in, outputs=x, name=name)

    return model

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/gpu:0'):
        normalize_config = NormalizeConfig('normalize_01', 'denormalize_01')
        model = model(4, 32, 4, normalize_config)
        input_tensor = tf.random.uniform((1, 200, 200, 3), 0, 255)
        output_tensor = model(input_tensor)
        print(output_tensor.shape)