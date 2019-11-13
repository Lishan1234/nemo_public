import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from model.common import NormalizeConfig

NETWORK_NAME = 'EDSR_ED_S'

def residual_block(x_in, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x_in)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.Add()([x_in, x])
    return x

#TODO: use tf.name_scope()
class EDSR_ED_S():
    def __init__(self, normalize_config):
        self.normalize_config = normalize_config

    @staticmethod
    def _encoder(x_in, num_filters, num_blocks):
        #feature extraction
        x = b = layers.Conv2D(num_filters, 3, padding='same')(x_in)
        for i in range(num_blocks):
            b = residual_block(b, num_filters)
        b = layers.Conv2D(num_filters, 3, padding='same')(b)
        x = layers.Add()([x, b])

        #feature reduction
        x = layers.Conv2D(3, 3, padding='same')(x)
        return x

    @staticmethod
    def _decoder(x_in, num_filters, num_blocks, scale):
        with tf.variable_scope('decoder'):
        #feature expansion
        x = b = layers.Conv2D(num_filters, 3, padding='same')(x_in)

        #feature extraction
        for i in range(num_blocks):
            b = residual_block(b, num_filters)
        b = layers.Conv2D(num_filters, 3, padding='same')(b)
        x = layers.Add()([x, b])

        #feature upscaling
        x = layers.Conv2DTranspose(3, 5, scale, padding='same')(x)
        return x

    def build_encoder(self, num_filters, num_blocks):
        name = 'ENC'
        name += '_B{}'.format(num_blocks)
        name += '_F{}'.format(num_filters)

        x_in = layers.Input(shape=(None, None, 3))
        if self.normalize_config : x_in = layers.Lambda(self.normalize_config.normalize)(x_in)
        x = self._encoder(x_in, num_filters, num_blocks)

        model = Model(inputs=x_in, outputs=x, name=name)
        return model

    def build_decoder(self, num_filters, num_blocks, scale):
        name = 'DEC'
        name += '_B{}'.format(num_blocks)
        name += '_F{}'.format(num_filters)
        name += '_S{}'.format(scale)

        x_in = layers.Input(shape=(None, None, 3))
        _ = self._encoder(x_in, enc_num_filters, enc_num_blocks) #caution: just for correct variable names
        x = self._decoder(x_in, num_filters, num_blocks, scale)
        if normalize_config : x = layers.Lambda(normalize_config.denormalize)(x)

        model = Model(inputs=x_in, outputs=x, name=name)
        return model

    def build_model(self, enc_num_filters, enc_num_blocks, dec_num_filters, dec_num_blocks, scale):
        name = NETWORK_NAME
        name += '_B{}'.format(enc_num_blocks)
        name += '_F{}'.format(enc_num_filters)
        name += '_B{}'.format(dec_num_blocks)
        name += '_F{}'.format(dec_num_filters)
        name += '_S{}'.format(scale)

        x_in = layers.Input(shape=(None, None, 3))
        x_feature = self._encoder(x_in, enc_num_filters, enc_num_blocks)
        x = self._decoder(x_feature, dec_num_filters, dec_num_blocks, scale)
        if normalize_config : x = layers.Lambda(normalize_config.denormalize)(x)

        model = Model(inputs=x_in, outputs=[x_feature, x], name=name)

        return model

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/gpu:0'):
        normalize_config = NormalizeConfig('normalize_01', 'denormalize_01')
        edsr_ed_s = EDSR_ED_S(None)
        model = edsr_ed_s.build_model(64, 8, 16, 1, 4)
        input_tensor = tf.random.uniform((1, 200, 200, 3), 0, 255)
        feature_tensor, output_tensor = model(input_tensor)
        print(feature_tensor.shape)
        print(output_tensor.shape)
