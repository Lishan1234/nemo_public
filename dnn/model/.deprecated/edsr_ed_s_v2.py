import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

from model.common import NormalizeConfig

NETWORK_NAME = 'EDSR_ED_S_V2'

def residual_block(x_in, num_filters, name_func):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu', name=name_func())(x_in)
    x = layers.Conv2D(num_filters, 3, padding='same', name=name_func())(x)
    x = layers.Add()([x_in, x])
    return x

class EDSR_ED_S_V2():
    def __init__(self, enc_num_blocks, enc_num_filters, \
            dec_num_blocks, dec_num_filters, feature_dims, scale, normalize_config, tag=None):
        self.enc_num_blocks = enc_num_blocks
        self.enc_num_filters = enc_num_filters
        self.dec_num_blocks = dec_num_blocks
        self.dec_num_filters = dec_num_filters
        self.feature_dims = feature_dims
        self.scale = scale
        self.normalize_config = normalize_config

        self.name = NETWORK_NAME
        self.name += '_B{}'.format(self.enc_num_blocks)
        self.name += '_F{}'.format(self.enc_num_filters)
        self.name += '_D{}'.format(self.feature_dims)
        self.name += '_B{}'.format(self.dec_num_blocks)
        self.name += '_F{}'.format(self.dec_num_filters)
        self.name += '_S{}'.format(self.scale)
        if tag is not None:
            self.name += '_{}'.format(tag)

        self.enc_conv_idx = 0
        self.dec_conv_idx = 2 * self.enc_num_blocks + 3

    def _enc_conv_name(self):
        if self.enc_conv_idx == 0:
            name = 'conv2d'
        else:
            name = 'conv2d_{}'.format(self.enc_conv_idx)
        self.enc_conv_idx += 1
        return name

    def _dec_conv_name(self):
        if self.dec_conv_idx == 0:
            name = 'conv2d'
        else:
            name = 'conv2d_{}'.format(self.dec_conv_idx)
        self.dec_conv_idx += 1
        return name

    def _encoder(self, x_in):
        #feature extraction
        x = b = layers.Conv2D(self.enc_num_filters, 3, padding='same', name=self._enc_conv_name())(x_in)
        for i in range(self.enc_num_blocks):
            b = residual_block(b, self.enc_num_filters, self._enc_conv_name)
        b = layers.Conv2D(self.enc_num_filters, 3, padding='same', name=self._enc_conv_name())(b)
        x = layers.Add()([x, b])

        #feature reduction
        x = layers.Conv2D(self.feature_dims, 3, padding='same', name=self._enc_conv_name())(x)

        return x

    def _decoder(self, x_in):
        #feature expansion
        x = b = layers.Conv2D(self.dec_num_filters, 3, padding='same', name=self._dec_conv_name())(x_in)

        #feature extraction
        for i in range(self.dec_num_blocks):
            b = residual_block(b, self.dec_num_filters, self._dec_conv_name)
        b = layers.Conv2D(self.dec_num_filters, 3, padding='same', name=self._dec_conv_name())(b)
        x = layers.Add()([x, b])

        #feature upscaling
        x = layers.Conv2DTranspose(3, 5, self.scale, padding='same', name='conv2d_transpose')(x)

        return x

    def build_encoder(self):
        x_in = layers.Input(shape=(None, None, 3))
        if self.normalize_config : x_in = layers.Lambda(self.normalize_config.normalize)(x_in)
        x = self._encoder(x_in)
        model = Model(inputs=x_in, outputs=x, name=self.name)
        self.enc_conv_idx = 0
        return model

    def build_decoder(self):
        x_in = layers.Input(shape=(None, None, self.feature_dims))
        x = self._decoder(x_in)
        if self.normalize_config : x = layers.Lambda(self.normalize_config.denormalize)(x)
        model = Model(inputs=x_in, outputs=x, name=self.name)
        self.dec_conv_idx = 2 * self.enc_num_blocks + 3
        return model

    def build_model(self):
        #name = '{}_B{},{}_F{},{}_S{}'.format(self.__class__.__name__, self.enc_num_blocks, self.dec_num_blocks, \
        #                                    self.enc_num_filters, self.dec_num_filters, self.scale)
        x_in = layers.Input(shape=(None, None, 3))
        if self.normalize_config : x_in = layers.Lambda(self.normalize_config.normalize)(x_in)
        x_feature = self._encoder(x_in)
        x = self._decoder(x_feature)
        if self.normalize_config : x = layers.Lambda(self.normalize_config.denormalize)(x)
        model = Model(inputs=x_in, outputs=[x_feature, x], name=self.name)
        self.enc_conv_idx = 0
        self.dec_conv_idx = 2 * self.enc_num_blocks + 3
        return model

    def convert_to_h5(self, checkpoint_dir):
        model = self.build_model()
        print(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                        directory=checkpoint_dir, max_to_keep=3)
        checkpoint_path = checkpoint_manager.latest_checkpoint
        print('checkpoint: {}'.format(checkpoint_path))
        assert(checkpoint_path is not None)
        h5_path = '{}.h5'.format(os.path.splitext(checkpoint_path)[0])

        if not os.path.exists(h5_path):
            checkpoint.restore(checkpoint_path)
            checkpoint.model.save_weights(h5_path)
        return h5_path

    def load_encoder(self, checkpoint_dir):
        h5_path = self.convert_to_h5(checkpoint_dir)
        encoder = self.build_encoder()
        encoder.load_weights(h5_path, by_name=True)
        return encoder

    def load_decoder(self, checkpoint_dir):
        h5_path = self.convert_to_h5(checkpoint_dir)
        decoder = self.build_decoder()
        decoder.load_weights(h5_path, by_name=True)
        return decoder

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/gpu:0'):
        normalize_config = NormalizeConfig('normalize_01', 'denormalize_01')
        edsr_ed_s = EDSR_ED_S(4, 32, 4, 32, 4, normalize_config)
        model = edsr_ed_s.build_model()
        input_tensor = tf.random.uniform((1, 200, 200, 3), 0, 255)
        output_tensor = model(input_tensor)
        print(output_tensor[1].shape)
