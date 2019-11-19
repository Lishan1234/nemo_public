import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

from model.common import NormalizeConfig

NETWORK_NAME = 'EDSR_ED_S'

def residual_block(x_in, num_filters, name_func):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu', name=name_func())(x_in)
    x = layers.Conv2D(num_filters, 3, padding='same', name=name_func())(x)
    x = layers.Add()([x_in, x])
    return x

class EDSR_EDD_S():
    def __init__(self, enc_num_blocks, enc_num_filters, \
            dec_lr_num_blocks, dec_lr_num_filters, \
            dec_sr_num_blocks, dec_sr_num_filters, \
            scale, normalize_config):
        self.enc_num_blocks = enc_num_blocks
        self.enc_num_filters = enc_num_filters
        self.dec_lr_num_blocks = dec_lr_num_blocks
        self.dec_lr_num_filters = dec_lr_num_filters
        self.dec_sr_num_blocks = dec_sr_num_blocks
        self.dec_sr_num_filters = dec_sr_num_filters
        self.scale = scale
        self.normalize_config = normalize_config

        self.name = NETWORK_NAME
        self.name += '_B{}'.format(self.enc_num_blocks)
        self.name += '_F{}'.format(self.enc_num_filters)
        self.name += '_B{}'.format(self.dec_lr_num_blocks)
        self.name += '_F{}'.format(self.dec_lr_num_filters)
        self.name += '_B{}'.format(self.dec_sr_num_blocks)
        self.name += '_F{}'.format(self.dec_sr_num_filters)
        self.name += '_S{}'.format(self.scale)

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

    def _encoder(self, x_in, num_blocks, num_filters):
        #feature extraction
        x = b = layers.Conv2D(num_filters, 3, padding='same', name=self._enc_conv_name())(x_in)
        for i in range(num_blocks):
            b = residual_block(b, num_filters, self._enc_conv_name)
        b = layers.Conv2D(num_filters, 3, padding='same', name=self._enc_conv_name())(b)
        x = layers.Add()([x, b])

        #feature reduction
        x = layers.Conv2D(3, 3, padding='same', name=self._enc_conv_name())(x)

        return x

    def _decoder_sr(self, x_in, num_blocks, num_filters, scale):
        #feature expansion
        x = b = layers.Conv2D(num_filters, 3, padding='same', name=self._dec_conv_name())(x_in)

        #feature extraction
        for i in range(num_blocks):
            b = residual_block(b, num_filters, self._dec_conv_name)
        b = layers.Conv2D(num_filters, 3, padding='same', name=self._dec_conv_name())(b)
        x = layers.Add()([x, b])

        #feature upscaling
        x = layers.Conv2DTranspose(3, 5, scale, padding='same', name='conv2d_transpose')(x)

        return x

    def _decoder_lr(self, x_in, num_blocks, num_filters, scale):
        #feature expansion
        x = b = layers.Conv2D(num_filters, 3, padding='same', name=self._dec_conv_name())(x_in)

        #feature extraction
        for i in range(num_blocks):
            b = residual_block(b, num_filters, self._dec_conv_name)
        b = layers.Conv2D(num_filters, 3, padding='same', name=self._dec_conv_name())(b)
        x = layers.Add()([x, b])

        #feature recover
        x = layers.Conv2D(3, 3, padding='same', name=self._dec_conv_name())(x)

        return x

    def build_encoder(self):
        #name = '{}_B{}_F{}'.format(self.__class__.__name__, self.enc_num_blocks, self.enc_num_filters)
        x_in = layers.Input(shape=(None, None, 3))

        if self.normalize_config : x_in = layers.Lambda(self.normalize_config.normalize)(x_in)
        x = self._encoder(x_in, self.enc_num_blocks, self.enc_num_filters)

        model = Model(inputs=x_in, outputs=x, name=self.name)
        self.enc_conv_idx = 0
        return model

    def build_decoder(self):
        #name = '{}_B{}_F{}_S{}'.format(self.__class__.__name__, self.dec_num_blocks, self.dec_num_filters, self.scale)
        x_in = layers.Input(shape=(None, None, 3))

        lr = self._decoder_lr(x_in, self.dec_lr_num_blocks, self.dec_lr_num_filters, self.scale)
        if self.normalize_config : lr = layers.Lambda(self.normalize_config.denormalize)(lr)

        sr = self._decoder_sr(x_in, self.dec_sr_num_blocks, self.dec_sr_num_filters, self.scale)
        if self.normalize_config : sr = layers.Lambda(self.normalize_config.denormalize)(sr)

        model = Model(inputs=x_in, outputs=[lr, sr], name=self.name)
        self.dec_conv_idx = 2 * self.enc_num_blocks + 3
        return model

    def build_model(self):
        #name = '{}_B{},{}_F{},{}_S{}'.format(self.__class__.__name__, self.enc_num_blocks, self.dec_num_blocks, \
        #                                    self.enc_num_filters, self.dec_num_filters, self.scale)
        x_in = layers.Input(shape=(None, None, 3))

        if self.normalize_config : x_in = layers.Lambda(self.normalize_config.normalize)(x_in)
        x_feature = self._encoder(x_in, self.enc_num_blocks, self.enc_num_filters)

        lr = self._decoder_lr(x_feature, self.dec_lr_num_blocks, self.dec_sr_num_filters, self.scale)
        if self.normalize_config : lr = layers.Lambda(self.normalize_config.denormalize)(lr)

        sr = self._decoder_sr(x_feature, self.dec_sr_num_blocks, self.dec_sr_num_filters, self.scale)
        if self.normalize_config : sr = layers.Lambda(self.normalize_config.denormalize)(sr)

        model = Model(inputs=x_in, outputs=[x_feature, lr, sr], name=self.name)
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

    def visualize_graph(self, log_dir):
        assert(not tf.executing_eagerly())
        with tf.Graph().as_default(), tf.Session() as sess:
            init = tf.global_variables_initializer()
            model = self.build_model()
            sess.run(init)
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

if __name__ == '__main__':
    with tf.device('/gpu:0'):
        edsr_edd_s = EDSR_EDD_S(4, 32, 4, 32, 4, 32, 4, None)
        model = edsr_edd_s.build_model()
        input_tensor = tf.random.uniform((1, 200, 200, 3), 0, 255)
        output_tensor = model(input_tensor)
        edsr_edd_s.visualize_graph('./.log')
        print(output_tensor[0].shape)
        print(output_tensor[1].shape)
        print(output_tensor[2].shape)
