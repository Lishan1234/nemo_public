import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class NAS_S():
    def __init__(self, num_blocks, num_filters, scale, upsample_type='deconv'):
        assert(upsample_type == 'deconv' or upsample_type == 'subpixel')

        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.scale = scale
        self.conv_idx = 0
        self.upsample_type = upsample_type

        #name
        self.name = self.__class__.__name__
        self.name += '_B{}'.format(self.num_blocks)
        self.name += '_F{}'.format(self.num_filters)
        self.name += '_S{}'.format(self.scale)
        self.name += '_{}'.format(upsample_type)

    def conv_name(self):
        if self.conv_idx== 0:
            name = 'conv2d'
        else:
            name = 'conv2d_{}'.format(self.conv_idx)
        self.conv_idx += 1
        return name

    def residual_block(self, x_in, num_filters):
        x = layers.Conv2D(num_filters, 3, padding='same', activation='relu', name=self.conv_name())(x_in)
        x = layers.Conv2D(num_filters, 3, padding='same', name=self.conv_name())(x)
        x = layers.Add()([x_in, x])
        return x

    def build_model(self, resolution=None):
        x_in = layers.Input(shape=(None, None, 3))

        x = b = layers.Conv2D(self.num_filters, 3, padding='same', name=self.conv_name())(x_in)
        for i in range(self.num_blocks):
            b = self.residual_block(b, self.num_filters)
        b = layers.Conv2D(self.num_filters, 3, padding='same', name=self.conv_name())(b)
        x = layers.Add()([x, b])

        if self.upsample_type == 'deconv':
            if self.scale in [2, 3, 4]:
                x = layers.Conv2DTranspose(self.num_filters, 5, self.scale, padding='same')(x)
            else:
                raise NotImplementedError
        elif self.upsample_type == 'subpixel':
            if self.scale == 2 or self.scale ==  3:
                x = layers.Conv2D(self.num_filters * (self.scale ** 2), 3, padding='same', name=self.conv_name())(x)
                x = layers.Lambda(lambda x:tf.nn.depth_to_space(x, self.scale))(inputs=x)
            elif self.scale == 4:
                x = layers.Conv2D(self.num_filters * 4, 3, padding='same', name=self.conv_name())(x)
                x = layers.Lambda(lambda x:tf.nn.depth_to_space(x, 2))(inputs=x)
                x = layers.Conv2D(self.num_filters * 4, 3, padding='same', name=self.conv_name())(x)
                x = layers.Lambda(lambda x:tf.nn.depth_to_space(x, 2))(inputs=x)
            else:
                raise NotImplementedError
        x = layers.Conv2D(3, 3, padding='same', name=self.conv_name())(x)

        if resolution is not None:
            x = tf.image.resize_bilinear(x, (resolution[0], resolution[1]), half_pixel_centers=True)
            x = tf.minimum(x, 255)
            x = tf.maximum(x, 0)

        model = Model(inputs=x_in, outputs=x, name=self.name)

        return model

    def load_checkpoint(self, checkpoint_dir):
        model = self.build_model()
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                        directory=checkpoint_dir, max_to_keep=3)
        checkpoint_path = checkpoint_manager.latest_checkpoint
        print('checkpoint: {}'.format(checkpoint_path))
        assert(checkpoint_path is not None)
        checkpoint.restore(checkpoint_path).expect_partial()

        return checkpoint

if __name__ == '__main__':
    tf.enable_eager_execution()
    with tf.device('/gpu:0'):
        model = NAS_S(4, 32, 4, 'subpixel').build_model()
        input_tensor = tf.random.uniform((1, 200, 200, 3), 0, 255)
        output_tensor = model(input_tensor)
        print(model.name, output_tensor.shape)
