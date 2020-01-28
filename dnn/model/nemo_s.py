import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class NEMO_S():
    def __init__(self, num_blocks, num_filters, scale):
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.scale = scale
        self.conv_idx = 0

        #name
        self.name = self.__class__.__name__
        self.name += '_B{}'.format(self.num_blocks)
        self.name += '_F{}'.format(self.num_filters)
        self.name += '_S{}'.format(self.scale)

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

    def build_model(self):
        x_in = layers.Input(shape=(None, None, 3))

        x = b = layers.Conv2D(self.num_filters, 3, padding='same', name=self.conv_name())(x_in)

        for i in range(self.num_blocks):
            b = self.residual_block(b, self.num_filters)
        b = layers.Conv2D(self.num_filters, 3, padding='same', name=self.conv_name())(b)
        x = layers.Add()([x, b])

        x = layers.Conv2DTranspose(3, 5, self.scale, padding='same')(x)

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
        model = NAS_S(4, 32, 4).build_model()
        input_tensor = tf.random.uniform((1, 200, 200, 3), 0, 255)
        output_tensor = model(input_tensor)
        print(model.name, output_tensor.shape)
