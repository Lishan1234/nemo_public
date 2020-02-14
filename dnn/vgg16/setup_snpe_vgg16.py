import argparse
import os

import tensorflow as tf

IMG_SHAPE = (224, 224, 3)

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=True, weights='imagenet')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                    directory=args.log_dir,
                                                    max_to_keep=1)
    checkpoint_manager.save()
