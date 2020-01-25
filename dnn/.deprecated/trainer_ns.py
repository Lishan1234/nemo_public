import time
import argparse
import os

from utility import FFmpegOption
from dataset import train_image_dataset, valid_image_dataset, setup_images
from model.edsr_ns import EDSR_NS

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

tf.enable_eager_execution()

class Trainer:
    def __init__(self, model, loss, learning_rate, checkpoint_dir, log_dir):
        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                psnr=tf.Variable(-1.0),
                                                optimizer=Adam(learning_rate),
                                                model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                                directory=checkpoint_dir,
                                                                max_to_keep=3)
        self.writer = tf.contrib.summary.create_file_writer(log_dir)

    @property
    def model(self):
        return self.checkpoint.model

    def evaluate(self, dataset):
        psnr_values = []
        for feature, lr in dataset:
            feature = tf.cast(feature, tf.float32)
            lr_ = model(feature)
            lr_ = tf.clip_by_value(lr_, 0, 255)
            lr_ = tf.round(lr_)
            lr_ = tf.cast(lr_, tf.uint8)
            psnr_value = tf.image.psnr(lr_, lr, max_val=255)[0]
            psnr_values.append(psnr_value)
        return tf.reduce_mean(psnr_values)

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()
        self.now = time.perf_counter()

        for feature, lr in train_dataset.take(steps - self.checkpoint.step.numpy()):
            self.checkpoint.step.assign_add(1)
            step = self.checkpoint.step.numpy()

            loss = self.train_step(feature, lr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                self.checkpoint.psnr = psnr_value
                self.checkpoint_manager.save()

                with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Training_Loss_LR', loss_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Validation_PSNR_LR', psnr_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.flush(self.writer)

                self.now = time.perf_counter()

    def train_step(self, feature, lr):
        with tf.GradientTape() as tape:
            feature = tf.cast(feature, tf.float32)
            lr = tf.cast(lr, tf.float32)

            lr_ = self.checkpoint.model(feature, training=True)
            loss_value = self.loss(lr_, lr)

            gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
            self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

class EDSRTrainer(Trainer):
    def __init__(self,
                    model,
                    checkpoint_dir,
                    log_dir,
                    learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, log_dir=log_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=False):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--feature_video_name', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--encode_model_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')

    #dataset_
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--enable_normalization', action='store_true')

    #log
    parser.add_argument('--custom_tag', type=str, default=None)

    args = parser.parse_args()

    #setting (lr)
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    assert(os.path.exists(lr_video_path))

    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dnn
    if args.enable_normalization:
        #TODO: rgb mean
        #normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
        pass
    else:
        normalize_config = None
    edsr_ns = EDSR_NS(args.num_blocks, args.num_filters, \
                            normalize_config)
    model = edsr_ns.build_model()

    #setting (feature)
    feature_video_path = os.path.join(args.dataset_dir, 'video', args.encode_model_name, args.feature_video_name)
    print(feature_video_path)
    assert(os.path.exists(feature_video_path))
    feature_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.feature_video_name), args.encode_model_name)
    setup_images(feature_video_path, feature_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dataset
    train_ds = train_image_dataset(feature_image_dir, lr_image_dir, args.batch_size, args.patch_size, 1, args.load_on_memory)
    valid_ds = valid_image_dataset(feature_image_dir, lr_image_dir)

    #trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.feature_video_name), model.name)
    log_dir = os.path.join('.log', ffmpeg_option.summary(args.feature_video_name), model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    trainer = EDSRTrainer(model, checkpoint_dir, log_dir)
    trainer.train(train_ds, valid_ds, steps=100000)
