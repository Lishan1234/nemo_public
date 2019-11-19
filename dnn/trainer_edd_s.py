import time
import argparse
import os

from utility import FFmpegOption, upscale_factor
from dataset import train_image_dataset, valid_image_dataset, setup_images
from model.edsr_edd_s import EDSR_EDD_S

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

tf.enable_eager_execution()

class Trainer:
    def __init__(self, model, loss, loss_type, learning_rate, checkpoint_dir, log_dir):
        assert(loss_type in ['separate', 'joint'])

        self.now = None
        self.loss = loss
        self.loss_type = loss_type
        checkpoint_name = 'ckpt_{}'.format(self.loss_type)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                psnr=tf.Variable(-1.0),
                                                optimizer=Adam(learning_rate),
                                                model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                                directory=checkpoint_dir,
                                                                max_to_keep=3,
                                                                checkpoint_name=checkpoint_name)

        self.writer = tf.contrib.summary.create_file_writer(log_dir)

    @property
    def model(self):
        return self.checkpoint.model

    def evaluate(self, dataset):
        lr_psnr_values = []
        sr_psnr_values = []
        for lr, hr in dataset:
            lr = tf.cast(lr, tf.float32)
            _, lr_, sr = self.checkpoint.model(lr)

            lr_ = tf.clip_by_value(lr_, 0, 255)
            lr_ = tf.round(lr_)
            lr_ = tf.cast(lr_, tf.uint8)
            lr = tf.cast(lr, tf.uint8)
            lr_psnr_value = tf.image.psnr(lr_, lr, max_val=255)[0]
            lr_psnr_values.append(lr_psnr_value)

            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            sr_psnr_value = tf.image.psnr(sr, hr, max_val=255)[0]
            sr_psnr_values.append(sr_psnr_value)

        return tf.reduce_mean(lr_psnr_values), tf.reduce_mean(sr_psnr_values)

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        lr_loss_mean = Mean()
        sr_loss_mean = Mean()
        total_loss_mean = Mean()
        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - self.checkpoint.step.numpy()):
            self.checkpoint.step.assign_add(1)
            step = self.checkpoint.step.numpy()

            lr_loss, sr_loss, total_loss = self.train_step(lr, hr)
            lr_loss_mean(lr_loss)
            sr_loss_mean(sr_loss)
            total_loss_mean(total_loss)

            if step % evaluate_every == 0:
                lr_loss_value = lr_loss_mean.result()
                sr_loss_value = sr_loss_mean.result()
                total_loss_value = total_loss_mean.result()
                lr_loss_mean.reset_states()
                sr_loss_mean.reset_states()
                total_loss_mean.reset_states()

                lr_psnr_value, sr_psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: lr_loss = {lr_loss_value.numpy():.3f}, \
                        sr_loss = {sr_loss_value.numpy():.3f}, \
                        total_loss = {total_loss_value.numpy():.3f}, \
                        PNSR (lr) = {lr_psnr_value.numpy():.3f}, \
                        PSNR (sr) = {sr_psnr_value.numpy():3f}, \
                        ({duration:.2f}s)')

                with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Training_Loss_LR', lr_loss_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Training_Loss_SR', sr_loss_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Training_Loss_Total', total_loss_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Validation_PSNR_LR', lr_psnr_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Validation_PSNR_SR', sr_psnr_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.flush(self.writer)

                self.now = time.perf_counter()

    def train_step(self, lr, hr):
        if self.loss_type == 'separate':
            with tf.GradientTape(persistent=True) as tape_sr:
                with tf.GradientTape(persistent=True) as tape_lr:
                    lr = tf.cast(lr, tf.float32)
                    hr = tf.cast(hr, tf.float32)

                    _, lr_, sr = self.checkpoint.model(lr, training=True)
                    lr_loss_value = self.loss(lr_, lr)
                    sr_loss_value = self.loss(sr, hr)
                    total_loss_value = lr_loss_value + sr_loss_value

                lr_gradients = tape_lr.gradient(lr_loss_value, self.checkpoint.model.trainable_variables)
                self.checkpoint.optimizer.apply_gradients(zip(lr_gradients, self.checkpoint.model.trainable_variables))
            sr_gradients = tape_sr.gradient(sr_loss_value, self.checkpoint.model.trainable_variables)
            self.checkpoint.optimizer.apply_gradients(zip(sr_gradients, self.checkpoint.model.trainable_variables))

        elif self.loss_type == 'joint':
            with tf.GradientTape(persistent=True) as tape:
                lr = tf.cast(lr, tf.float32)
                hr = tf.cast(hr, tf.float32)

                _, lr_, sr = self.checkpoint.model(lr, training=True)
                lr_loss_value = self.loss(lr_, lr)
                sr_loss_value = self.loss(sr, hr)
                total_loss_value = lr_loss_value + sr_loss_value

            total_gradients = tape.gradient(total_loss_value, self.checkpoint.model.trainable_variables)
            self.checkpoint.optimizer.apply_gradients(zip(total_gradients, self.checkpoint.model.trainable_variables))

        return lr_loss_value, sr_loss_value, total_loss_value

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

#TODO (minor): boundry should be different for each step?
#TODO: boundaries should be different

class EDSRTrainer(Trainer):
    def __init__(self,
                    model,
                    loss_type,
                    checkpoint_dir,
                    log_dir,
                    learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), loss_type=loss_type, learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, log_dir=log_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=False):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str, choices=['uniform', 'keyframes',], default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')

    #dataset
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--loss_type', type=str, required=True)

    #architecture
    parser.add_argument('--enc_num_filters', type=int, required=True)
    parser.add_argument('--enc_num_blocks', type=int, required=True)
    parser.add_argument('--dec_lr_num_filters', type=int, required=True)
    parser.add_argument('--dec_lr_num_blocks', type=int, required=True)
    parser.add_argument('--dec_sr_num_filters', type=int, required=True)
    parser.add_argument('--dec_sr_num_blocks', type=int, required=True)
    parser.add_argument('--enable_normalization', action='store_true')

    #log
    parser.add_argument('--custom_tag', type=str, default=None)

    args = parser.parse_args()

    #0. setting
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))

    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #1. dnn
    scale = upscale_factor(lr_video_path, hr_video_path)
    if args.enable_normalization:
        #TODO: rgb mean
        #normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
        pass
    else:
        normalize_config = None
    edsr_ed_s = EDSR_EDD_S(args.enc_num_blocks, args.enc_num_filters, \
                           args.dec_lr_num_blocks, args.dec_lr_num_filters, \
                           args.dec_sr_num_blocks, args.dec_sr_num_filters, \
                            scale, normalize_config, args.loss_type)
    model = edsr_ed_s.build_model()

    #1. dataset
    train_ds = train_image_dataset(lr_image_dir, hr_image_dir, args.batch_size, args.patch_size, scale, args.load_on_memory)
    valid_ds = valid_image_dataset(lr_image_dir, hr_image_dir)

    #2. create a trainer
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), model.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.lr_video_name), model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    trainer = EDSRTrainer(model, args.loss_type, checkpoint_dir, log_dir)
    trainer.train(train_ds, valid_ds, steps=100000)
