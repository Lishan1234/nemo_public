import time
import argparse
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from utility import FFmpegOption, upscale_factor
from dataset import train_feature_dataset, valid_feature_dataset, setup_images
from model.common import NormalizeConfig, QuantizeConfig, quantize, dequantize
from model.edsr_ed_s import EDSR_ED_S

tf.enable_eager_execution()

class Trainer:
    def __init__(self, edsr_ed_s, loss, learning_rate, quantization_policy, pretrained_checkpoint_dir, checkpoint_dir, log_dir):
        self.now = None
        self.loss = loss

        #restore
        self.decoder = edsr_ed_s.load_decoder(pretrained_checkpoint_dir)
        self.qnt_config = QuantizeConfig(quantization_policy)
        self.qnt_config.load(pretrained_checkpoint_dir)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                psnr=tf.Variable(-1.0),
                                                optimizer=Adam(learning_rate),
                                                model=self.decoder)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                                directory=checkpoint_dir,
                                                                max_to_keep=3)
        self.writer = tf.contrib.summary.create_file_writer(log_dir)

    @property
    def model(self):
        return self.checkpoint.model

    def evaluate(self, dataset):
        psnr_values = []
        for lr, feature_qnt, hr in dataset:
            feature_qnt = tf.cast(feature_qnt, tf.float32)
            feature = dequantize(feature_qnt, self.qnt_config.enc_min, self.qnt_config.enc_max)
            sr = self.checkpoint.model(feature)
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            psnr_value = tf.image.psnr(sr, hr, max_val=255)[0]
            psnr_values.append(psnr_value)
        return tf.reduce_mean(psnr_values)

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()
        self.now = time.perf_counter()

        for _, feature_qnt, hr in train_dataset.take(steps - self.checkpoint.step.numpy()):
            self.checkpoint.step.assign_add(1)
            step = self.checkpoint.step.numpy()

            feature = dequantize(feature_qnt, self.qnt_config.enc_min, self.qnt_config.enc_max)
            loss = self.train_step(feature, hr)
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
                    tf.contrib.summary.scalar('Training_Loss_SR', loss_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Validation_PSNR_SR', psnr_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.flush(self.writer)

                self.now = time.perf_counter()

    def train_step(self, feature, hr):
        with tf.GradientTape() as tape:
            feature = tf.cast(feature, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(feature, training=True)
            loss_value = self.loss(sr, hr)

            gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
            self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

#TODO: we should lower learning rate for fine-tuning (try a couple of numbers)
class EDSRTrainer(Trainer):
    def __init__(self,
                    model,
                    quantization_policy,
                    pretrained_checkpoint_dir,
                    checkpoint_dir,
                    log_dir,
                    learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, quantization_policy=quantization_policy, pretrained_checkpoint_dir=pretrained_checkpoint_dir, checkpoint_dir=checkpoint_dir, log_dir=log_dir)

    #def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=False):
    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=10, save_best_only=False):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--feature_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    parser.add_argument('--ffprobe_path', type=str, default='usr/bin/ffprobe')

    #video metadata
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)
    parser.add_argument('--upsample', type=str, default='bilinear')

    #dataset
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--load_on_memory', action='store_true')

    #architecture
    parser.add_argument('--enc_num_filters', type=int, required=True)
    parser.add_argument('--enc_num_blocks', type=int, required=True)
    parser.add_argument('--dec_num_filters', type=int, required=True)
    parser.add_argument('--dec_num_blocks', type=int, required=True)
    parser.add_argument('--enable_normalization', action='store_true')
    parser.add_argument('--quantization_policy', type=str, required=True)

    args = parser.parse_args()

    #setting (lr, hr)
    lr_video_path = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_path))
    assert(os.path.exists(hr_video_path))

    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, args.upsample)
    lr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.lr_video_name))
    hr_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.hr_video_name))
    setup_images(lr_video_path, lr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())
    setup_images(hr_video_path, hr_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dnn
    scale = upscale_factor(lr_video_path, hr_video_path)
    if args.enable_normalization:
        #TODO: rgb mean
        #normalize_config = NormalizeConfig('normalize', 'denormalize', rgb_mean)
        pass
    else:
        normalize_config = None
    edsr_ed_s = EDSR_ED_S(args.enc_num_blocks, args.enc_num_filters, \
                           args.dec_num_blocks, args.dec_num_filters, \
                            scale, normalize_config)

    #setting (feature)
    feature_video_path = os.path.join(args.dataset_dir, 'video', edsr_ed_s.name, args.feature_video_name)
    assert(os.path.exists(feature_video_path))
    feature_image_dir = os.path.join(args.dataset_dir, 'image', ffmpeg_option.summary(args.feature_video_name))
    setup_images(feature_video_path, feature_image_dir, args.ffmpeg_path, ffmpeg_option.filter())

    #dataset
    train_ds = train_feature_dataset(lr_image_dir, feature_image_dir, hr_image_dir, args.batch_size, args.patch_size, scale, args.load_on_memory)
    valid_ds = valid_feature_dataset(lr_image_dir, feature_image_dir, hr_image_dir)

    #trainer
    pretrained_checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), edsr_ed_s.name)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.feature_video_name), edsr_ed_s.name)
    log_dir = os.path.join(args.dataset_dir, 'log', ffmpeg_option.summary(args.feature_video_name), edsr_ed_s.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    trainer = EDSRTrainer(edsr_ed_s, args.quantization_policy, pretrained_checkpoint_dir, checkpoint_dir, log_dir)
    trainer.train(train_ds, valid_ds, steps=100000)
