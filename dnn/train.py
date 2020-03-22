import time
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from dnn.utility import resolve_bilinear

class SingleTrainer:
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
        self.initial_step = 0

    @property
    def model(self):
        return self.checkpoint.model

    def evaluate(self, dataset):
        sr_psnr_values = []
        bilinear_psnr_values = []
        for idx, ds in enumerate(dataset):
            lr = ds[0]
            hr = ds[1]

            lr = tf.cast(lr, tf.float32)
            sr = self.checkpoint.model(lr)
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            sr_psnr_value = tf.image.psnr(sr, hr, max_val=255)[0]
            sr_psnr_values.append(sr_psnr_value)

            height = tf.shape(hr)[1]
            width = tf.shape(hr)[2]
            bilinear = resolve_bilinear(lr, height, width)
            bilinear_psnr_value = tf.image.psnr(bilinear, hr, max_val=255)[0]
            bilinear_psnr_values.append(bilinear_psnr_value)

            if idx == 20:
                break

        return tf.reduce_mean(sr_psnr_values), tf.reduce_mean(bilinear_psnr_values)

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()
        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps + self.initial_step - self.checkpoint.step.numpy()):
            self.checkpoint.step.assign_add(1)
            step = self.checkpoint.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                sr_psnr_value, bilinear_psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR(SR) = {sr_psnr_value.numpy():3f}, PSNR(Bilinear) = {bilinear_psnr_value.numpy():3f} ({duration:.2f}s)')

                self.checkpoint.psnr = sr_psnr_value
                self.checkpoint_manager.save()

                with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Training_Loss_SR', loss_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Validation_PSNR_SR', sr_psnr_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Validation_PSNR_BILINEAR', bilinear_psnr_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.flush(self.writer)

                self.now = time.perf_counter()

    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(sr, hr)

            gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
            self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def restore(self, checkpoint_dir):
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint, directory=checkpoint_dir, max_to_keep=None)
        print(checkpoint_manager.latest_checkpoint)
        print(checkpoint_dir)
        if checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
        self.initial_step = self.checkpoint.step.numpy()

class SingleTrainerV1(SingleTrainer):
    def __init__(self,
                    model,
                    checkpoint_dir,
                    log_dir,
                    learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, log_dir=log_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=False):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)
