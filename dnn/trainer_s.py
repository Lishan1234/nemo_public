from importlib import import_module
import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from utility import evaluate

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
        #self.restore() #caution: use for testing or finetuning
        self.writer = tf.contrib.summary.create_file_writer(log_dir)

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()
        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - self.checkpoint.step.numpy()):
            self.checkpoint.step.assign_add(1)
            step = self.checkpoint.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                _, psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                #TODO: early-stop after inspecting a validation psnr curve

                if save_best_only and psnr_value <= self.checkpoint.psnr:
                    self.now = time.perf_counter()
                    continue

                self.checkpoint.psnr = psnr_value
                self.checkpoint_manager.save()

                with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Training_Loss', loss_value, step=self.checkpoint.step.numpy())
                    tf.contrib.summary.scalar('Validation_PSNR', psnr_value, step=self.checkpoint.step.numpy())
                    #TODO: also print learning rate
                    #tf.contrib.summary.scalar('Learning_Rate', self.checkpoint.optimizer.lr.numpy(), step=self.checkpoint.step.numpy())
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

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

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
    #def train(self, train_dataset, valid_dataset, steps=20, evaluate_every=10, save_best_only=False):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)

if __name__ == '__main__':
    pass
