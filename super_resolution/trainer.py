import tensorflow as tf
import os
from importlib import import_module

tfe = tf.contrib.eager

class Trainer():
    def __init__(self, args, model, dataset, loss):
        self.args = args
        self.loss = loss
        self.model = model
        self.train_dataset = dataset.create_train_dataset()
        self.valid_dataset = dataset.create_test_dataset()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        self.root = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model,
                            optimizer_step=tf.train.get_or_create_global_step())
        self.checkpoint_dir = os.path.join(self.args.model_dir, self.model.get_name())
        self.writer = tf.contrib.summary.create_file_writer(self.args.board_dir)
        self.writer.set_as_default()

        #Tensorboard
        self.training_loss = tfe.metrics.Mean("Training Loss")
        self.validation_loss = tfe.metrics.Mean("Validation Loss")
        self.validation_psnr = tfe.metrics.Mean("Validation PSNR")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            self.root.restore(checkpoint_path)
        else:
            self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def save_model(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.root.save(checkpoint_prefix)

    def train(self):
        for input_images, target_images in self.train_dataset.take(self.args.num_batch_per_epoch):
            with tf.GradientTape() as tape:
                output_images = self.model(input_images)
                loss_value = self.loss(output_images, target_images)

            grads = tape.gradient(loss_value, self.model.variables)
            self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                    global_step=tf.train.get_or_create_global_step())
            self.training_loss(loss_value)

        with tf.contrib.summary.record_summaries_every_n_global_steps(100, global_step=tf.train.get_or_create_global_step()):
            tf.contrib.summary.scalar('Average Traning Loss', self.training_loss.result())
        self.training_loss.init_variables()

    def validate(self):
        for input_image, target_image in self.valid_dataset.make_one_shot_iterator():
            output_image = self.model(input_image)
            loss_value = self.loss_func(output_images, target_image)

            output_image = self.clip_by_value(output_image, 0.0, 1.0)
            psnr_value = tf.image.psnr(output_image, target_image, max_val=1.0)

            self.validation_loss(loss_value)
            self.validation_psnr(psnr_value)

        with tf.contrib.summary.record_summaries_every_n_global_steps(100, global_step=tf.train.get_or_create_global_step()):
            tf.contrib.summary.scalar('Average Validation Loss', self.validation_loss.result())
            tf.contrib.summary.scalar('Average Validation PSNR', self.validation_psnr.result())
        self.validation_loss.init_variables()
        self.validation_psnr.init_variables()

    def visualize(self, num_image):
        count = 0
        for input_image, target_image in self.valid_dataset.take(num_image):
            output_image = self.model(input_image)
            output_image = self.clip_by_value(output_image, 0.0, 1.0)

            with tf.contrib.summary.record_summaries_every_n_global_steps(100, global_step=tf.train.get_or_create_global_step()):
                tf.contrib.summary.image('Input{}'.format(count), input_image)
                tf.contrib.summary.image('Output{}'.format(count), output_image)
                tf.contrib.summary.image('Target{}'.format(count), target_image)

            count += 1
