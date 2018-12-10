import tensorflow as tf
import os, time
from importlib import import_module

tfe = tf.contrib.eager

class Trainer():
    def __init__(self, args, model, dataset, loss):
        self.args = args
        self.loss = loss
        self.model = model
        self.dataset = dataset
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        self.root = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model,
                            optimizer_step=tf.train.get_or_create_global_step())
        self.checkpoint_dir = os.path.join(self.args.model_dir, self.model.get_name())
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        board_dir = os.path.join(self.args.board_dir, self.model.get_name())
        os.makedirs(board_dir, exist_ok=True)
        self.writer = tf.contrib.summary.create_file_writer(board_dir)

        #Tensorboard
        self.training_loss = tfe.metrics.Mean("Training Loss")
        self.validation_loss = tfe.metrics.Mean("Validation Loss")
        self.validation_psnr = tfe.metrics.Mean("Validation PSNR")
        self.validation_baseline_loss= tfe.metrics.Mean("Validation Baseline Loss")
        self.validation_baseline_psnr = tfe.metrics.Mean("Validation Baseline PSNR")


    def load_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            self.root.restore(checkpoint_path)
        else:
            self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def save_model(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.root.save(checkpoint_prefix)

    def train(self):
        train_dataset = self.dataset.create_train_dataset()
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for input_images, target_images in train_dataset.take(self.args.num_batch_per_epoch):
                with tf.GradientTape() as tape:
                    output_images = self.model(input_images)
                    loss_value = self.loss(output_images, target_images)

                grads = tape.gradient(loss_value, self.model.variables)
                self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                        global_step=tf.train.get_or_create_global_step())
                self.training_loss(loss_value)

            tf.contrib.summary.scalar('Average Traning Loss', self.training_loss.result())
            tf.contrib.summary.flush(self.writer)

            self.training_loss.init_variables()

    def validate(self):
        valid_dataset = self.dataset.create_test_dataset()
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for input_image, target_image, baseline_image in valid_dataset:
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)

                output_loss_value = self.loss(output_image, target_image)
                output_psnr_value = tf.image.psnr(output_image, target_image, max_val=1.0)

                baseline_loss_value = self.loss(baseline_image, target_image)
                baseline_psnr_value = tf.image.psnr(baseline_image, target_image, max_val=1.0)

                self.validation_loss(output_loss_value)
                self.validation_psnr(output_psnr_value)
                self.validation_baseline_loss(baseline_loss_value)
                self.validation_baseline_psnr(baseline_psnr_value)

            tf.contrib.summary.scalar('Average Validation Loss', self.validation_loss.result())
            tf.contrib.summary.scalar('Average Validation PSNR', self.validation_psnr.result())
            tf.contrib.summary.scalar('Average Baseline Validation Loss', self.validation_baseline_loss.result())
            tf.contrib.summary.scalar('Average Baseline Validation PSNR', self.validation_baseline_psnr.result())
            tf.contrib.summary.flush(self.writer)

            self.validation_loss.init_variables()
            self.validation_baseline_loss.init_variables()
            self.validation_psnr.init_variables()
            self.validation_baseline_psnr.init_variables()

    def visualize(self, num_image):
        valid_dataset = self.dataset.create_test_dataset(num_image)
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            count = 0
            for input_image, target_image, baseline_image in valid_dataset:
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)

                tf.contrib.summary.image('Input{}'.format(count), input_image)
                tf.contrib.summary.image('Output{}'.format(count), output_image)
                tf.contrib.summary.image('Target{}'.format(count), target_image)
                tf.contrib.summary.image('Basline{}'.format(count), baseline_image)
                tf.contrib.summary.flush(self.writer)

                count += 1
