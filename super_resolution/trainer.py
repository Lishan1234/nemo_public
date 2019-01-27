import tensorflow as tf
import os, time
from importlib import import_module
import utility as util

tfe = tf.contrib.eager

def loss_func(loss_type):
    assert loss_type in ['l1', 'l2']

    if loss_type == 'l1':
        return tf.losses.absolute_difference
    elif loss_type == 'l2':
        return tf.losses.mean_squared_error

class Trainer():
    def __init__(self, args, model_builder, dataset):
        self.args = args
        self.model = model_builder.build()
        self.loss = loss_func(args.loss_type)

        #Optimizer
        self.learning_rate = tfe.Variable(self.args.lr, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        #Checkpoint
        self.root = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model,
                            optimizer_step=tf.train.get_or_create_global_step())
        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.train_data, model_builder.get_name())
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        #Dataset
        self.train_dataset = dataset.create_train_dataset()
        self.valid_dataset = dataset.create_valid_dataset()

        #Tensorboard
        log_dir = os.path.join(self.args.log_dir, self.args.train_data, model_builder.get_name())
        os.makedirs(log_dir, exist_ok=True)
        self.writer = tf.contrib.summary.create_file_writer(log_dir)
        self.training_loss = tfe.metrics.Mean("Training Loss")
        self.validation_loss = tfe.metrics.Mean("Validation Loss")
        self.validation_psnr = tfe.metrics.Mean("Validation PSNR")
        self.validation_baseline_loss= tfe.metrics.Mean("Validation Baseline Loss")
        self.validation_baseline_psnr = tfe.metrics.Mean("Validation Baseline PSNR")

    def apply_lr_decay(self, lr_decay_rate):
        self.learning_rate.assign(self.learning_rate * lr_decay_rate)

    #TODO: save model for .pb, .h5 with input shape
    def load_model(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            self.root.restore(checkpoint_dir)
        else:
            self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    #TODO: save model for .pb, .h5 with input shape
    def save_model(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.root.save(checkpoint_prefix)

    def train(self):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            count = 0
            #Iterate until num_batch_per_epoch
            start_time = time.time()
            for idx, batch in enumerate(self.train_dataset.take(self.args.num_batch_per_epoch)):
                """@deprecated
                #validate & visualize
                if idx != 0 and idx % self.args.num_batch_per_epoch == 0:
                    print('[Train-{}epoch] End (take {} seconds)'.format(idx//self.args.num_batch_per_epoch, time.time()-start_time))
                    start_time = time.time()
                    self.validate()
                    self.visualize(self.args.num_sample)
                    self.save_model()
                    print('[Validation-{}epoch] End (take {} seconds)'.format(idx//self.args.num_batch_per_epoch, time.time()-start_time))
                    start_time = time.time()

                #finish training
                if idx == self.args.num_batch_per_epoch * self.args.num_epoch:
                    break
                """

                #train
                input_images = batch[0]
                target_images = batch[1]
                with tf.GradientTape() as tape:
                    output_images = self.model(input_images)
                    loss_value = self.loss(output_images, target_images)

                grads = tape.gradient(loss_value, self.model.variables)
                self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                        global_step=tf.train.get_or_create_global_step())
                self.training_loss(loss_value)

                util.print_progress((idx % self.args.num_batch_per_epoch + 1), self.args.num_batch_per_epoch, 'Train Progress:', 'Complete', 1, 50)

            tf.contrib.summary.scalar('Average Traning Loss', self.training_loss.result())
            tf.contrib.summary.scalar('Learning rate', self.learning_rate)
            tf.contrib.summary.flush(self.writer)

            self.training_loss.init_variables()

    #TODO: measure PSNR in uint8 precision (slight difference)
    def validate(self):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            for input_image, target_image, baseline_image in self.valid_dataset:
            #.take(self.args.num_sample):
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

    def visualize(self):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device('gpu:{}'.format(self.args.gpu_idx)):
            count = 0
            for input_image, target_image, baseline_image in self.valid_dataset.take(self.args.num_sample):
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)

                tf.contrib.summary.image('Input{}'.format(count), input_image)
                tf.contrib.summary.image('Output{}'.format(count), output_image)
                tf.contrib.summary.image('Target{}'.format(count), target_image)
                tf.contrib.summary.image('Basline{}'.format(count), baseline_image)
                tf.contrib.summary.flush(self.writer)

                count += 1
