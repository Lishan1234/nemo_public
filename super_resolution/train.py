import os, time, sys
from importlib import import_module

import tensorflow as tf

import utility as util
from dataset import TFRecordDataset
from option import args

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
        self.learning_rate = tfe.Variable(self.args.lr_init, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        #Checkpoint
        self.root = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model,
                            optimizer_step=tf.train.get_or_create_global_step())
        self.checkpoint_dir = os.path.join(args.data_dir, args.train_data, args.train_datatype, args.checkpoint_dir, model_builder.get_name())
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

    def load_model(self):
        assert tf.train.latest_checkpoint(self.checkpoint_dir) is not None
        self.root.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    #TODO: save model for .pb, .h5 with input shape
    def save_model(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.root.save(checkpoint_prefix)
        #self.model.save_weights(os.path.join(self.checkpoint_dir, 'keras'), save_format='h5')
        #self.model.save(os.path.join(self.checkpoint_dir, 'model.h5'), include_optimizer=False)

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
            count = 0
            for input_image, target_image, baseline_image in self.valid_dataset:
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
            #for input_image, target_image, baseline_image in self.valid_dataset.take(self.args.num_sample):
            for input_image, target_image, baseline_image in self.valid_dataset.take(1):
                output_image = self.model(input_image)
                output_image = tf.clip_by_value(output_image, 0.0, 1.0)

                tf.contrib.summary.image('Input{}'.format(count), input_image)
                tf.contrib.summary.image('Output{}'.format(count), output_image)
                tf.contrib.summary.image('Target{}'.format(count), target_image)
                tf.contrib.summary.image('Basline{}'.format(count), baseline_image)
                tf.contrib.summary.flush(self.writer)

                count += 1

if __name__ == '__main__':
    tf.enable_eager_execution()

    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    dataset = TFRecordDataset(args)
    trainer = Trainer(args, model_builder, dataset)

    """transfer learning or continual learning
    #trainer.load_model(args.checkpoint_path)
    """

    for epoch in range(args.num_epoch):
        #train
        start_time = time.time()
        print('[Train-{}epoch] Start'.format(epoch))
        trainer.train()
        print('[Train-{}epoch] End (take {} seconds)'.format(epoch, time.time()-start_time))

        #validate
        if epoch % args.valid_interval == 0:
            start_time = time.time()
            print('[Validation-{}epoch] Start'.format(epoch))
            trainer.validate()
            print('[Validation-{}epoch] End (take {} seconds)'.format(epoch, time.time()-start_time))
            #visualize
            start_time = time.time()
            print('[Visualization-{}epoch] Start'.format(epoch))
            trainer.visualize()
            print('[Visualization-{}epoch] End (take {} seconds)'.format(epoch, time.time()-start_time))

        #checkpoint
        if (epoch == args.num_epoch - 1) or (epoch % args.valid_interval == 0):
            trainer.save_model()

        #lr decaying
        if epoch != 0 and epoch % args.lr_decay_epoch == 0:
            trainer.apply_lr_decay(args.lr_decay_rate)
