import tensorflow as tf
from importlib import import_module
import time

from trainer import Trainer
from option import args
from dataset import TFRecordDataset

tf.enable_eager_execution()

model_module = import_module('model.' + args.model_type.lower())

model = model_module.make_model(args)
dataset = TFRecordDataset(args)
trainer = Trainer(args, model, dataset)

"""transfer learning or continual learning
#trainer.load_model(args.checkpoint_path)
"""

for epoch in range(args.num_epoch):
    start_time = time.time()
    print('[Train-{}epoch] Start'.format(epoch))
    trainer.train()
    print('[Train-{}epoch] End (take {} seconds)'.format(epoch, time.time()-start_time))
    start_time = time.time()
    print('[Validation-{}epoch] Start'.format(epoch))
    trainer.validate()
    print('[Validation-{}epoch] End (take {} seconds)'.format(epoch, time.time()-start_time))
    start_time = time.time()
    print('[Visualization-{}epoch] Start'.format(epoch))
    trainer.visualize(args.num_sample)
    print('[Visualization-{}epoch] End (take {} seconds)'.format(epoch, time.time()-start_time))
    trainer.save_model()

    if epoch != 0 and epoch % args.lr_decay_epoch == 0:
        trainer.apply_lr_decay(args.lr_decay_rate)
