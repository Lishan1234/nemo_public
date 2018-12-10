import tensorflow as tf
from importlib import import_module
from trainer import Trainer
from option import args

tf.enable_eager_execution()

def loss_func(loss_type):
    assert loss_type in ['l1', 'l2']

    if loss_type == 'l1':
        return tf.losses.absolute_difference
    elif loss_type == 'l2':
        return tf.losses.mean_squared_error

model_module = import_module('model.' + args.model_name.lower())
dataset_module = import_module('dataset.' + args.data_name.lower())

model = model_module.make_model(args, 4)
dataset = dataset_module.make_dataset(args, 4)
loss = loss_func(args.loss_type)
trainer = Trainer(args, model, dataset, loss)

trainer.load_model( args.checkpoint_path)

for epoch in range(args.num_epoch):
    print('[Train-{}epoch] Start'.format(epoch))
    trainer.train()
    print('[Train-{}epoch] End'.format(epoch))
    print('[Validation-{}epoch] Start'.format(epoch))
    trainer.validate()
    print('[Validation-{}epoch] End'.format(epoch))
    print('[Visualization-{}epoch] Start'.format(epoch))
    trainer.visualize(3)
    print('[Visualization-{}epoch] End'.format(epoch))
    trainer.save_model()
