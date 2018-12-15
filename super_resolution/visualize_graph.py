import tensorflow as tf
from importlib import import_module
import os

from option import args

with tf.Graph().as_default():
    init = tf.global_variables_initializer()
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    model = model_builder.build()
    log_dir = os.path.join(args.log_dir, model_builder.get_name())
    os.makedirs(log_dir, exist_ok=True)
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
