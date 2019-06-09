import os, time, sys, time
from importlib import import_module

from config import *
from dataset import InferenceDataset
from option import args
import utility as util

import scipy.misc
import numpy as np
import tensorflow as tf
from PIL import Image

#TODO: convert from Tensorflow checkpoint to SNPE dlc

assert args.hwc is not None

#setup
model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
model = model_builder.build()
model_name = model_builder.get_name()
root = tf.train.Checkpoint(model=model)
checkpoint_dir = os.path.join(args.data_dir, args.train_data, args.train_datatype, args.checkpoint_dir, model_builder.get_name())

#load a model
assert tf.train.latest_checkpoint(checkpoint_dir) is not None
status = root.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Save input, output tensor names to a config file
input_name = model.inputs[0].name.split(':')[0]
output_name = model.outputs[0].name.split(':')[0]

#Save fronzen graph (.pb) file
pb_filename = 'final_{}_{}_{}.pb'.format(args.hwc[0], args.hwc[1], args.hwc[2])
sess = tf.keras.backend.get_session()
status.initialize_or_restore(sess)
my_graph=tf.get_default_graph()
frozen_graph = util.freeze_session(sess, output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, checkpoint_dir, pb_filename, as_text=False)
sess.close()

#Convert to dlc file
dlc_filename = 'final_{}_{}_{}.dlc'.format(args.hwc[0], args.hwc[1], args.hwc[2])
optimized_pb_filename = util.optimize_for_inference(pb_filename, input_name, output_name, checkpoint_dir)
util.convert_to_dlc(optimized_pb_filename, input_name, output_name, dlc_filename, checkpoint_dir, args.hwc[0], args.hwc[1], args.hwc[2])
