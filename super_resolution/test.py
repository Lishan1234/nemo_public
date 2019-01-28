import tensorflow as tf
from importlib import import_module
import time

from tester import Tester
from option import args
from dataset import ImageDataset

#gpu_option = tf.GPUOptions(allow_growth=True)
#config = tf.ConfigProto(gpu_options=gpu_option)
tf.enable_eager_execution()

#redefine scale
args.scale = args.hr // args.lr

model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
dataset = ImageDataset(args)
tester = Tester(args, model_builder, dataset)
tester.load_model()
start_time = time.time()
print('[Test] Start')
tester.validate()
print('[Test] End (take {} seconds)'.format(time.time()-start_time))
