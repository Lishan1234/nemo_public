import tensorflow as tf
from importlib import import_module
import time

from trainer import Trainer
from tester import Tester
from option import args
from dataset import TFRecordDataset, FeatureDataset

tf.enable_eager_execution()

args.mode = 0
model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
tester = Tester(args, model_builder)
tester.load_model()
tester.save_feature()

args.mode = 1
model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
tester_ = Tester(args, model_builder)
tester_.load_model()
tester_.validate_feature()

"""
args.mode = 2
model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
tester__ = Tester(args, model_builder)
tester__.load_model()
tester__.validate()
"""

#tester.validate_feature()
#tester.save_feature()
