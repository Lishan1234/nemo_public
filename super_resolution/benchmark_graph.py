# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests and Benchmarks for Densenet model under graph execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from importlib import import_module

from option import args

def data_format():
    #return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'
    #return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'
    #return 'channels_last'
    return 'channels_first'

def image_shape(batch_size):
    if data_format() == 'channels_first':
        return [batch_size, 3, 270, 480]
    else:
        return [batch_size, 270, 480, 3]

def random_batch(batch_size):
    images = np.random.rand(*image_shape(batch_size)).astype(np.float32)
    return images

class DensenetBenchmark(tf.test.Benchmark):
    def __init__(self):
        model_module = import_module('model.' + args.model_name.lower())
        dataset_module = import_module('dataset.' + args.data_name.lower())
        args.data_format = data_format()
        self.model = model_module.make_model(args, 4)

    def _report(self, label, start, num_iters, batch_size):
        avg_time = (time.time() - start) / num_iters
        dev = 'gpu' if tf.test.is_gpu_available() else 'cpu'
        name = 'graph_%s_%s_batch_%d_%s' % (label, dev, batch_size, data_format())
        extras = {'examples_per_sec': batch_size / avg_time}
        self.report_benchmark(
            iters=num_iters, wall_time=avg_time, name=name, extras=extras)

    def benchmark_graph_apply(self):
        with tf.Graph().as_default():
            images = tf.placeholder(tf.float32, image_shape(None))
            predictions = self.model(images)

            init = tf.global_variables_initializer()

            batch_size = 1
            with tf.Session() as sess:
                sess.run(init)
                np_images = random_batch(batch_size)
                num_burn, num_iters = (3, 30)
                for _ in range(num_burn):
                    sess.run(predictions, feed_dict={images: np_images})
                start = time.time()
                for _ in range(num_iters):
                    sess.run(predictions, feed_dict={images: np_images})
                self._report('apply', start, num_iters, batch_size)

if __name__ == '__main__':
    DensenetBenchmark().benchmark_graph_apply()
