import os, time, sys, time
import subprocess
import argparse
import collections
import json

import numpy as np
import tensorflow as tf

from model.edsr_s import EDSR_S
from tool.common import freeze_session, optimize_for_inference, check_attached_devices

#TODO: random to 1, iter to 100 as global variable

class Profiler():
    model_name = 'mock'
    json_name = 'snpe_benchmark.json'
    total_run = 10
    total_num = 1

    def __init__(self, model, nhwc, log_dir, snpe_dir):
        self.model = model
        self.nhwc = nhwc
        self.log_dir = log_dir
        self.snpe_dir = snpe_dir
        os.makedirs(log_dir, exist_ok=True)

        self.pb_name = '{}.pb'.format(self.model_name)
        self.opt_pb_name = '{}_opt.pb'.format(self.model_name)
        self.dlc_name = '{}_{}.dlc'.format(self.model_name, self.nhwc[0])
        self.qnt_dlc_name = '{}_quantized_{}.dlc'.format(self.model_name, self.nhwc[0])

        #check python version
        python_version = sys.version_info
        if not (python_version[0] == 3 and python_version[1] == 4):
            raise RuntimeError('Unsupported Python version: {}'.format(python_version))

        #check tensorflow version, location
        cmd = 'pip show tensorflow'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = proc.stdout.readlines()
        self.tensorflow_dir = None
        for line in lines:
            line = line.decode().rstrip('\r\n')
            if 'Version' in line:
                tensorflow_ver = line.split(' ')[1]
                if not tensorflow_ver.startswith('1.'):
                    raise RuntimeError('Tensorflow verion is wrong: {}'.format(tensorflow_ver))
            if 'Location' in line:
                tensorflow_dir = line.split(' ')[1]
                self.tensorflow_dir = os.path.join(tensorflow_dir, 'tensorflow')
        if tensorflow_dir is None:
            raise RuntimeError('Tensorflow is not installed')

    def _snpe_benchmark(self, json_path):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/benchmarks/snpe_bench.py -c {} -json'.format(self.snpe_dir, json_path)

        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def _snpe_dlc_viewer(self, dlc_path, html_path):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-dlc-viewer\
                -i {} \
                -s {}'.format(self.snpe_dir, \
                                dlc_path, \
                                html_path)

        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def _snpe_tensorflow_to_dlc(self, pb_path, dlc_path, input_name, output_name):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-tensorflow-to-dlc \
                -i {} \
                --input_dim {} {} \
                --out_node {} \
                -o {} \
                --allow_unconsumed_nodes'.format(self.snpe_dir, \
                                pb_path, \
                                input_name, \
                                '{},{},{},{}'.format(self.nhwc[0], self.nhwc[1], self.nhwc[2], self.nhwc[3]), \
                                output_name, \
                                dlc_path)
        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def create_model(self):
        pb_path = os.path.join(self.log_dir, self.pb_name)
        opt_pb_path = os.path.join(self.log_dir, self.opt_pb_name)
        dlc_path = os.path.join(self.log_dir, self.dlc_name)
        qnt_dlc_path = os.path.join(self.log_dir, self.qnt_dlc_name)
        html_path = os.path.join(self.log_dir, '{}.html'.format(self.dlc_name))
        input_name = self.model.inputs[0].name.split(':')[0]
        output_name = self.model.outputs[0].name.split(':')[0]

        if os.path.exists(dlc_path):
            print('dlc exists')
            return

        #save a frozen graph (.pb)
        if not os.path.exists(pb_path):
            graph = tf.get_default_graph()
            sess = tf.keras.backend.get_session()
            frozen_graph = freeze_session(sess, output_names=[out.op.name for out in self.model.outputs])
            tf.train.write_graph(frozen_graph, self.log_dir, self.pb_name, as_text=False)

        #optimize a frozen graph
        if not os.path.exists(opt_pb_path):
            optimize_for_inference(pb_path, opt_pb_path, input_name, output_name)

        #convert to a dlc (.dlc)
        if not os.path.exists(dlc_path):
            self._snpe_tensorflow_to_dlc(pb_path, dlc_path, input_name, output_name)

        #convcert to a quantized dlc (.quantized.dlc)
        if not os.path.exists(qnt_dlc_path):
            pass
            #TODO: generate random images / create image list

        #visualize a dlc
        if not os.path.exists(html_path):
            self._snpe_dlc_viewer(dlc_path, html_path)

    def create_json(self, device_id, runtime, perf='default'):
        result_dir = os.path.join(self.log_dir, device_id, runtime, str(self.nhwc[0]))
        json_path = os.path.join(result_dir, self.json_name)
        os.makedirs(result_dir, exist_ok=True)

        benchmark = collections.OrderedDict()
        benchmark['Name'] = self.model.name
        benchmark['HostRootPath'] = os.path.abspath(self.log_dir)
        benchmark['HostResultsDir'] = os.path.abspath(result_dir)
        benchmark['DevicePath'] = '/data/local/tmp/snpebm'
        benchmark['Devices'] = [device_id]
        benchmark['HostName'] = 'localhost'
        benchmark['Runs'] = self.total_run
        benchmark['Model'] = collections.OrderedDict()
        benchmark['Model']['Name'] = self.model.name
        benchmark['Model']['Dlc'] = os.path.abspath(os.path.join(self.log_dir, self.dlc_name))
        benchmark['Model']['RandomInput'] = self.total_num
        benchmark['Runtimes'] = [runtime]
        benchmark['Measurements'] = ['timing']
        benchmark['ProfilingLevel'] = 'detailed'
        benchmark['BufferTypes'] = ['float']

        with open(json_path, 'w') as outfile:
            json.dump(benchmark, outfile, indent=4)

    def _load_pb(self, pb_path):
        with tf.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    #TODO: this includes intializere
    def measure_flops(self):
        input_tensor = tf.random.uniform((args.nhwc[0], args.nhwc[1], args.nhwc[2], args.nhwc[3]), 0, 255)
        output_tensor = model(input_tensor)
        sess = tf.keras.backend.get_session()
        with sess.graph.as_default():
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(sess.graph, options=opts)
        return flops.total_float_ops

    #TODO: include qnt_dlc_size
    def measure_size(self):
        opt_pb_path = os.path.join(self.log_dir, self.opt_pb_name)
        dlc_path = os.path.join(self.log_dir, self.dlc_name)
        #qnt_dlc_path = os.path.join(self.log_dir, self.qnt_dlc_name)

        opt_pb_size = os.path.getsize(opt_pb_path)
        dlc_size = os.path.getsize(dlc_path)
        #qnt_dlc_size = os.path.getsize(qnt_dlc_path)

        #return opt_pb_size, dlc_size, qnt_dlc_size
        return opt_pb_size, dlc_size

    def measure_parameters(self):
        input_tensor = tf.random.uniform((args.nhwc[0], args.nhwc[1], args.nhwc[2], args.nhwc[3]), 0, 255)
        output_tensor = model(input_tensor)
        sess = tf.keras.backend.get_session()
        with sess.graph.as_default():
            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            params_stats = tf.profiler.profile(sess.graph, options=opts)
        return params_stats.total_parameters

    def measure_latency(self, device_id, runtime):
        result_dir = os.path.join(self.log_dir, device_id, runtime, str(self.nhwc[0]))
        config_json_path = os.path.join(result_dir, self.json_name)
        assert(os.path.exists(config_json_path))

        self._snpe_benchmark(config_json_path)

        output_json_path = os.path.join(result_dir, 'latest_results', 'benchmark_stats_{}.json'.format(self.model.name))
        assert(os.path.exists(output_json_path))

        json_data = open(output_json_path).read()
        data = json.loads(json_data)
        return float(data['Execution_Data']['GPU_FP16']['Total Inference Time']['Avg_Time'])

    def measure_all(self, device_id, runtime):
        flops = self.measure_flops()
        num_params = self.measure_parameters()
        latency = self.measure_latency(device_id, runtime)
        pb_size, dlc_size = self.measure_size()

        summary = {}
        summary['flops'] = flops
        summary['num_params'] = num_params
        summary['latency'] = latency / 1000.0
        summary['pb_size'] = pb_size / 1000.0
        summary['dlc_size'] = dlc_size / 1000.0

        result_dir = os.path.join(self.log_dir, device_id, runtime, str(self.nhwc[0]))
        json_path = os.path.join(result_dir, 'summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--snpe_dir', type=str, required=True)

    #architecture
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--nhwc', nargs='+', type=int, required=True)

    #device
    parser.add_argument('--device_id', type=str, default=True)
    parser.add_argument('--runtime', type=str, required=True)

    args = parser.parse_args()

    assert(len(args.nhwc) == 4)

    #model
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, args.scale, None)
    model = edsr_s.build_model()

    #setup
    log_dir = os.path.join(args.model_dir, model.name)
    profiler = Profiler(model, args.nhwc, log_dir, args.snpe_dir)
    profiler.create_model()
    profiler.create_json(args.device_id, args.runtime)
    profiler.measure_all(args.device_id, args.runtime)
    #print(profiler.measure_latency(args.device_id, args.runtime))
    #print(profiler.measure_size())
    #print(profiler.measure_gflops())
    #print(profiler.measure_parameters())

