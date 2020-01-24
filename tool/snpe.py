import subprocess
import os
import sys
import glob
import collections
import json

import numpy as np
import imageio
import tensorflow as tf

from tool.tf import get_tensorflow_dir
from tool.adb import adb_pull

DEVICE_ROOTDIR = '/data/local/tmp/snpebm'
BENCHMARK_CONFIG_NAME = 'benchmark.json'
BENCHMARK_RAW_LIST = 'target_raw_list.txt'
OPT_4_INFERENCE_SCRIPT  = 'optimize_for_inference.py'

#check python version
python_version = sys.version_info
if not (python_version[0] == 3 and python_version[1] == 4):
    raise RuntimeError('Unsupported Python version: {}'.format(python_version))

#check tensorflow, snpe directory
TENSORFLOW_ROOT = get_tensorflow_dir()
SNPE_ROOT = os.path.join(os.environ['MOBINAS_CODE_ROOT'], 'third_party', 'snpe')
assert(os.path.exists(TENSORFLOW_ROOT))
assert(os.path.exists(SNPE_ROOT))

#TODO: fix
def snpe_dlc_viewer(dlc_path, html_path):
    setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(SNPE_ROOT, TENSORFLOW_ROOT)
    snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-dlc-viewer\
            -i {} \
            -s {}'.format(SNPE_ROOT, \
                            dlc_path, \
                            html_path)

    cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

def snpe_tensorflow_to_dlc(pb_path, dlc_path, input_name, output_name, nhwc):
    setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(SNPE_ROOT, TENSORFLOW_ROOT)
    snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-tensorflow-to-dlc \
            -i {} \
            --input_dim {} {} \
            --out_node {} \
            -o {} \
            --allow_unconsumed_nodes'.format(SNPE_ROOT, \
                            pb_path, \
                            input_name, \
                            '{},{},{},{}'.format(nhwc[0], nhwc[1], nhwc[2], nhwc[3]), \
                            output_name, \
                            dlc_path)
    cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

def snpe_benchmark(json_path):
    setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(SNPE_ROOT, TENSORFLOW_ROOT)
    snpe_cmd = 'python {}/benchmarks/snpe_bench.py -c {} -json'.format(SNPE_ROOT, json_path)

    cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

def snpe_benchmark_output(device_id, device_dir, host_dir, raw_list, output_name):
    with open(raw_list, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            device_file = os.path.join(device_dir, 'Result_{}'.format(idx), '{}.raw'.format(output_name))
            host_file = os.path.join(host_dir, line.split('/')[1])
            adb_pull(device_file, host_file, device_id)

def read_image(image_file):
    image = imageio.imread(image_file, as_gray=False, pilmode='RGB')
    image_ndarray = np.asarray(image) # read it
    if len(image_ndarray.shape) != 3:
        raise RuntimeError('Image shape' + str(image_ndarray.shape))
    if (image_ndarray.shape[2] != 3):
        raise RuntimeError('Require image with rgb but channel is %d' % image_ndarray.shape[2])
    # reverse last dimension: rgb -> bgr
    return image_ndarray

def snpe_convert_dataset(image_dir, image_format, save_uint8=False):
    #read file
    image_files = sorted(glob.glob('{}/*.{}'.format(image_dir, image_format)))

    #convert to raw images
    raw_subdir = 'raw'
    raw_list = []
    os.makedirs(os.path.join(image_dir, raw_subdir), exist_ok=True)
    for image_file in image_files:
        raw = read_image(image_file)

        if save_uint8:
            raw = raw.astype(np.uint8)
        else:
            raw = raw.astype(np.float32)

        filename, ext = os.path.splitext(image_file)
        raw_file = os.path.join(raw_subdir, '{}.raw'.format(os.path.basename(filename)))
        raw_list.append(raw_file)
        raw.tofile(os.path.join(image_dir, raw_file))

def snpe_convert_model(model, nhwc, checkpoint_dir):
    assert(not tf.executing_eagerly()) #note: output layer name is wrong in TF v1.3 with eager execution

    #restore
    checkpoint = tf.train.Checkpoint(model=model)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest_checkpoint:
        raise RuntimeError('checkpoint does not exist: {}'.format(checkpoint_dir))

    checkpoint_name = os.path.basename(latest_checkpoint).split('.')[0]
    pb_name = '{}.pb'.format(checkpoint_name)
    opt_pb_name = '{}_opt.pb'.format(checkpoint_name)
    dlc_name = '{}.dlc'.format(checkpoint_name)
    qnt_dlc_name = '{}_quantized.dlc'.format(checkpoint_name)

    dlc_dict = {'model_name': model.name , \
            'input_name': model.inputs[0].name, \
            'output_name': model.outputs[0].name, \
            'qnt_dlc_name': qnt_dlc_name, \
            'dlc_name': dlc_name,
            'dlc_path': os.path.join(checkpoint_dir, dlc_name)}

    #save a frozen graph (.pb)
    pb_path = os.path.join(checkpoint_dir, pb_name)
    if not os.path.exists(pb_path):
        status = checkpoint.restore(latest_checkpoint)
        sess = tf.keras.backend.get_session()
        status.initialize_or_restore(sess)
        graph = tf.get_default_graph()
        frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, checkpoint_dir, pb_name, as_text=False)

    #optimize a frozen graph
    opt_pb_path = os.path.join(checkpoint_dir, opt_pb_name)
    if not os.path.exists(opt_pb_path):
        input_name = model.inputs[0].name.split(':')[0]
        output_name = model.outputs[0].name.split(':')[0]
        optimize_for_inference(pb_path, opt_pb_path, input_name, output_name)

    #convert to a dlc (.dlc)
    dlc_path = os.path.join(checkpoint_dir, dlc_name)
    if not os.path.exists(dlc_path):
        snpe_tensorflow_to_dlc(pb_path, dlc_path, input_name, output_name, nhwc)
        snpe_tensorflow_to_dlc(pb_path, dlc_path, input_name, output_name, nhwc)

    #convcert to a quantized dlc (.quantized.dlc)
    #TODO

    #visualize a dlc
    html_path = os.path.join(checkpoint_dir, '{}.html'.format(dlc_name))
    if not os.path.exists(html_path):
        snpe_dlc_viewer(dlc_path, html_path)

    return dlc_dict

def snpe_benchmark_config(device_id, runtime, model, dlc_file, log_dir, image_dir, perf='default'):
    result_dir = os.path.join(log_dir, device_id, runtime)
    raw_dir = os.path.join(image_dir, 'raw')
    json_file = os.path.join(result_dir, BENCHMARK_CONFIG_NAME)
    raw_list_file = os.path.join(result_dir, BENCHMARK_RAW_LIST)
    os.makedirs(result_dir, exist_ok=True)

    with open(raw_list_file, 'w') as f:
        raw_files = sorted(glob.glob('{}/*.raw'.format(raw_dir)))
        for raw_file in raw_files:
            f.write('raw/{}\n'.format(os.path.basename(raw_file)))

    benchmark = collections.OrderedDict()
    benchmark['Name'] = model.name
    benchmark['HostRootPath'] = os.path.abspath(log_dir)
    benchmark['HostResultsDir'] = os.path.abspath(result_dir)
    benchmark['DevicePath'] = DEVICE_ROOTDIR
    benchmark['Devices'] = [device_id]
    benchmark['HostName'] = 'localhost'
    benchmark['Runs'] = 1
    benchmark['Model'] = collections.OrderedDict()
    benchmark['Model']['Name'] = model.name
    benchmark['Model']['Dlc'] = dlc_file
    benchmark['Model']['InputList'] = raw_list_file
    benchmark['Model']['Data'] = [raw_dir]
    benchmark['Runtimes'] = [runtime]
    benchmark['Measurements'] = ['timing']
    benchmark['ProfilingLevel'] = 'detailed'
    benchmark['BufferTypes'] = ['float']

    with open(json_file, 'w') as f:
        json.dump(benchmark, f, indent=4)

    return json_file

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def find_optimize_for_inference():
    for root, dirs, files in os.walk(TENSORFLOW_ROOT):
        if OPT_4_INFERENCE_SCRIPT in files:
            return os.path.join(root, OPT_4_INFERENCE_SCRIPT)
    return None

def optimize_for_inference(pb_path, opt_pb_path, input_name, output_name):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        cmd = ['python', opt_4_inference_file,
               '--input', pb_path,
               '--output', opt_pb_path,
               '--input_names', input_name,
               '--output_names', output_name]
        subprocess.call(cmd)
