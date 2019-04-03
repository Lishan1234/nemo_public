#
# Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run inception_v3 model with SNPE SDK.
'''
import tensorflow as tf
import numpy as np
import os
import subprocess
import shutil
import hashlib
import argparse
import sys
import glob
import argparse
from importlib import import_module
sys.path.append('../../')
from option import args
args.hiai_project_root = '../../../hiai'
import json
import collections

from PIL import Image

INCEPTION_V3_ARCHIVE_CHECKSUM       = 'a904ddf15593d03c7dd786d552e22d73'
INCEPTION_V3_ARCHIVE_FILE           = 'inception_v3_2016_08_28_frozen.pb.tar.gz'
INCEPTION_V3_ARCHIVE_URL            = 'https://storage.googleapis.com/download.tensorflow.org/models/' + INCEPTION_V3_ARCHIVE_FILE
INCEPTION_V3_PB_FILENAME            = 'inception_v3_2016_08_28_frozen.pb'
INCEPTION_V3_PB_OPT_FILENAME        = 'inception_v3_2016_08_28_frozen_opt.pb'
INCEPTION_V3_DLC_FILENAME           = 'inception_v3.dlc'
INCEPTION_V3_DLC_QUANTIZED_FILENAME = 'inception_v3_quantized.dlc'
INCEPTION_V3_LBL_FILENAME           = 'imagenet_slim_labels.txt'
OPT_4_INFERENCE_SCRIPT              = 'optimize_for_inference.py'
RAW_LIST_FILE                       = 'raw_list.txt'
TARGET_RAW_LIST_FILE                = 'target_raw_list.txt'
TARGET_ABS_RAW_LIST_FILE            = 'target_abs_raw_list.txt'

def wget(download_dir, file_url):
    cmd = ['wget', '-N', '--directory-prefix={}'.format(download_dir), file_url]
    subprocess.call(cmd)

def generateMd5(path):
    checksum = hashlib.md5()
    with open(path, 'rb') as data_file:
        while True:
            block = data_file.read(checksum.block_size)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()

def find_optimize_for_inference():
    for root, dirs, files in os.walk(args.snpe_tensorflow_root):
        if OPT_4_INFERENCE_SCRIPT in files:
            return os.path.join(root, OPT_4_INFERENCE_SCRIPT)

def optimize_for_inference(pb_filename, input_name, output_name, model_dir, tensorflow_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        cmd = ['python', opt_4_inference_file,
               '--input', os.path.join(tensorflow_dir, pb_filename),
               '--output', os.path.join(tensorflow_dir, pb_filename + '_opt'),
               '--input_names', input_name,
               '--output_names', output_name]
        subprocess.call(cmd)
        pb_filename = pb_filename + '_opt'

    return pb_filename

def __get_img_raw(img_filepath):
    img_filepath = os.path.abspath(img_filepath)
    img = Image.open(img_filepath)
    img_ndarray = np.array(img) # read it
    if len(img_ndarray.shape) != 3:
        raise RuntimeError('Image shape' + str(img_ndarray.shape))
    if (img_ndarray.shape[2] != 3):
        raise RuntimeError('Require image with rgb but channel is %d' % img_ndarray.shape[2])
    # reverse last dimension: rgb -> bgr
    return img_ndarray

def __create_raw_img(img_filepath, div, req_bgr_raw, save_uint8):
    img_raw = __get_img_raw(img_filepath)
    #mean_raw = __create_mean_raw(img_raw, mean_rgb)

    #snpe_raw = img_raw - mean_raw
    snpe_raw = img_raw
    snpe_raw = snpe_raw.astype(np.float32)
    # scalar data divide
    snpe_raw /= div

    if req_bgr_raw:
        snpe_raw = snpe_raw[..., ::-1]

    if save_uint8:
        snpe_raw = snpe_raw.astype(np.uint8)
    else:
        snpe_raw = snpe_raw.astype(np.float32)

    img_filepath = os.path.abspath(img_filepath)
    filename, ext = os.path.splitext(img_filepath)
    snpe_raw_filename = filename
    snpe_raw_filename += '.raw'
    snpe_raw.tofile(snpe_raw_filename)

    return 0

def create_file_list(input_dir, output_filename, ext_pattern, print_out=False, rel_path=False):
    input_dir = os.path.abspath(input_dir)
    output_filename = os.path.abspath(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.isdir(input_dir):
        raise RuntimeError('input_dir %s is not a directory' % input_dir)

    if not os.path.isdir(output_dir):
        raise RuntimeError('output_filename %s directory does not exist' % output_dir)

    glob_path = os.path.join(input_dir, ext_pattern)
    file_list = sorted(glob.glob(glob_path))

    if rel_path:
        file_list = [os.path.relpath(file_path, output_dir) for file_path in file_list]

    if len(file_list) <= 0:
        if print_out: print('No results with %s' % glob_path)
    else:
        with open(output_filename, 'w') as f:
            f.write('\n'.join(file_list))
            if print_out: print('%s created listing %d files.' % (output_filename, len(file_list)))

def prepare_data_images(src_data_dir, dst_data_dir):
    # make a copy of the (lr, hr) image files
    lr_src_img_files = os.path.join(src_data_dir, '{}p'.format(args.lr), 'original')
    hr_src_img_files = os.path.join(src_data_dir, '{}p'.format(args.lr * args.scale), 'original')

    lr_dst_data_dir = os.path.join(dst_data_dir, '{}p'.format(args.lr))
    hr_dst_data_dir = os.path.join(dst_data_dir, '{}p'.format(args.lr * args.scale))

    if not os.path.isdir(lr_dst_data_dir): os.makedirs(lr_dst_data_dir)
    if not os.path.isdir(hr_dst_data_dir): os.makedirs(hr_dst_data_dir)

    assert len(glob.glob('{}/*.png'.format(lr_src_img_files))) != 0
    assert len(glob.glob('{}/*.png'.format(hr_src_img_files))) != 0

    for file in glob.glob('{}/*.png'.format(lr_src_img_files)):
        shutil.copy(file, lr_dst_data_dir)
    for file in glob.glob('{}/*.png'.format(hr_src_img_files)):
        shutil.copy(file, hr_dst_data_dir)

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

def fix_frozen_graph_def(graph_def):
    # fix nodes
    for node in graph_def.node:
        if "dilations" in node.attr:
            del node.attr["dilations"]

def convert_to_pb(model, model_name, save_path):
    #Restore parameters
    checkpoint_dir = os.path.join('../../', args.checkpoint_dir, args.train_data, model_name)
    root = tf.train.Checkpoint(model=model)
    assert tf.train.latest_checkpoint(checkpoint_dir) is not None
    status = root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    #status.assert_consumed()

    #Save input, output tensor names to a config file
    input_name = model.inputs[0].name.split(':')[0]
    output_name = model.outputs[0].name.split(':')[0]

    #Save fronzen graph (.pb) file
    pb_filename = 'final_{}_{}_{}.pb'.format(args.hwc[0], args.hwc[1], args.hwc[2])
    pb_txt_filename = 'final_{}_{}_{}.txt'.format(args.hwc[0], args.hwc[1], args.hwc[2])
    sess = tf.keras.backend.get_session()
    status.initialize_or_restore(sess)
    my_graph=tf.get_default_graph()
    frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])

    #Fix a graph
    fix_frozen_graph_def(frozen_graph)

    tf.train.write_graph(frozen_graph, save_path, pb_filename, as_text=False)
    tf.train.write_graph(frozen_graph, save_path, pb_txt_filename, as_text=True)

    sess.close()

    return pb_filename, input_name, output_name

def prepare_param_file(offline_dir, input_name, output_name):
    offline_file_name = 'final_{}_{}_{}'.format(args.hwc[0], args.hwc[1], args.hwc[2])
    param_file_name = 'final_{}_{}_{}.txt'.format(args.hwc[0], args.hwc[1], args.hwc[2])
    param_file = os.path.join(offline_dir, param_file_name)

    with open(param_file, 'w') as f:
        f.write('model_name:{}.cambricon\n'.format(offline_file_name))
        f.write('session_run{\n')
        f.write('  input_nodes(1):\n')
        f.write('  "{}",1,{},{},{}\n'.format(input_name, args.hwc[0], args.hwc[1], args.hwc[2]))
        f.write('  output_nodes(1):\n')
        f.write('  "{}"\n'.format(output_name))
        f.write('}')

def convert_to_offline(offline_dir, tensorflow_dir, is_quantized):
    converter_path = os.path.abspath(os.path.join(args.hiai_project_root, 'tools', 'tools_tensorflow', 'tf_offline_tools'))
    library_path = os.path.abspath(os.path.join(args.hiai_project_root, 'tools', 'tools_tensorflow', 'tf_offline_tools', 'so'))
    tensorflow_dir = os.path.abspath(tensorflow_dir)
    offline_dir = os.path.abspath(offline_dir)
    os.environ['PATH'] += os.pathsep + converter_path
    print(os.environ['PATH'])
    os.environ['LD_LIBRARY_PATH'] += os.pathsep + library_path

    if is_quantized:
        arg_ver = '--mlu_core_version=1H8'
    else:
        arg_ver = '--mlu_core_version=1H16'

    param_filename = 'final_{}_{}_{}.txt'.format(args.hwc[0], args.hwc[1], args.hwc[2])
    pb_filename = 'final_{}_{}_{}.pb'.format(args.hwc[0], args.hwc[1], args.hwc[2])

    #subprocess.call(['pb_to_offline', '--graph={}/'.format(tensorflow_dir, pb_filename), '--param_file={}/{}'.format(offline_dir, param_filename), arg_ver], cwd = offline_dir)
    cmd = 'pb_to_offline --graph={}/{} --param_file={}/{} {}'.format(tensorflow_dir, pb_filename, offline_dir, param_filename, arg_ver)
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, cwd=offline_dir)
    output = p.stdout.read()
    print(output)

def setup_assets():
    #Build model
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    model = model_builder.build()
    model_name = model_builder.get_name()

    #Make directory
    model_dir = os.path.join(args.hiai_project_root, 'custom', args.train_data, 'models', model_name)
    data_dir = os.path.join(args.hiai_project_root, 'custom', args.train_data, 'data')
    tensorflow_dir = os.path.join(model_dir, 'tensorflow')
    offline_dir = os.path.join(model_dir, 'offline')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(offline_dir, exist_ok=True)
    os.makedirs(tensorflow_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    prepare_data_images(os.path.join('../../', args.data_dir, args.train_data, args.data_type), data_dir)

    #Convert to frozen graph (.pb) file
    pb_filename, input_name, output_name = convert_to_pb(model, model_name, tensorflow_dir)
    pb_filename = optimize_for_inference(pb_filename, input_name, output_name, model_dir, tensorflow_dir)

    #Convert to offline model
    prepare_param_file(offline_dir, input_name, output_name)
    convert_to_offline(offline_dir, tensorflow_dir, False)

if __name__ == '__main__':
    assert args.hwc is not None

    setup_assets()