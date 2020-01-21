import subprocess
import os
import sys

import tensorflow as tf

from tool.tf import get_tensorflow_dir

#check python version
python_version = sys.version_info
if not (python_version[0] == 3 and python_version[1] == 4):
    raise RuntimeError('Unsupported Python version: {}'.format(python_version))

#check tensorflow, snpe directory
OPT_4_INFERENCE_SCRIPT  = 'optimize_for_inference.py'
TENSORFLOW_ROOT = get_tensorflow_dir()
SNPE_ROOT = os.path.join(os.environ['MOBINAS_CODE_ROOT'], 'third_party', 'snpe')
assert(os.path.exists(TENSORFLOW_ROOT))
assert(os.path.exists(SNPE_ROOT))

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

def snpe_convert_model(model, nhwc, checkpoint_dir):
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
    status = checkpoint.restore(latest_checkpoint)
    sess = tf.keras.backend.get_session()
    status.initialize_or_restore(sess)
    graph = tf.get_default_graph()
    frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, checkpoint_dir, pb_name, as_text=False)

    #optimize a frozen graph
    input_name = model.inputs[0].name.split(':')[0]
    output_name = model.outputs[0].name.split(':')[0]
    pb_path = os.path.join(checkpoint_dir, pb_name)
    opt_pb_path = os.path.join(checkpoint_dir, opt_pb_name)
    optimize_for_inference(pb_path, opt_pb_path, input_name, output_name)

    #convert to a dlc (.dlc)
    dlc_path = os.path.join(checkpoint_dir, dlc_name)
    snpe_tensorflow_to_dlc(pb_path, dlc_path, input_name, output_name, nhwc)

    #convcert to a quantized dlc (.quantized.dlc)
    #TODO

    #visualize a dlc
    html_path = os.path.join(checkpoint_dir, '{}.html'.format(dlc_name))
    snpe_dlc_viewer(dlc_path, html_path)

    return dlc_dict

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
