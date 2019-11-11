import subprocess
import os

import tensorflow as tf

OPT_4_INFERENCE_SCRIPT              = 'optimize_for_inference.py'

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
    cmd = 'pip show tensorflow'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = proc.stdout.readlines()
    tensorflow_dir = None
    for line in lines:
        line = line.decode().rstrip('\r\n')
        if 'Location' in line:
            tensorflow_dir = line.split(' ')[1]
            for root, dirs, files in os.walk(os.path.join(tensorflow_dir, 'tensorflow')):
                if OPT_4_INFERENCE_SCRIPT in files:
                    return os.path.join(root, OPT_4_INFERENCE_SCRIPT)
            break
    return None

def optimize_for_inference(pb_name, opt_pb_name, input_name, output_name, checkpoint_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        cmd = ['python', opt_4_inference_file,
               '--input', os.path.join(checkpoint_dir, pb_name),
               '--output', os.path.join(checkpoint_dir, opt_pb_name),
               '--input_names', input_name,
               '--output_names', output_name]
        subprocess.call(cmd)

def check_attached_devices(target_device_id=None):
    cmd = 'adb devices'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = proc.stdout.readlines()
    count = 0

    for idx, line in enumerate(lines):
        if idx == 0 or idx == len(lines) - 1:
            continue
        line = line.decode().rstrip('\r\n')
        device_id = line.split('\t')[0]
        device_status = line.split('\t')[1]
        if device_id == target_device_id and device_status == 'device':
            return True
        count += 1

    if target_device_id is None and count == 1:
        return True

    return False
