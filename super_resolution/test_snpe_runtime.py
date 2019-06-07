import subprocess
from importlib import import_module
import numpy as np
import collections
import json

import tensorflow as tf

from option import *
from config import *

TARGET_RAW_LIST_FILE                = 'target_abs_raw_list.txt'
DATA_NAME                           = 'input.raw'
DEVICE_ROOT_DIR                     = '/data/local/tmp/snpebm'
OPT_4_INFERENCE_SCRIPT              = 'optimize_for_inference.py'
SNPE_BENCH_SCRIPT                   = 'snpe_bench.sh'

assert args.hwc is not None
assert args.benchmark_device_id is not None

if args.benchmark_device_id is None:
    adb_cmd_prefix = 'adb '
else:
    adb_cmd_prefix = 'adb -s {} '.format(args.benchmark_device_id)

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

def prepare_fake_pb(model, model_name, checkpoint_dir):
    input_name = model.inputs[0].name.split(':')[0]
    output_name = model.outputs[0].name.split(':')[0]

    pb_filename = 'final_{}_{}_{}.pb'.format(args.hwc[0], args.hwc[1], args.hwc[2])
    init_op = tf.initialize_all_variables()
    sess = tf.keras.backend.get_session()
    sess.run(init_op)
    my_graph=tf.get_default_graph()
    frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, checkpoint_dir, pb_filename, as_text=False)
    sess.close()

    return pb_filename, input_name, output_name

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

def optimize_for_inference(pb_filename, input_name, output_name, checkpoint_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        cmd = ['python', opt_4_inference_file,
               '--input', os.path.join(checkpoint_dir, pb_filename),
               '--output', os.path.join(checkpoint_dir, pb_filename + '_opt'),
               '--input_names', input_name,
               '--output_names', output_name]
        subprocess.call(cmd)
        pb_filename = pb_filename + '_opt'

    return pb_filename

def convert_to_dlc(pb_filename, input_name, output_name, model_name, checkpoint_dir):
    dlc_filename = '{}.dlc'.format(model_name)
    print('INFO: Converting ' + pb_filename +' to SNPE DLC format')
    cmd = ['snpe-tensorflow-to-dlc',
           '--graph', os.path.join(checkpoint_dir, pb_filename),
           '--input_dim', input_name, '1,{},{},{}'.format(args.hwc[0], args.hwc[1], args.hwc[2]),
           '--out_node', output_name,
           '--dlc', os.path.join(checkpoint_dir, dlc_filename),
           '--allow_unconsumed_nodes']
    subprocess.call(cmd)

def prepare_fake_image(data_dir, prefix):
    img_h = args.target_resolution // args.scale
    img_w = WIDTH[img_h]
    img_fake = np.random.rand(img_h, img_w, args.channel_in)
    img_fake = img_fake * 255.0
    img_fake = img_fake.astype(np.uint8)
    img_fake.tofile(os.path.join(data_dir, DATA_NAME))

    with open(os.path.join(data_dir,TARGET_RAW_LIST_FILE), 'w') as f:
        f.write(os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale), DATA_NAME))

def setup_assets(model, model_name, prefix):
    root_dir = os.path.join(args.data_dir, 'runtime')
    checkpoint_dir = os.path.join(root_dir, prefix, 'checkpoint')
    data_dir = os.path.join(root_dir, prefix, 'data', '{}p'.format(args.target_resolution // args.scale))
    result_dir = os.path.join(root_dir, prefix, 'result')
    benchmark_dir = os.path.join(root_dir, prefix, 'benchmark')

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)

    #convert to frozen graph (.pb) file
    pb_filename, input_name, output_name = prepare_fake_pb(model, model_name, checkpoint_dir)
    pb_filename = optimize_for_inference(pb_filename, input_name, output_name, checkpoint_dir)

    #convert to dlc file
    convert_to_dlc(pb_filename, input_name, output_name, model_name, checkpoint_dir)

    #setup dataset: a) check, b) save a randomly intialized frame
    prepare_fake_image(data_dir, prefix)

    #setup benchmark prerequisite: a) check, b) make a json file
    name = model_name
    benchmark = collections.OrderedDict()
    benchmark['Name'] = name
    benchmark['HostRootPath'] = name
    benchmark['HostResultsDir'] = os.path.abspath(result_dir)
    benchmark['DevicePath'] = '/data/local/tmp/snpebm'
    if args.benchmark_device_id == None:
        benchmark['Devices'] = []
    else:
        benchmark['Devices'] = [str(args.benchmark_device_id)]
    benchmark['HostName'] = 'localhost'
    benchmark['Runs'] = args.benchmark_iter_num
    benchmark['Model'] = collections.OrderedDict()
    benchmark['Model']['Name'] = name
    benchmark['Model']['Dlc'] = os.path.abspath(os.path.join(checkpoint_dir, '{}.dlc'.format(model_name)))
    benchmark['Model']['InputList'] = os.path.abspath(os.path.join(data_dir, TARGET_RAW_LIST_FILE))
    benchmark['Model']['Data'] = [os.path.abspath(data_dir)]
    benchmark['Runtimes'] = ['GPU_FP16']
    benchmark['Measurements'] = ['timing']

    with open(os.path.join(benchmark_dir, '{}.json'.format(name)),'w') as outfile:
            json.dump(benchmark, outfile, indent=4)

def setup_prerequisites():
    os.environ['SNPE_TARGET_ARCH']='aarch64-android-clang6.0'
    os.environ['SNPE_TARGET_ARCH_OBJ_DIR']='arm64-v8a'
    os.environ['SNPE_ROOT']= os.path.abspath('../snpe')
    os.system(adb_cmd_prefix + 'shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin"')
    os.system(adb_cmd_prefix + 'shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib"')
    os.system(adb_cmd_prefix + 'shell "mkdir -p /data/local/tmp/snpeexample/dsp/lib"')
    os.system(adb_cmd_prefix + 'push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/* /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system(adb_cmd_prefix + 'push $SNPE_ROOT/lib/dsp/* /data/local/tmp/snpeexample/dsp/lib')
    os.system('echo $SNPE_TARGET_ARCH')
    os.system(adb_cmd_prefix + 'push $SNPE_ROOT/sample_cpp/obj/local/$SNPE_TARGET_ARCH_OBJ_DIR/snpe-sample /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin')

#Copy dlc and data
def setup_dlc_data(model_name, prefix):
    root_dir = os.path.join(args.data_dir, 'runtime')
    checkpoint_dir = os.path.join(root_dir, prefix, 'checkpoint')

    src_data_dir = os.path.join(root_dir, prefix, 'data', '{}p'.format(args.target_resolution // args.scale))
    src_dlc_path = os.path.join(checkpoint_dir, '{}.dlc'.format(model_name))
    dst_dlc_path = os.path.join(DEVICE_ROOT_DIR, 'checkpoint',  '{}.dlc'.format(model_name))
    dst_data_dir = os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale))

    os.system(adb_cmd_prefix + 'shell "mkdir -p {}"'.format(os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale))))
    os.system(adb_cmd_prefix + 'shell "mkdir -p {}"'.format(os.path.join(DEVICE_ROOT_DIR, 'checkpoint')))
    os.system(adb_cmd_prefix + 'push {} {}'.format(src_dlc_path, dst_dlc_path))
    os.system(adb_cmd_prefix + 'push {}/* {}'.format(src_data_dir, dst_data_dir))

#measure end-to-end runtime: a) measure, b) save a log file
def measure_dnn_latency(model_name, prefix):
    src_result_dir = os.path.join(args.data_dir, 'runtime', prefix, 'result')
    dst_dlc_path = os.path.join(DEVICE_ROOT_DIR, 'checkpoint', '{}.dlc'.format(model_name))
    dst_data_list_path = os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale), TARGET_RAW_LIST_FILE)
    dst_result_dir = os.path.join(DEVICE_ROOT_DIR, 'result')

    os.system(adb_cmd_prefix + 'shell rm -rf {}'.format(dst_result_dir))

    if args.benchmark_runtime == 'CPU':
        dlc_name = '{}.dlc'.format(model_name)
        runtime_opt = '-r cpu'
    #elif args.benchmark_runtime == 'CPU_IP8':
    #    dlc_name = 'quantized_{}.dlc'.format(model_name)
    #    runtime_opt = '-r cpu_ip8'
    elif args.benchmark_runtime == 'GPU':
        dlc_name = '{}.dlc'.format(model_name)
        runtime_opt = '-r gpu'
    elif args.benchmark_runtime == 'GPU_FP16':
        dlc_name = '{}.dlc'.format(model_name)
        runtime_opt = '-r gpu_fp16'
    else:
        raise NotImplementedError

    cmds = ['export SNPE_TARGET_ARCH=aarch64-android-clang6.0',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib',
            'export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin',
            'cd {}'.format(os.path.join(DEVICE_ROOT_DIR, 'data')),
            'snpe-sample -d {} {} -i {} -o ./{}'.format(dst_dlc_path, runtime_opt, dst_data_list_path, dst_result_dir),
            'exit']

    print(dst_dlc_path, dst_data_list_path, dst_result_dir)

    cmd_script_path = os.path.join(SNPE_BENCH_SCRIPT)
    with open(cmd_script_path, 'w') as cmd_script:
        cmd_script.write('#!/system/bin/sh'+'\n')
        for ln in cmds:
            cmd_script.write(ln + '\n')

    os.system(adb_cmd_prefix + 'push {} {}'.format(cmd_script_path, DEVICE_ROOT_DIR))
    os.system(adb_cmd_prefix + 'shell sh {}'.format(os.path.join(DEVICE_ROOT_DIR, SNPE_BENCH_SCRIPT)))
    os.system(adb_cmd_prefix + 'pull {} {}'.format(dst_result_dir, src_result_dir))

def measure_layer_latency(model_name, prefix):
    bench_json_path = os.path.abspath(os.path.join(args.data_dir, 'runtime', prefix, 'benchmark', '{}.json'.format(model_name)))
    os.chdir("../snpe/benchmarks")
    #print(os.getcwd())
    os.system('/usr/bin/python2 snpe_bench.py -c {} -a'.format(bench_json_path))
#measure layer-wise runtime: a) measure, b) save a xls file

if __name__ == '__main__':
    cmd_board = adb_cmd_prefix + 'shell getprop ro.product.board'
    cmd_model = adb_cmd_prefix + 'shell getprop ro.product.model'
    proc_board = subprocess.Popen(cmd_board, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    board = proc_board.stdout.readlines()[0].decode().rstrip('\r\n').replace(' ', '')
    proc_model = subprocess.Popen(cmd_model, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    model = proc_model.stdout.readlines()[0].decode().rstrip('\r\n').replace(' ', '')
    prefix = '{}_{}'.format(board, model)

    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    model = model_builder.build()
    model_name = model_builder.get_name()
    setup_assets(model, model_name, prefix)

    #setup_prerequisites()
    #setup_dlc_data(model_name, prefix)
    #measure_dnn_latency(model_name, prefix)
    measure_layer_latency(model_name, prefix)
