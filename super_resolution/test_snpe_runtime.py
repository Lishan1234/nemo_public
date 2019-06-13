import subprocess
from importlib import import_module
import numpy as np
import collections
import json

import utility as util

import tensorflow as tf

from option import *
from config import *

TARGET_RAW_LIST_FILE                = 'target_abs_raw_list.txt'
DATA_NAME                           = 'input.raw'
DEVICE_ROOT_DIR                     = '/data/local/tmp/snpebm'
SNPE_BENCH_SCRIPT                   = 'snpe_bench.sh'

assert args.hwc is not None
assert args.benchmark_device_id is not None

if args.benchmark_device_id is None:
    adb_cmd_prefix = 'adb '
else:
    adb_cmd_prefix = 'adb -s {} '.format(args.benchmark_device_id)

def prepare_fake_pb(model, model_name, checkpoint_dir):
    input_name = model.inputs[0].name.split(':')[0]
    output_name = model.outputs[0].name.split(':')[0]

    pb_filename = '{}.pb'.format(model_name)
    init_op = tf.initialize_all_variables()
    sess = tf.keras.backend.get_session()
    sess.run(init_op)
    my_graph=tf.get_default_graph()
    frozen_graph = util.freeze_session(sess, output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, checkpoint_dir, pb_filename, as_text=False)
    sess.close()

    return pb_filename, input_name, output_name

def prepare_fake_image(data_dir, prefix):
    img_h = args.target_resolution // args.scale
    img_w = WIDTH[img_h]
    img_fake = np.random.rand(img_h, img_w, args.channel_in)
    img_fake = img_fake * 255.0
    img_fake = img_fake.astype(np.float32)
    img_fake.tofile(os.path.join(data_dir, DATA_NAME))

    with open(os.path.join(data_dir,TARGET_RAW_LIST_FILE), 'w') as f:
        f.write(os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale), DATA_NAME))

def setup_local_asset(model, model_name, dlc_name, prefix):
    root_dir = os.path.join(args.data_dir, 'runtime')
    checkpoint_dir = os.path.join(root_dir, 'checkpoint')
    data_dir = os.path.join(root_dir, 'data', '{}p'.format(args.target_resolution // args.scale))
    benchmark_dir = os.path.join(root_dir, 'benchmark', prefix, model_name)

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)

    #convert to frozen graph (.pb) file
    pb_filename, input_name, output_name = prepare_fake_pb(model, model_name, checkpoint_dir)
    pb_filename = util.optimize_for_inference(pb_filename, input_name, output_name, checkpoint_dir)

    #convert to dlc file
    util.convert_to_dlc(pb_filename, input_name, output_name, dlc_name, checkpoint_dir, args.hwc[0], args.hwc[1], args.hwc[2])

    #setup dataset: a) check, b) save a randomly intialized frame
    prepare_fake_image(data_dir, prefix)

    #setup benchmark prerequisite: a) check, b) make a json file
    name = model_name
    benchmark = collections.OrderedDict()
    benchmark['Name'] = name
    benchmark['HostRootPath'] = os.path.abspath(benchmark_dir)
    benchmark['HostResultsDir'] = os.path.abspath(benchmark_dir)
    benchmark['DevicePath'] = '/data/local/tmp/snpebm'
    if args.benchmark_device_id == None:
        benchmark['Devices'] = []
    else:
        benchmark['Devices'] = [str(args.benchmark_device_id)]
    benchmark['HostName'] = 'localhost'
    benchmark['Runs'] = args.benchmark_iter_num
    benchmark['Model'] = collections.OrderedDict()
    benchmark['Model']['Name'] = name
    benchmark['Model']['Dlc'] = os.path.abspath(os.path.join(checkpoint_dir, dlc_name))
    benchmark['Model']['InputList'] = os.path.abspath(os.path.join(data_dir, TARGET_RAW_LIST_FILE))
    benchmark['Model']['Data'] = [os.path.abspath(data_dir)]
    benchmark['Runtimes'] = ['GPU_FP16']
    benchmark['Measurements'] = ['timing']
    benchmark['BufferTypes'] = ['ub_float']

    with open(os.path.join(benchmark_dir, '{}.json'.format(name)),'w') as outfile:
            json.dump(benchmark, outfile, indent=4)

def setup_remote_asset(model_name, dlc_name, prefix):
    #setup library
    os.environ['SNPE_TARGET_ARCH']='aarch64-android-clang6.0'
    os.environ['SNPE_TARGET_ARCH_OBJ_DIR']='arm64-v8a'
    os.environ['SNPE_ROOT']= os.path.abspath('../third_party/snpe')
    os.system(adb_cmd_prefix + 'shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin"')
    os.system(adb_cmd_prefix + 'shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib"')
    os.system(adb_cmd_prefix + 'shell "mkdir -p /data/local/tmp/snpeexample/dsp/lib"')
    os.system(adb_cmd_prefix + 'push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/* /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system(adb_cmd_prefix + 'push $SNPE_ROOT/lib/dsp/* /data/local/tmp/snpeexample/dsp/lib')
    #os.system(adb_cmd_prefix + 'push $SNPE_ROOT/examples/NativeCpp/SampleCode/libs/$SNPE_TARGET_ARCH_OBJ_DIR/* /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system(adb_cmd_prefix + 'push $SNPE_ROOT/../../android_dnn_sdk/snpe/latency_profiler/libs/$SNPE_TARGET_ARCH_OBJ_DIR/* /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    #os.system(adb_cmd_prefix + 'push $SNPE_ROOT/examples/NativeCpp/SampleCode/obj/local/$SNPE_TARGET_ARCH_OBJ_DIR/snpe-sample /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin')
    os.system(adb_cmd_prefix + 'push $SNPE_ROOT/../../android_dnn_sdk/snpe/latency_profiler/obj/local/$SNPE_TARGET_ARCH_OBJ_DIR/snpe-sample /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin')

    #setup dlc, data
    root_dir = os.path.join(args.data_dir, 'runtime')
    src_data_dir = os.path.join(root_dir, 'data', '{}p'.format(args.target_resolution // args.scale))
    dst_data_dir = os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale))
    src_dlc_path = os.path.join(root_dir, 'checkpoint', dlc_name)
    dst_dlc_path = os.path.join(DEVICE_ROOT_DIR, 'checkpoint', dlc_name)

    os.system(adb_cmd_prefix + 'shell "mkdir -p {}"'.format(os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale))))
    os.system(adb_cmd_prefix + 'shell "mkdir -p {}"'.format(os.path.join(DEVICE_ROOT_DIR, 'checkpoint')))
    os.system(adb_cmd_prefix + 'push {} {}'.format(src_dlc_path, dst_dlc_path))
    os.system(adb_cmd_prefix + 'push {}/* {}'.format(src_data_dir, dst_data_dir))

#measure end-to-end runtime: a) measure, b) save a log file
def measure_dnn_latency(model_name, prefix):
    src_log_dir = os.path.join(args.data_dir, 'runtime', 'log', prefix)
    os.makedirs(src_log_dir, exist_ok=True)
    dst_dlc_path = os.path.join(DEVICE_ROOT_DIR, 'checkpoint', '{}.dlc'.format(model_name))
    dst_data_list_path = os.path.join(DEVICE_ROOT_DIR, 'data', '{}p'.format(args.target_resolution // args.scale), TARGET_RAW_LIST_FILE)
    dst_log_dir = os.path.join(DEVICE_ROOT_DIR, 'log')
    log_filename = 'latency_{}p_{}p_{}.log'.format(args.target_resolution, args.target_resolution // args.scale, model_name)

    #os.system(adb_cmd_prefix + 'shell rm -rf {}'.format(dst_log_dir))

    if args.benchmark_runtime == 'CPU':
        runtime_opt = '-r cpu'
    #elif args.benchmark_runtime == 'CPU_IP8':
    #    runtime_opt = '-r cpu_ip8'
    elif args.benchmark_runtime == 'GPU':
        runtime_opt = '-r gpu'
    elif args.benchmark_runtime == 'GPU_FP16':
        runtime_opt = '-r gpu_fp16'
    else:
        raise NotImplementedError

    cmds = ['export SNPE_TARGET_ARCH=aarch64-android-clang6.0',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib',
            'export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin',
            'cd {}'.format(os.path.join(DEVICE_ROOT_DIR, 'data')),
            #'snpe-sample -d {} {} -i {}'.format(dst_dlc_path, runtime_opt, dst_data_list_path),
            'snpe-sample -d {} {} -n {} -i {} -o {} -l {}'.format(dst_dlc_path, runtime_opt, args.benchmark_iter_num, dst_data_list_path, dst_log_dir, log_filename),
            'exit']

    cmd_script_path = os.path.join(SNPE_BENCH_SCRIPT)
    with open(cmd_script_path, 'w') as cmd_script:
        cmd_script.write('#!/system/bin/sh'+'\n')
        for ln in cmds:
            cmd_script.write(ln + '\n')

    os.system(adb_cmd_prefix + 'push {} {}'.format(cmd_script_path, DEVICE_ROOT_DIR))
    os.system(adb_cmd_prefix + 'shell sh {}'.format(os.path.join(DEVICE_ROOT_DIR, SNPE_BENCH_SCRIPT)))
    os.system(adb_cmd_prefix + 'pull {} {}'.format(os.path.join(dst_log_dir, log_filename), src_log_dir))

#measure layer-wise runtime: a) measure, b) save a xls file
def measure_layer_latency(model_name, prefix):
    bench_json_path = os.path.abspath(os.path.join(args.data_dir, 'runtime', 'benchmark', prefix, model_name, '{}.json'.format(model_name)))
    cwd = os.getcwd()
    os.chdir("../third_party/snpe/benchmarks")
    os.system('/usr/bin/python2 snpe_bench.py -c {} -a -json'.format(bench_json_path))
    os.chdir(cwd)

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
    dlc_name = '{}.dlc'.format(model_name)

    setup_local_asset(model, model_name, dlc_name, prefix)
    setup_remote_asset(model_name, dlc_name, prefix)
    #measure_dnn_latency(model_name, prefix)
    measure_layer_latency(model_name, prefix)
