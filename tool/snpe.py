import subprocess
import os
import sys
import glob
import collections
import json

import numpy as np
import imageio
import tensorflow as tf

from tool.adb import adb_pull

DEVICE_ROOTDIR = '/data/local/tmp/snpebm'
BENCHMARK_CONFIG_NAME = 'benchmark.json'
BENCHMARK_RAW_LIST = 'target_raw_list.txt'
OPT_4_INFERENCE_SCRIPT  = 'optimize_for_inference.py'

#check tensorflow, snpe directory
TENSORFLOW_ROOT = os.path.join(os.environ['MOBINAS_CODE_ROOT'], 'third_party', 'tensorflow')
SNPE_ROOT = os.path.join(os.environ['MOBINAS_CODE_ROOT'], 'third_party', 'snpe')
assert(os.path.exists(TENSORFLOW_ROOT))
assert(os.path.exists(SNPE_ROOT))

def decode_raw(filepath, width, height, channel, precision):
    file = tf.io.read_file(filepath)
    image = tf.decode_raw(file, precision)
    image = tf.reshape(image, [height, width, channel])
    #return image, filepath
    return image

def raw_dataset(image_dir, width, height, channel, exp, precision):
    m = re.compile(exp)
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if m.search(f)])
    #images = sorted(glob.glob('{}/{}'.format(image_dir, pattern)))
    #images = sorted(glob.glob('{}/[0-9][0-9][0-9][0-9].raw'.format(image_dir)))
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda x: decode_raw(x, width, height, channel, precision), num_parallel_calls=AUTOTUNE)
    return ds, len(images)

def summary_raw_dataset(lr_image_dir, sr_image_dir, hr_image_dir, width, height, channel, scale, exp, repeat_count=1, precision=tf.uint8):
    lr_ds, length = raw_dataset(lr_image_dir, width, height, channel, exp, precision)
    hr_ds, _ = raw_dataset(hr_image_dir, width * scale, height * scale, channel, exp, precision)
    sr_ds, _ = raw_dataset(sr_image_dir, width * scale, height * scale, channel, exp, precision)
    ds = tf.data.Dataset.zip((lr_ds, sr_ds, hr_ds))
    ds = ds.batch(1)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def raw_quality(lr_raw_dir, sr_raw_dir, hr_raw_dir, nhwc, scale, precision=tf.float32):
    bilinear_psnr_values= []
    sr_psnr_values = []
    summary_raw_ds = summary_raw_dataset(lr_raw_dir, sr_raw_dir, hr_raw_dir, nhwc[1], nhwc[2],
                                                    scale, precision=precision)
    for idx, imgs in enumerate(summary_raw_ds):
        lr = imgs[0][0]
        sr = imgs[1][0]
        hr = imgs[2][0]

        if precision == tf.float32:
            hr = tf.clip_by_value(hr, 0, 255)
            hr = tf.round(hr)
            hr = tf.cast(hr, tf.uint8)
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)

        bilinear = resolve_bilinear_tf(lr, nhwc[1] * scale, nhwc[2] * scale)
        bilinear_psnr_value = tf.image.psnr(bilinear, hr, max_val=255)[0].numpy()
        bilinear_psnr_values.append(bilinear_psnr_value)
        sr_psnr_value = tf.image.psnr(sr, hr, max_val=255)[0].numpy()
        sr_psnr_values.append(sr_psnr_value)
        print('{} frame: PSNR(SR)={:.2f}, PSNR(Bilinear)={:.2f}'.format(idx, sr_psnr_value, bilinear_psnr_value))
    print('Summary: PSNR(SR)={:.2f}, PSNR(Bilinear)={:.2f}'.format(np.average(sr_psnr_values), np.average(bilinear_psnr_values)))

    return sr_psnr_values, bilinear_psnr_values

def check_python_version():
    #check python version
    python_version = sys.version_info
    if not (python_version[0] == 3 and python_version[1] == 4):
        raise RuntimeError('Unsupported Python version: {}'.format(python_version))

def snpe_dlc_viewer(dlc_path, html_path):
    check_python_version()
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
    check_python_version()
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

def snpe_benchmark(json_file):
    check_python_version()
    setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(SNPE_ROOT, TENSORFLOW_ROOT)
    snpe_cmd = 'python {}/benchmarks/snpe_bench.py -c {} -json'.format(SNPE_ROOT, json_file)

    cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

def snpe_benchmark_output(json_file, host_dir, output_name):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        device_id = json_data['Devices'][0]
        device_dir = os.path.join(json_data['DevicePath'], json_data['Name'])
        raw_list_file = json_data['Model']['InputList']

    with open(raw_list_file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            device_file = os.path.join(device_dir, 'output', 'Result_{}'.format(idx), '{}.raw'.format(output_name))
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
    status = checkpoint.restore(latest_checkpoint)
    sess = tf.keras.backend.get_session()
    status.initialize_or_restore(sess)
    graph = tf.get_default_graph()
    frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, checkpoint_dir, pb_name, as_text=False)

    #optimize a frozen graph
    opt_pb_path = os.path.join(checkpoint_dir, opt_pb_name)
    input_name = model.inputs[0].name.split(':')[0]
    output_name = model.outputs[0].name.split(':')[0]
    optimize_for_inference(pb_path, opt_pb_path, input_name, output_name)

    #convert to a dlc (.dlc)
    dlc_path = os.path.join(checkpoint_dir, dlc_name)
    snpe_tensorflow_to_dlc(pb_path, dlc_path, input_name, output_name, nhwc)
    snpe_tensorflow_to_dlc(pb_path, dlc_path, input_name, output_name, nhwc)

    #convcert to a quantized dlc (.quantized.dlc)
    #TODO

    #visualize a dlc
    html_path = os.path.join(checkpoint_dir, '{}.html'.format(dlc_name))
    snpe_dlc_viewer(dlc_path, html_path)

    return dlc_dict

def snpe_benchmark_config(device_id, runtime, model_name, dlc_file, log_dir, raw_dir, perf='default', exp='*.raw'):
    result_dir = os.path.join(log_dir, device_id, runtime)
    json_file = os.path.join(result_dir, BENCHMARK_CONFIG_NAME)
    raw_list_file = os.path.join(result_dir, BENCHMARK_RAW_LIST)
    os.makedirs(result_dir, exist_ok=True)

    with open(raw_list_file, 'w') as f:
        raw_files = sorted(glob.glob('{}/{}'.format(raw_dir, exp)))
        for raw_file in raw_files:
            f.write('{}/{}\n'.format(os.path.basename(raw_dir), os.path.basename(raw_file)))

    benchmark = collections.OrderedDict()
    benchmark['Name'] = model_name
    benchmark['HostRootPath'] = os.path.abspath(log_dir)
    benchmark['HostResultsDir'] = os.path.abspath(result_dir)
    benchmark['DevicePath'] = DEVICE_ROOTDIR
    benchmark['Devices'] = [device_id]
    benchmark['HostName'] = 'localhost'
    benchmark['Runs'] = 1
    benchmark['Model'] = collections.OrderedDict()
    benchmark['Model']['Name'] = model_name
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

def snpe_benchmark_random_config(device_id, runtime, model_name, dlc_file, log_dir, total_num=20, perf='default'):
    result_dir = os.path.join(log_dir, device_id, runtime)
    json_file = os.path.join(result_dir, BENCHMARK_CONFIG_NAME)
    os.makedirs(result_dir, exist_ok=True)

    benchmark = collections.OrderedDict()
    benchmark['Name'] = model_name
    benchmark['HostRootPath'] = os.path.abspath(log_dir)
    benchmark['HostResultsDir'] = os.path.abspath(result_dir)
    benchmark['DevicePath'] = DEVICE_ROOTDIR
    benchmark['Devices'] = [device_id]
    benchmark['HostName'] = 'localhost'
    benchmark['Runs'] = 1
    benchmark['Model'] = collections.OrderedDict()
    benchmark['Model']['Name'] = model_name
    benchmark['Model']['Dlc'] = dlc_file
    benchmark['Model']['RandomInput'] = total_num
    benchmark['Runtimes'] = [runtime]
    benchmark['Measurements'] = ['timing']
    benchmark['ProfilingLevel'] = 'detailed'
    benchmark['BufferTypes'] = ['float']

    with open(json_file, 'w') as f:
        json.dump(benchmark, f, indent=4)

    return json_file

def snpe_benchmark_result(device_id, runtime,  model, lr_image_dir, hr_image_dir, log_dir, perf='default'):
    #quality
    lr_raw_dir = os.path.join(lr_image_dir, 'raw')
    sr_raw_dir = os.path.join(lr_image_dir, model.name, runtime, 'raw')
    hr_raw_dir = os.path.join(hr_image_dir, 'raw')
    sr_psnr_values, bilinear_psnr_values = raw_quality(lr_raw_dir, sr_raw_dir, hr_raw_dir, model.nhwc, model.scale)
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    with open(quality_log_file, 'w') as f:
        f.write('Average\t{:.2f}\t{:.2f}\n'.format(np.average(sr_psnr_values), np.average(bilinear_psnr_values)))
        for idx, psnr_values in enumerate(list(zip(sr_psnr_values, bilinear_psnr_values))):
            f.write('{}\t{:.2f}\t{:.2f}\n'.format(idx, psnr_values[0], psnr_values[1]))
    avg_bilinear_psnr = np.average(bilinear_psnr_values)
    avg_sr_psnr = np.average(sr_psnr_values)

    #latency
    result_json_file = os.path.join(log_dir, 'latest_results', 'benchmark_stats_{}.json'.format(model.name))
    assert(os.path.exists(result_json_file))
    with open(result_json_file, 'r') as f:
        json_data = json.load(f)
        avg_latency = float(json_data['Execution_Data']['GPU_FP16']['Total Inference Time']['Avg_Time']) / 1000

    #size
    config_json_file = os.path.join(log_dir, 'benchmark.json')
    with open(config_json_file, 'r') as f:
        json_data = json.load(f)
        dlc_file = json_data['Model']['Dlc']
        size = os.path.getsize(dlc_file) / 1000

    #log
    summary_log_file = os.path.join(log_dir, 'summary.txt')
    with open(summary_log_file, 'w') as f:
        f.write('PSNR (dB)\t{:.2f}\t{:.2f}\n'.format(avg_sr_psnr, avg_bilinear_psnr))
        f.write('Latency (msec)\t{:.2f}\n'.format(avg_latency))
        f.write('Size (KB)\t{:.2f}\n'.format(size))

    return avg_sr_psnr, avg_bilinear_psnr, avg_latency, size

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
