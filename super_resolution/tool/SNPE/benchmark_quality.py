import sys
import os
sys.path.append('../../')
from option import args
from importlib import import_module
import math
import scipy.misc

import numpy as np

TARGET_RAW_LIST_FILE = 'target_raw_list.txt'
SNPE_BENCH_SCRIPT='snpe_bench.sh'
RESOLUTION={240: (240, 426)}

model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
model = model_builder.build()
output_name = model.outputs[0].name
model_name = model_builder.get_name()

os.makedirs('tmp', exist_ok=True)

#Setup pre-requisites
def setup_prerequisites():
    os.system('export SNPE_TARGET_ARCH=aarch64-android-gcc4.9')
    os.system('export SNPE_TARGET_STL=libgnustl_shared.so')

    os.system('adb shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin"')
    os.system('adb shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib"')
    os.system('adb shell "mkdir -p /data/local/tmp/snpeexample/dsp/lib"')
    os.system('adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/$SNPE_TARGET_STL \
          /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system('adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/libsymphony-cpu.so \
          /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system('adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/libsymphonypower.so \
          /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system('adb push $SNPE_ROOT/lib/dsp/libsnpe_dsp_skel.so \
          /data/local/tmp/snpeexample/dsp/lib')
    os.system('adb push $SNPE_ROOT/lib/dsp/libsnpe_dsp_domains_skel.so \
          /data/local/tmp/snpeexample/dsp/lib')
    os.system('adb push $SNPE_ROOT/lib/dsp/libsnpe_dsp_v65_domains_v2_skel.so \
          /data/local/tmp/snpeexample/dsp/lib')
    os.system('adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/libsnpe_adsp.so \
          /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system('adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/libsnpe_dsp_domains.so \
          /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system('adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/libSNPE.so \
          /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib')
    os.system('adb push $SNPE_ROOT/bin/$SNPE_TARGET_ARCH/snpe-net-run \
          /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin')

#Copy dlc and data
def setup_dlc_data(copy_data=False):
    src_dlc_path = os.path.join(args.snpe_project_root, 'custom', args.train_data, 'models', model_name, 'dlc', '{}.dlc'.format(model_name))
    src_quantized_dlc_path = os.path.join(args.snpe_project_root, 'custom', args.train_data, 'models', model_name, 'dlc', 'quantized_{}.dlc'.format(model_name))
    src_data_path = os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', '{}p'.format(args.lr))
    src_data_list_path = os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', TARGET_RAW_LIST_FILE)

    dst_root_dir = '/data/local/tmp/snpebm'
    dst_dlc_path = os.path.join(dst_root_dir, args.train_data, 'models', model_name, 'dlc', '{}.dlc'.format(model_name))
    dst_quantized_dlc_path = os.path.join(dst_root_dir, args.train_data, 'models', model_name, 'dlc', 'quantized_{}.dlc'.format(model_name))
    dst_data_path = os.path.join(dst_root_dir, args.train_data, 'data', '{}p'.format(args.lr))
    dst_data_list_path = os.path.join(dst_root_dir, args.train_data, 'data', TARGET_RAW_LIST_FILE)

    os.system('adb shell "mkdir -p {}"'.format(os.path.join(dst_root_dir, args.train_data, 'data')))
    os.system('adb shell "mkdir -p {}"'.format(os.path.join(dst_root_dir, args.train_data, 'models', model_name, 'dlc')))
    os.system('adb push {} {}'.format(src_dlc_path, dst_dlc_path))
    os.system('adb push {} {}'.format(src_quantized_dlc_path, dst_quantized_dlc_path))

    if copy_data:
        os.system('adb push {} {}'.format(src_data_path, dst_data_path))
        os.system('adb push {} {}'.format(src_data_list_path, dst_data_list_path))

def execute_network(runtime):
    assert runtime in ['FIX_CPU', 'GPU', 'GPU_FP16']

    dst_root_dir = '/data/local/tmp/snpebm'
    if runtime == 'FIX_CPU':
        dlc_name = 'quantized_{}.dlc'.format(model_name)
        runtime_opt = '--use_fxp_cpu'
        output_dir = '{}p_8bit'.format(args.lr*args.scale)
    elif runtime == 'GPU_FP16':
        dlc_name = '{}.dlc'.format(model_name)
        runtime_opt = '--use_gpu --gpu_mode float16'
        output_dir = '{}p_16bit'.format(args.lr*args.scale)
    elif runtime == 'GPU':
        dlc_name = '{}.dlc'.format(model_name)
        runtime_opt = '--use_gpu --gpu_mode default'
        output_dir = '{}p_32bit'.format(args.lr*args.scale)

    cmds = ['export SNPE_TARGET_ARCH=aarch64-android-gcc4.9',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib',
            'export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin',
            'cd {}'.format(os.path.join(dst_root_dir, args.train_data, 'data')),
            'snpe-net-run --container {} {} --input_list {} --output_dir ./{}'.format(os.path.join(dst_root_dir, args.train_data, 'models', model_name, 'dlc', dlc_name), runtime_opt, os.path.join(dst_root_dir, args.train_data, 'data', TARGET_RAW_LIST_FILE), output_dir),
            'exit']

    cmd_script_path = os.path.join('tmp', SNPE_BENCH_SCRIPT)
    with open(cmd_script_path, 'w') as cmd_script:
        cmd_script.write('#!/system/bin/sh'+'\n')
        for ln in cmds:
            cmd_script.write(ln + '\n')

    os.makedirs(os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', output_dir), exist_ok=True)
    os.system('adb push {} {}'.format(cmd_script_path, dst_root_dir))
    os.system('adb shell sh {}'.format(os.path.join(dst_root_dir, SNPE_BENCH_SCRIPT)))
    os.system('adb pull {} {}'.format(os.path.join(dst_root_dir, args.train_data, 'data', output_dir), os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', output_dir)))

def psnr(img1, img2, max_value):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * math.log10(max_value / math.sqrt(mse))

#Calculate PSNR
def calculate_psnr(runtime):
    src_data_list_path = os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', TARGET_RAW_LIST_FILE)
    src_data_path = os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', '{}p'.format(args.lr*args.scale))

    if runtime == 'FIX_CPU':
        output_dir = os.path.abspath(os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', '{}p_8bit'.format(args.lr*args.scale)))
    elif runtime == 'GPU_FP16':
        output_dir = os.path.abspath(os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', '{}p_16bit'.format(args.lr*args.scale)))
    elif runtime == 'GPU':
        output_dir = os.path.abspath(os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', '{}p_32bit'.format(args.lr*args.scale)))

    with open(src_data_list_path, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]

    psnr_results = []
    for idx, val in enumerate(input_files):
        #Read a super-resolution output
        cur_results_file = os.path.join(output_dir, 'Result_' + str(idx), '{}.raw'.format(output_name))
        if not os.path.isfile(cur_results_file):
            raise RuntimeError('missing results file: ' + cur_results_file)

        target_array = np.fromfile(cur_results_file, dtype=np.float32)
        h, w = RESOLUTION[args.lr]
        target_array = np.reshape(target_array, (h*args.scale, w*args.scale, 3))

        #Test: raw input
        """
        test_lr_raw = os.path.join(args.snpe_project_root, 'custom', args.train_data, 'data', '{}p'.format(args.lr), '0001.raw')
        target_array = np.fromfile(test_lr_raw, dtype=np.float32)
        target_array = np.reshape(target_array, (h, w, 3))
        scipy.misc.imsave(os.path.join(output_dir, 'lr_{:04d}.png'.format(idx+1)), target_array)
        sys.exit()
        """

        #Read a reference frame
        cur_reference_file = os.path.abspath(os.path.join(src_data_path, '{:04d}.raw'.format(idx+1)))
        reference_array = np.fromfile(cur_reference_file, dtype=np.float32)
        reference_array = np.reshape(reference_array, (h*args.scale, w*args.scale, 3))

        #Calculate PSNR
        psnr_result = psnr(target_array, reference_array, 1.0)
        psnr_results.append(psnr_result)

        print('[{}] ({}/{}): {}dB'.format(runtime, idx+1, len(input_files), round(psnr_result, 3)))

    log_path = os.path.join(output_dir, 'quality.log')
    with open(log_path, 'w') as log_script:
        log_script.write(str(round(np.mean(psnr_results),3))+'\n')
        for psnr_result in psnr_results:
            log_script.write(str(round(psnr_result, 3))+'\n')

#TODO: Should we use userbuffer_tf8?

if __name__ == '__main__':
    if args.snpe_copy_lib:
        setup_prerequisites()
    setup_dlc_data(args.snpe_copy_data)

    #runtimes = ['FIX_CPU', 'GPU_FP16', 'GPU']
    runtimes = ['GPU']
    for runtime in runtimes:
        execute_network(runtime)
        calculate_psnr(runtime)
