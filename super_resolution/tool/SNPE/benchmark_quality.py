import sys
import os
sys.path.append('../../')
from option import args
from importlib import import_module

TARGET_RAW_LIST_FILE = 'target_raw_list.txt'
SNPE_BENCH_SCRIPT='snpe_bench.sh'

model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
model_name = model_builder.get_name()

os.makedirs('tmp', exist_ok=True)

#Setup pre-requisites
if args.snpe_copy_lib:
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

if args.snpe_copy_data:
    os.system('adb push {} {}'.format(src_data_path, dst_data_path))
    os.system('adb push {} {}'.format(src_data_list_path, dst_data_list_path))

#Run FIX_CPU (8bit quality)
cmds = ['export SNPE_TARGET_ARCH=aarch64-android-gcc4.9',
        'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib',
        'export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin',
        'cd {}'.format(os.path.join(dst_root_dir, args.train_data, 'data')),
        'snpe-net-run --container {} --input_list {}'.format(os.path.join(dst_root_dir, args.train_data, 'models', model_name, 'dlc', 'quantized_{}.dlc'.format(model_name)), os.path.join(dst_root_dir, args.train_data, 'data', TARGET_RAW_LIST_FILE)),
        'exit']

cmd_script_path = os.path.join('tmp', SNPE_BENCH_SCRIPT)
with open(cmd_script_path, 'w') as cmd_script:
    cmd_script.write('#!/system/bin/sh' '\n')
    for ln in cmds:
        cmd_script.write(ln + '\n')
    #os.chmod(cmd_script_path, 0o555)

os.system('adb push {} {}'.format(cmd_script_path, dst_root_dir))
os.system('adb shell sh {}'.format(os.path.join(dst_root_dir, SNPE_BENCH_SCRIPT)))

#refer snpebm_bm.py script generation code

#'adb -H %s -s %s %s' % (host, dev_serial, cmd)

#Run GPU (32bit quality)

#Run GPU_FP16 (16bit quality)

#Measure PSNR quality

#Save a log
