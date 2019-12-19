import argparse
import sys
import os

from tool.adb import *
from tool.snpe import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #video
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #device
    parser.add_argument('--abi', type=str, default='arm64-v8a')
    parser.add_argument('--device_id', type=str, required=True)

    args = parser.parse_args()

    #setup directory
    device_root_dir = '/data/local/tmp'
    device_bin_dir = os.path.join(device_root_dir, 'bin')
    device_libs_dir = os.path.join(device_root_dir, 'libs')
    adb_mkdir(device_bin_dir, args.device_id)
    adb_mkdir(device_libs_dir, args.device_id)

    #setup vpxdec
    vpxdec_path = os.path.join('libs', args.abi, 'vpxdec')
    adb_push(device_bin_dir, vpxdec_path)

    #setup library
    c_path = os.path.join('libs', args.abi, 'libc++_shared.so')
    snpe_path = os.path.join('libs', args.abi, 'libSNPE.so')
    symphony_path = os.path.join('libs', args.abi, 'libsymphony-cpu.so')
    adb_push(device_libs_dir, c_path)
    adb_push(device_libs_dir, snpe_path)
    adb_push(device_libs_dir, symphony_path)
