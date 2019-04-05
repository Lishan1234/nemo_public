import os
import sys
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='setup_libvpx')
parser.add_argument('--abi', type=str, default='arm64-v8a')
args = parser.parse_args()

if os.path.isfile('libvpx'):
    os.remove('libvpx')
    os.symlink('../../../../../libvpx', 'libvpx')
else:
    os.symlink('../../../../../libvpx', 'libvpx')

config_file = 'libvpx_android_configs/{}/vpx_config.h'.format(args.abi)
assert os.path.isfile(config_file)
if os.path.isfile('vpx_config.h'):
    os.remove('vpx_config.h')
    os.symlink(config_file, 'vpx_config.h')
else:
    os.symlink(config_file, 'vpx_config.h')


"""
SOURCE = os.path.abspath('../../../../../libvpx')
TARGET = os.path.abspath('libvpx')


srcs_file = 'libvpx_android_configs/{}/libvpx_srcs.txt'.format(args.abi)
srcs_file_ = 'libvpx_android_configs/libvpx_srcs.txt'.format(args.abi)
assert os.path.isfile(srcs_file)

with open(srcs_file) as config:
    for line in config:
        src = os.path.join(SOURCE, line.split('\n')[0])
        dest = os.path.join(TARGET, line.split('\n')[0])

        if os.path.isfile(dest):
            continue

        if not os.path.isfile(src):
            print('no file: {}'.format(src))
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.symlink(src,dest)
        #copyfile(src, dest) - replace by symbolic links

with open(srcs_file_) as config:
    for line in config:
        src = os.path.join(SOURCE, line.split('\n')[0])
        dest = os.path.join(TARGET, line.split('\n')[0])

        if os.path.isfile(dest):
            continue

        if not os.path.isfile(src):
            print('no file: {}'.format(src))
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.symlink(src,dest)
        #copyfile(src, dest) - replace by symbolic links
"""

