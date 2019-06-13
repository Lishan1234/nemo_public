import os, glob, random, sys, time, argparse
import re
import shutil

sys.path.insert(0, './')
from config import WIDTH
import utility as util

parser = argparse.ArgumentParser(description="Runtime test")

parser.add_argument('--target_resolution', type=int, default=1080) #target HR resolution
parser.add_argument('--device_id', type=str, required=True)

args = parser.parse_args()

num_blocks = [1, 2, 4, 8]
num_filters = [4, 8, 16, 32]
scales = [2, 3, 4]

count = 0
for scale in scales:
    h = args.target_resolution // scale
    w = WIDTH[h]
    for num_block in num_blocks:
        for num_filter in num_filters:
            cmd = 'python test_snpe_runtime.py --num_blocks {} --num_filters {} --scale {} --hwc {},{},{} --benchmark_device_id {} --benchmark_iter_num 20'.format(num_block, num_filter, scale, h, w, 3, args.device_id)
            os.system(cmd)
            #util.print_progress(count, len(scales) * len(num_blocks) * len(num_filters), 'Test Progress:', 'Complete', 1, 50)
