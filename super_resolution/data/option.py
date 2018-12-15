import argparse
import os

parser = argparse.ArgumentParser(description='MnasNet')

parser.add_argument('--train_data', type=str, default='291')
parser.add_argument('--valid_data', type=str, default='Set5')
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--num_patch', type=int, default=100000)
parser.add_argument('--enable_debug', action='store_true')

args = parser.parse_args()
