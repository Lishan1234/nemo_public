import argparse
import os

parser = argparse.ArgumentParser(description='MnasNet')

parser.add_argument('--train_data', type=str, default='291')
parser.add_argument('--valid_data', type=str, default='Set5')
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument('--num_patch', type=int, default=50000)
parser.add_argument('--enable_debug', action='store_true')
parser.add_argument('--scale', type=int, default=4) #for image based dataset
parser.add_argument('--hr', type=int, default=1080) #for video based dataset

args = parser.parse_args()
