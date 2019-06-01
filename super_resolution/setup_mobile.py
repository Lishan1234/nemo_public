import os, glob, random, sys, time, argparse
import utility as util
import re
import shutil

import tensorflow as tf
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Video dataset")

parser.add_argument('--data_dir', type=str, default="/ssd1/data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--device_id', type=str, default=None)

args = parser.parse_args()

#remove an existing dataset in mobile
#....

#copy a new dataset

