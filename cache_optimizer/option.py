import argparse
import os

parser = argparse.ArgumentParser(description='Cache optimizer')

#options
parser.add_argument('--vpxdec_path', type=str, required=True)
parser.add_argument('--content_dir', type=str, required=True)
parser.add_argument('--input_video_name', type=str, required=True)
parser.add_argument('--dnn_video_name', type=str, required=True)
parser.add_argument('--compare_video_name', type=str, required=True)

#options for CRA
parser.add_argument('--cra_total_frames', default=None, type=int)
parser.add_argument('--cra_num_cores', default=1, type=int)
parser.add_argument('--cra_num_threads', default=1, type=int)
#parser.add_argument('--cra.video_format', default="webm", type=str)

#options for all modules

args = parser.parse_args()
