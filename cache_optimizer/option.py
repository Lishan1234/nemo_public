import argparse
import os

parser = argparse.ArgumentParser(description='Cache optimizer')

#options for CRA
parser.add_argument('--cra_content_dir', type=str, required=True)
parser.add_argument('--cra_input_video_name', type=str, required=True)
parser.add_argument('--cra_dnn_video_name', type=str, required=True)
parser.add_argument('--cra_compare_video_name', type=str, required=True)
parser.add_argument('--cra_total_frames', default=0, type=int)
parser.add_argument('--cra_num_cores', default=1, type=int)
#parser.add_argument('--cra.video_format', default="webm", type=str)

#options for all modules

args = parser.parse_args()
