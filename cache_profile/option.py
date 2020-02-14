import argparse
import os

parser = argparse.ArgumentParser(description='Cache optimizer')

#options
parser.add_argument('--vpxdec_path', type=str, required=True)
parser.add_argument('--content_dir', type=str, required=True)
parser.add_argument('--input_video_name', type=str, required=True)
parser.add_argument('--dnn_video_name', type=str, required=True)
parser.add_argument('--compare_video_name', type=str, required=True)
parser.add_argument('--num_frames', default=None, type=int)

#options for CRA
parser.add_argument('--cra_num_cores', default=1, type=int)
parser.add_argument('--cra_num_threads', default=1, type=int)
#parser.add_argument('--cra.video_format', default="webm", type=str)

#options for APS
parser.add_argument('--aps_threshold', default=1.0, type=float)
parser.add_argument('--aps_start_idx', default=None, type=int)
parser.add_argument('--aps_end_idx', default=None, type=int)

#options for all modules

args = parser.parse_args()
