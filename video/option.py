import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default=None)
parser.add_argument('--content_name', type=str, default=None)

#Downloader
parser.add_argument('--url', type=str, default=None)

#Encoder
parser.add_argument('--gop', type=str, default=None)
parser.add_argument('--num_threads', type=int, default=None)
parser.add_argument('--start_time', type=int, default=None)
parser.add_argument('--duration', type=int, default=None)
parser.add_argument('--video_fmt', type=str, default=None)

args = parser.parse_args()
