import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default=None)
parser.add_argument('--content_name', type=str, default=None)

#Downloader
parser.add_argument('--url', type=str, default=None)

#Encoder

args = parser.parse_args()
