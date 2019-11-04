import argparse

parser = argparse.ArgumentParser()

#directory, path
parser.add_argument('--video_dir', type=str, required=True)
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--ffmpeg_path', type=str, required=True)

#video metadata
parser.add_argument('--video_format', type=str, default='webm')
parser.add_argument('--video_start_time', type=int, default=None)
parser.add_argument('--video_duration', type=int, default=None)
parser.add_argument('--filter_type', type=str, choices=['uniform', 'keyframes',], default=1.0)
parser.add_argument('--filter_fps', type=float, default=1.0)
parser.add_argument('--upsample', type=str, default='bilinear')

#training: dataset
parser.add_argument('--resolution_pairs', nargs='+', type=str, required=True) #e.g., 240,1080
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--load_on_memory', action='store_true')


#training: architecture
#TODO

#training: hyper-parameters
#TODO

args = parser.parse_args()

#parse args.resolution_parse
args.resolution_dict = {}
for resolution_pair in args.resolution_pairs:
    input_resolution = int(resolution_pair.split(',')[0])
    output_resolution = int(resolution_pair.split(',')[1])
    args.resolution_dict[input_resolution] = output_resolution
