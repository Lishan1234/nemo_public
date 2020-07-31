import argparse
import sys

import nemo

from nemo.tool.adb import *
from nemo.tool.snpe import *
from nemo.tool.video import *
from nemo.tool.mobile import *
from nemo.tool.libvpx  import get_num_threads

import nemo.dnn.model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #path
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device_data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--video_name', type=str, required=True)

    #model
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')
    parser.add_argument('--train_type', type=str, default='finetune_video')

    #anchor point selector
    parser.add_argument('--algorithm', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)

    args = parser.parse_args()

    #setup directory
    device_root_dir = os.path.join(args.device_data_dir, args.content)

    #setup a dnn
    video_path = os.path.join(args.data_dir, args.content, 'video', args.video_name)
    video_profile = profile_video(video_path)
    input_shape = [1, video_profile['height'], video_profile['width'], 3]
    scale = args.output_height // video_profile['height']
    model = nemo.dnn.model.build(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type, apply_clip=True)

    #download a NEMO result
    device_log_path = os.path.join(device_root_dir, 'log', args.video_name, model.name, args.algorithm, 'latency.txt')
    host_log_dir = os.path.join(args.data_dir, args.content, 'log', args.video_name, model.name, args.algorithm, id_to_name(args.device_id))
    os.makedirs(host_log_dir, exist_ok=True)
    try:
        adb_pull(device_log_path, host_log_dir, args.device_id)
    except:
        pass

    device_log_path = os.path.join(device_root_dir, 'log', args.video_name, model.name, args.algorithm, 'metadata.txt')
    host_log_dir = os.path.join(args.data_dir, args.content, 'log', args.video_name, model.name, args.algorithm, id_to_name(args.device_id))
    os.makedirs(host_log_dir, exist_ok=True)
    try:
        adb_pull(device_log_path, host_log_dir, args.device_id)
    except:
        pass

    #setup a per-frame SR result
    device_log_path = os.path.join(device_root_dir, 'log', args.video_name, model.name, 'latency.txt')
    host_log_dir = os.path.join(args.data_dir, args.content, 'log', args.video_name, model.name, id_to_name(args.device_id))
    os.makedirs(host_log_dir, exist_ok=True)
    adb_pull(device_log_path, host_log_dir, args.device_id)

    device_log_path = os.path.join(device_root_dir, 'log', args.video_name, model.name, 'metadata.txt')
    host_log_dir = os.path.join(args.data_dir, args.content, 'log', args.video_name, model.name, id_to_name(args.device_id))
    os.makedirs(host_log_dir, exist_ok=True)
    adb_pull(device_log_path, host_log_dir, args.device_id)
