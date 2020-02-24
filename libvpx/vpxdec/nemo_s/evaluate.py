import argparse
import sys
import time

from tool.adb import *
from tool.snpe import *
from tool.video import *
from dnn.model.nemo_s import NEMO_S
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #dataset
    parser.add_argument('--filter_type', type=str, default='uniform')
    parser.add_argument('--filter_fps', type=float, default=1.0)

    #model
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--aps_class', type=str, choices=['nemo', 'random', 'uniform'], required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)
    #parser.add_argument('--runtime', type=str, required=True)

    #experiment
    parser.add_argument('--sleep', type=float, default=0)

    args = parser.parse_args()

    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_file))
    assert(os.path.exists(hr_video_file))

    #setup directory
    content = os.path.basename(args.dataset_dir)
    device_root_dir = os.path.join('/data/local/tmp', content)

    #model
    lr_video_profile = profile_video(lr_video_file)
    hr_video_profile = profile_video(hr_video_file)
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]
    scale = hr_video_profile['height'] // lr_video_profile['height']
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)
    model = nemo_s.build_model(apply_clip=True)

    #setup a cache profile
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random
    cache_profile_name = '{}_{}.profile'.format(aps_class.NAME1, args.threshold)

    #case 1: decode
    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name)
    device_log_dir= os.path.join(device_root_dir, 'log', args.lr_video_name)
    device_script_file = os.path.join(device_script_dir, 'decode_frame.sh')
    device_log_file = os.path.join(device_log_dir, 'latency.txt')
    host_log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, args.device_id)
    host_log_file = os.path.join(host_log_dir, 'latency.txt')
    os.makedirs(host_log_dir, exist_ok=True)

    start_time = time.time()
    command = 'adb shell sh {}'.format(device_script_file)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    adb_pull(device_log_file, host_log_file)
    end_time = time.time()
    print("decode takes {}sec".format(end_time - start_time))

    time.sleep(args.sleep)

    #case 2: online sr
    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name)
    device_log_dir= os.path.join(device_root_dir, 'log', args.lr_video_name, model.name)
    device_script_file = os.path.join(device_script_dir, 'online_sr_latency.sh')
    device_log_file = os.path.join(device_log_dir, 'latency.txt')
    host_log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, model.name, args.device_id)
    host_log_file = os.path.join(host_log_dir, 'latency.txt')
    os.makedirs(host_log_dir, exist_ok=True)

    start_time = time.time()
    command = 'adb shell sh {}'.format(device_script_file)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    adb_pull(device_log_file, host_log_file)
    end_time = time.time()
    print("online sr takes {}sec".format(end_time - start_time))

    time.sleep(args.sleep)

    #case 3: online sr (real-time)
    #TODO

    #case 4: online cache
    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name)
    device_log_dir= os.path.join(device_root_dir, 'log', args.lr_video_name, model.name, cache_profile_name)
    device_script_file = os.path.join(device_script_dir, 'online_profile_cache_latency.sh')
    device_log_file = os.path.join(device_log_dir, 'latency.txt')
    host_log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, model.name, cache_profile_name, args.device_id)
    host_log_file = os.path.join(host_log_dir, 'latency.txt')
    os.makedirs(host_log_dir, exist_ok=True)

    start_time = time.time()
    command = 'adb shell sh {}'.format(device_script_file)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    adb_pull(device_log_file, host_log_file)
    end_time = time.time()
    print("online cache takes {}sec".format(end_time - start_time))
