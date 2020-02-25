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

    start_time = time.time()
    time.sleep(args.sleep)
    end_time = time.time()
    print("sleep takes {}sec".format(end_time - start_time))

    time.sleep(args.sleep)
