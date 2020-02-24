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

    #device
    parser.add_argument('--device_id', type=str, required=True)
    #parser.add_argument('--runtime', type=str, required=True)

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

    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name, model.name)
    device_log_dir= os.path.join(device_root_dir, 'log', args.lr_video_name, model.name)
    device_script_file = os.path.join(device_script_dir, 'online_sr_latency.sh')
    device_log_file = os.path.join(device_log_dir, 'latency.txt')
    host_log_dir = os.path.join(args.dataset_dir, 'log', args.lr_video_name, model.name, args.device_id)
    host_log_file = os.path.join(host_log_dir, 'latency.txt')
    os.makedirs(host_log_dir, exist_ok=True)

    start_time = time.time()
    command = 'adb shell sh {}'.format(device_script_file)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    #os.system(command)
    adb_pull(device_log_file, host_log_file)
    end_time = time.time()
    print("online sr takes {}sec".format(end_time - start_time))
