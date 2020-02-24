import argparse
import sys

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

    #codec
    parser.add_argument('--threads', type=str, default=4)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    assert(os.path.exists(lr_video_file))
    assert(os.path.exists(hr_video_file))

    #load a dnn
    lr_video_profile = profile_video(lr_video_file)
    hr_video_profile = profile_video(hr_video_file)
    nhwc = [1, lr_video_profile['height'], lr_video_profile['width'], 3]
    scale = hr_video_profile['height'] // lr_video_profile['height']

    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale, args.upsample_type)
    model = nemo_s.build_model(apply_clip=True)
    model.nhwc = nhwc
    model.scale = scale
    ffmpeg_option = FFmpegOption(args.filter_type, args.filter_fps, None)
    checkpoint_dir = os.path.join(args.dataset_dir, 'checkpoint', ffmpeg_option.summary(args.lr_video_name), model.name)
    assert(os.path.exists(checkpoint_dir))

    #setup directory
    content = os.path.basename(args.dataset_dir)
    device_root_dir = os.path.join('/data/local/tmp', content)
    device_video_dir = os.path.join(device_root_dir, 'video')
    device_checkpoint_dir = os.path.join(device_root_dir, 'checkpoint', args.lr_video_name, model.name)
    device_libs_dir = os.path.join('/data/local/tmp', 'libs')
    device_bin_dir = os.path.join('/data/local/tmp', 'bin')
    device_cache_profile_dir = os.path.join(device_root_dir, 'profile', args.lr_video_name, model.name)
    adb_mkdir(device_video_dir, args.device_id)
    adb_mkdir(device_checkpoint_dir, args.device_id)
    adb_mkdir(device_cache_profile_dir, args.device_id)

    #setup videos
    adb_push(device_video_dir, lr_video_file, args.device_id)
    #adb_push(device_video_dir, hr_video_file, args.device_id)

    #setup a dnn
    dlc_dict = snpe_convert_model(model, model.nhwc, checkpoint_dir)
    dlc_path = os.path.join(checkpoint_dir, dlc_dict['dlc_name'])
    adb_push(device_checkpoint_dir, dlc_path)

    #setup a cache profile
    if args.aps_class == 'nemo':
        aps_class = APS_NEMO
    elif args.aps_class == 'uniform':
        aps_class = APS_Uniform
    elif args.aps_class == 'random':
        aps_class = APS_Random
    cache_profile = os.path.join(args.dataset_dir, 'profile', args.lr_video_name, model.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold))
    adb_push(device_cache_profile_dir, cache_profile, args.device_id)

    #setup scripts (setup.sh, offline_dnn.sh, online_dnn.sh)
    script_dir = '.script'
    os.makedirs(script_dir, exist_ok=True)

    #case 1: decode
    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name)
    adb_mkdir(device_script_dir, args.device_id)
    limit = '--limit={}'.format(args.limit) if args.limit is not None else ''
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --save-latency'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.lr_video_name, args.hr_video_name),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'decode_frame.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)
    os.system('adb shell "chmod +x {}"'.format(os.path.join(device_script_dir, '*.sh')))

    device_script_dir = os.path.join(device_root_dir, 'script', args.hr_video_name)
    adb_mkdir(device_script_dir, args.device_id)
    limit = '--limit={}'.format(args.limit) if args.limit is not None else ''
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --save-frame'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.hr_video_name, args.hr_video_name),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'decode_frame.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)
    os.system('adb shell "chmod +x {}"'.format(os.path.join(device_script_dir, '*.sh')))

    #case 2: online sr
    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name, model.name)
    adb_mkdir(device_script_dir, args.device_id)
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --decode-mode=1 --dnn-mode=1 --dnn-runtime=3 --dnn-name={} --checkpoint-name={} --resolution={} --save-quality --save-metadata'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name'], lr_video_profile['height']),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_sr_quality.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)

    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --decode-mode=1 --dnn-mode=1 --dnn-runtime=3 --dnn-name={} --checkpoint-name={} --resolution={} --save-latency'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name'], lr_video_profile['height']),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_sr_latency.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)
    os.system('adb shell "chmod +x {}"'.format(os.path.join(device_script_dir, '*.sh')))

    #case 3: online keyframe cache
    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name, model.name, 'cache_keyframe')
    adb_mkdir(device_script_dir, args.device_id)
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=1 --dnn-runtime=3 --cache-policy=2 --dnn-name={} --checkpoint-name={} --resolution={} --save-quality --save-metadata'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name'], lr_video_profile['height']),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_keyframe_cache_quality.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)

    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=1 --dnn-runtime=3 --cache-policy=2 --dnn-name={} --checkpoint-name={} --resolution={} --save-latency '.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name'], lr_video_profile['height']),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_keyframe_cache_latency.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)
    os.system('adb shell "chmod +x {}"'.format(os.path.join(device_script_dir, '*.sh')))

    #case 4: online profile cache
    device_script_dir = os.path.join(device_root_dir, 'script', args.lr_video_name, model.name, '{}_{}.profile'.format(aps_class.NAME1, args.threshold))
    adb_mkdir(device_script_dir, args.device_id)
    device_cache_profile_file = os.path.join(device_cache_profile_dir, '{}_{}.profile'.format(aps_class.NAME1, args.threshold))
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=1 --dnn-runtime=3 --cache-policy=1 --dnn-name={} --checkpoint-name={} --resolution={} --cache-profile={} --save-quality --save-metadata'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name'], lr_video_profile['height'],
                device_cache_profile_file),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_profile_cache_quality.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)

    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir={} --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=1 --dnn-runtime=3 --cache-policy=1 --dnn-name={} --checkpoint-name={} --resolution={} --cache-profile={} --save-latency'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, device_root_dir, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name'], lr_video_profile['height'],
                device_cache_profile_file),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_profile_cache_latency.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)
    os.system('adb shell "chmod +x {}"'.format(os.path.join(device_script_dir, '*.sh')))

