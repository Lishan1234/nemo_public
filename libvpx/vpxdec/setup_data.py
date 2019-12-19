import argparse
import sys

from tool.adb import *
from tool.snpe import *
from dnn.model.edsr_s import EDSR_S

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)

    #model
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--nhwc', nargs='+', type=int, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    #codec
    parser.add_argument('--threads', type=str, default=1)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    #setup directory
    device_root_dir = '/data/local/tmp'
    device_video_dir = os.path.join(device_root_dir, 'video')
    device_script_dir = os.path.join(device_root_dir, 'script')
    device_libs_dir = os.path.join(device_root_dir, 'libs')
    device_bin_dir = os.path.join(device_root_dir, 'bin')
    adb_mkdir(device_video_dir, args.device_id)
    adb_mkdir(device_script_dir, args.device_id)

    #setup video
    lr_video_path = os.path.join(args.video_dir, args.lr_video_name)
    hr_video_path = os.path.join(args.video_dir, args.hr_video_name)
    adb_push(device_video_dir, lr_video_path, args.device_id)
    adb_push(device_video_dir, hr_video_path, args.device_id)

    #setup dnn
    edsr_s = EDSR_S(args.num_blocks, args.num_filters, args.scale, None)
    model = edsr_s.build_model()
    checkpoint_dir = os.path.join(args.checkpoint_dir, model.name)
    dlc_dict = convert_model(model, args.nhwc, checkpoint_dir)

    device_checkpoint_dir = os.path.join(device_root_dir, 'checkpoint', model.name)
    adb_mkdir(device_checkpoint_dir, args.device_id)
    dlc_path = os.path.join(checkpoint_dir, dlc_dict['dlc_name'])
    adb_push(device_checkpoint_dir, dlc_path)

    #setup script (setup.sh, offline_dnn.sh, online_dnn.sh)
    script_dir = 'script'
    os.makedirs(script_dir, exist_ok=True)

    #setup.sh
    limit = '--limit={}'.format(args.limit) if args.limit is not None else ''
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir=. --input-video={} --save-frame'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, args.hr_video_name),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'setup.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)

    #online_sr.sh
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir=. --input-video={} --compare-video={} --decode-mode=1 --dnn-mode=1 --dnn-name={} --dnn-file={} --save-frame --save-quality'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name']),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_sr.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)

    #online_cache.sh
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir=. --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=1 --cache-policy=2 --dnn-name={} --dnn-file={} --save-frame --save-quality'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name']),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'online_cache.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)

    #offline_cache.sh
    limit = '--limit {}'.format(args.limit) if args.limit is not None else ''
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_libs_dir),
            'export PATH=$PATH:{}'.format(device_bin_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} --content-dir=. --input-video={} --compare-video={} --decode-mode=2 --dnn-mode=2 --cache-policy=2 --dnn-name={} --dnn-file={} --save-latency'.format(os.path.join(device_bin_dir, 'vpxdec'), args.threads, limit, args.lr_video_name, args.hr_video_name, model.name, dlc_dict['dlc_name']),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'offline_cache.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path)

    os.system('adb shell "chmod +x {}"'.format(os.path.join(device_script_dir, '*.sh')))
