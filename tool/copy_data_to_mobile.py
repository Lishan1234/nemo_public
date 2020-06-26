import os
import argparse

import nemo.dnn.model
from nemo.tool.snpe import snpe_convert_model
from nemo.tool.video import profile_video
from nemo.tool.adb import adb_mkdir, adb_push

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #directory, path
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)

    #training
    parser.add_argument('--train_type', type=str, required=True)

    #dnn
    parser.add_argument('--model_type', type=str, default='nemo_s')
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')
    parser.add_argument('--scale', type=int, default=None)

    #device
    parser.add_argument('--device_id', type=str, required=True)
    parser.add_argument('--device_rootdir', type=str, default='/storage/emulated/0/Android/data/android.example.testlibvpx/files')

    args = parser.parse_args()

    #setup videos
    lr_video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    lr_video_profile = profile_video(lr_video_path)
    hr_video_path = os.path.join(args.data_dir, args.content, 'video', args.hr_video_name)
    hr_video_profile = profile_video(hr_video_path)
    scale = args.output_height // lr_video_profile['height'] #NEMO upscales a LR image to a 1080p version
    input_shape = [1, lr_video_profile['height'], lr_video_profile['width'], 3]

    #setup a model
    model = nemo.dnn.model.build(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type, apply_clip=True)
    if args.train_type == 'train_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
        log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name, model.name)
    elif args.train_type == 'finetune_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, '{}_finetune'.format(model.name))
        log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name, '{}_finetune'.format(model.name))
    else:
        raise ValueError('Unsupported training types')
    dlc_dict = snpe_convert_model(model, input_shape, checkpoint_dir)

    #TODO
    #setup a cache profile

    #create directory at a device
    device_video_dir = os.path.join(args.device_rootdir, args.content, 'video')
    device_checkpoint_dir = os.path.join(args.device_rootdir, args.content, 'checkpoint', args.lr_video_name)
    adb_mkdir(device_video_dir, args.device_id)
    adb_mkdir(device_checkpoint_dir, args.device_id)

    #copy dnn, video to a device
    adb_push(device_video_dir, lr_video_path, args.device_id)
    adb_push(device_video_dir, hr_video_path, args.device_id)
    dlc_path = os.path.join(checkpoint_dir, dlc_dict['dlc_name'])
    adb_push(device_checkpoint_dir, dlc_path)

    #copy a cache profile to a device
