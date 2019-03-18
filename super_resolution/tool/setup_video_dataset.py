import os, glob, random, sys, time, argparse

parser = argparse.ArgumentParser(description="Video dataset")

parser.add_argument('--data_dir', type=str, default="../data")
parser.add_argument('--dataset', type=str, default="news")
parser.add_argument('--video_len', type=int, default=60)
parser.add_argument('--lr', type=int, default=240)
parser.add_argument('--raw', type=int, default=1080)
parser.add_argument('--video_format', type=str, default="mp4")
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--fps', type=float, default=0.5)

args = parser.parse_args()

RESOLUTION={240: (240, 426), 360: (360, 480), 480: (480, 858), 720: (720, 1280), 1080: (1080, 1920)}
hr = args.lr * args.scale
hr_h = RESOLUTION[args.lr][0] * args.scale
hr_w = RESOLUTION[args.lr][1] * args.scale

#Create directoriy
data_root_dir = os.path.join(args.data_dir, args.dataset, "{}_{}".format(args.video_len, args.fps))
os.makedirs(data_root_dir, exist_ok=True)

hr_root_dir = os.path.join(data_root_dir, "{}p".format(hr))
os.makedirs(hr_root_dir, exist_ok=True)
hr_raw_dir = os.path.join(hr_root_dir, "original")
os.makedirs(hr_raw_dir, exist_ok=True)

lr_root_dir = os.path.join(data_root_dir, "{}p".format(args.lr))
os.makedirs(lr_root_dir, exist_ok=True)
lr_raw_dir = os.path.join(lr_root_dir, "original")
os.makedirs(lr_raw_dir, exist_ok=True)
lr_bicubic_dir = os.path.join(lr_root_dir, "bicubic_{}p".format(hr))
os.makedirs(lr_bicubic_dir, exist_ok=True)

#Check/Encode videos
video_raw_full_source = os.path.join(args.data_dir, args.dataset, "video", "original_{}p.{}".format(args.raw, args.video_format))
video_raw_source = os.path.join(args.data_dir, args.dataset, "video", "original_{}p_{}.{}".format(args.raw, args.video_len, args.video_format))
video_lr_full_source = os.path.join(args.data_dir, args.dataset, "video", "original_{}p.{}".format(args.lr, args.video_format))
video_lr_source = os.path.join(args.data_dir, args.dataset, "video", "original_{}p_{}.{}".format(args.lr, args.video_len, args.video_format))

hour = "{:02d}".format(int(args.video_len / 3600))
remainder = args.video_len - int(args.video_len / 3600) * 3600
min = "{:02d}".format(int(remainder / 60))
sec = "{:02d}".format(args.video_len % 60)
if not os.path.isfile(video_raw_source): #cut inital part of a video
    cmd = "ffmpeg -i {} -ss 00:00:00 -t {}:{}:{} {}".format(video_raw_full_source, hour, min, sec, video_raw_source)
    os.system(cmd)
if not os.path.isfile(video_lr_source): #cut inital part of a video
    cmd = "ffmpeg -i {} -ss 00:00:00 -t {}:{}:{} {}".format(video_lr_full_source, hour, min, sec, video_lr_source)
    os.system(cmd)

#Decode
cmd = "ffmpeg -i {} -r {} -vf scale={}:{} {}/%04d.png".format(video_raw_source, args.fps, hr_w, hr_h, hr_raw_dir)
os.system(cmd)
cmd = "ffmpeg -i {} -r {} {}/%04d.png".format(video_lr_source, args.fps, lr_raw_dir)
os.system(cmd)
cmd = "ffmpeg -i {}/%04d.png -vf scale={}:{} {}/%04d.png".format(lr_raw_dir, hr_w, hr_h, lr_bicubic_dir)
os.system(cmd)

sys.exit()
