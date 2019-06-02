import os, glob, random, sys, time, argparse

parser = argparse.ArgumentParser(description="Video dataset")

parser.add_argument('--data_dir', type=str, default="/ssd1/data")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--device_id', type=str, default=None)

args = parser.parse_args()

#create a root dir if not existing
src_data_dir = os.path.join(args.data_dir, args.dataset, args.dataset)
dst_data_root = '/storage/emulated/0/Android/data/android.example.testlibvpx/files/mobinas'
dst_data_dir = os.path.join(dst_data_root, args.dataset)
cmd = 'adb shell "mkdir -p {}"'.format(dst_data_root)
os.system(cmd)

#remove an existing dataset in mobile
os.path.join(dst_data_dir, args.dataset)
if args.device_id is None:
    cmd = 'adb shell "rm -rf {}"'.format(dst_data_dir)
else:
    cmd = 'adb -s {} shell "rm -rf {}"'.format(args.device_id, dst_data_dir)
os.system(cmd)

#create a data dir
#cmd = 'adb shell "mkdir -p {}"'.format(dst_data_dir)
#os.system(cmd)

#copy a new dataset
if args.device_id is None:
    print(src_data_dir, dst_data_dir)
    cmd = 'adb push {} {}'.format(src_data_dir, dst_data_dir)
else:
    cmd = 'adb -s {} push {} {}'.format(args.device_id, src_data_dir, dst_data_dir)
os.system(cmd)
