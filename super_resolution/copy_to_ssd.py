from option import args
from shutil import copyfile
import os

SSD_DIRECTORY="/ssd1"

#Create directory
train_data_dir = os.path.join(SSD_DIRECTORY, args.train_data, args.data_type)
valid_data_dir = os.path.join(SSD_DIRECTORY, args.valid_data, args.data_type)
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(valid_data_dir, exist_ok=True)

train_tfrecord_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}_{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch, args.scale))
valid_tfrecord_path = os.path.join(args.data_dir, args.valid_data, args.data_type, '{}_{}_valid.tfrecords'.format(args.valid_data, args.scale))

copyfile(train_tfrecord_path, os.path.join(train_data_dir, '{}_{}_{}_{}_train.tfrecords'.format(args.train_data, args.patch_size, args.num_patch, args.scale)))
copyfile(valid_tfrecord_path, os.path.join(valid_data_dir,'{}_{}_valid.tfrecords'.format(args.valid_data, args.scale)))
