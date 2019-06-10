import os, glob, random, sys, time, argparse
import utility as util
import re
import shutil
import subprocess

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['font.size'] = 28
import numpy as np

#Info: Use video_list for evaluation

#TODO: set xlim, ylim adaptively depending on data (eval b, e, f)

parser = argparse.ArgumentParser(description="Video dataset")

parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--data_dir', type=str, default="./data")
parser.add_argument('--device_id', type=str, default=None)

#evaluation a, g.
parser.add_argument('--start_idx', type=int, default=None)
parser.add_argument('--end_idx', type=int, default=None)

args = parser.parse_args()

#0. setup
if args.device_id is None:
    cmd_board = 'adb shell getprop ro.product.board'
    cmd_model = 'adb shell getprop ro.product.model'
else:
    cmd_board = 'adb shell -s {} getprop ro.product.board'.format(args.device_id)
    cmd_model = 'adb shell -s {} getprop ro.product.model'.format(args.device_id)
proc_board = subprocess.Popen(cmd_board, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
board = proc_board.stdout.readlines()[0].decode().rstrip('\r\n').replace(' ', '')
proc_model = subprocess.Popen(cmd_model, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
model = proc_model.stdout.readlines()[0].decode().rstrip('\r\n').replace(' ', '')
prefix = '{}_{}'.format(board, model)
print(prefix)

result_dir = os.path.join(args.data_dir, args.dataset, 'result', prefix)
video_dir = os.path.join(args.data_dir, args.dataset, 'video')
video_list_path = os.path.join(video_dir, 'video_list')
os.makedirs(result_dir, exist_ok=True)

assert os.path.isfile(video_list_path)
with open(video_list_path, 'r') as f:
    video_list = f.readlines()
    postfix = video_list[3].rstrip('\r\n')

#1. download log files
src_log_dir = '/storage/emulated/0/Android/data/android.example.testlibvpx/files/mobinas/{}/log'.format(args.dataset)
dst_log_dir = os.path.join(args.data_dir, args.dataset, 'log', prefix)
os.makedirs(dst_log_dir, exist_ok=True)

if args.device_id is None:
    cmd = 'adb shell find "{}" -iname "*.log" | tr -d "\015" | while read line; do adb pull "$line" {}; done;'.format(src_log_dir, dst_log_dir)
    #cmd = 'adb shell "ls {} | xargs -n 1 adb pull"'.format(src_log_dir, dst_log_dir)
    #cmd = 'adb pull {} {}'.format(src_log_dir, dst_log_dir)
else:
    cmd = 'adb -s {} pull {} {}'.format(args.device_id, src_log_dir, dst_log_dir)
os.system(cmd)

#2. generage matplot graphs (for large-scale evaluation) & save into a 'result' folder
methods = ['bicubic', 'lq_dnn', 'hq_dnn', 'hq_dnn_cache']

#read files
log_file_dict = {}
for method in methods:
    log_file_dict[method] = {}

log_files = [f for f in os.listdir(dst_log_dir) if os.path.isfile(os.path.join(dst_log_dir, f))]

bicubic_pattern = video_list[2].rstrip('\r\n')
lq_dnn_pattern = video_list[4].rstrip('\r\n')
hq_dnn_pattern = video_list[3].rstrip('\r\n')

for log_file in log_files:
    print(log_file)
    prefix = None
    if bicubic_pattern in log_file:
        prefix = 'bicubic'
    if lq_dnn_pattern in log_file:
        prefix = 'lq_dnn'
    if hq_dnn_pattern in log_file:
        if 'cache' in log_file:
            prefix = 'hq_dnn_cache'
        else:
            prefix = 'hq_dnn'

    if prefix is not None:
        if 'quality' in log_file:
            log_file_dict[prefix]['quality'] = os.path.join(dst_log_dir, log_file)
        if 'latency' in log_file:
            log_file_dict[prefix]['latency'] = os.path.join(dst_log_dir, log_file)
        if 'metadata' in log_file:
            log_file_dict[prefix]['metadata'] = os.path.join(dst_log_dir, log_file)

#load values
metrics = ['quality', 'latency', 'super_frame', 'block', 'intra_block', 'inter_block', 'inter_block_skip']

log_dict = {}
for method in methods:
    log_dict[method] = {}
    for metric in metrics:
        log_dict[method][metric] = []

    print(log_file_dict[method]['quality'])
    with open(log_file_dict[method]['quality'], 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\t')
            log_dict[method]['quality'].append(float(line[1]))

    with open(log_file_dict[method]['latency'], 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\t')
            log_dict[method]['latency'].append(float(line[1]))

    with open(log_file_dict[method]['metadata'], 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\t')
            log_dict[method]['super_frame'].append(int(line[1]))
            log_dict[method]['block'].append(int(line[2]))
            log_dict[method]['intra_block'].append(int(line[3]))
            log_dict[method]['inter_block'].append(int(line[4]))
            log_dict[method]['inter_block_skip'].append(int(line[5]))

#validate values
length = 0
for method in methods:
    for metric in metrics:
        if length == 0:
            length = len(log_dict[method][metric])
        else:
            assert length == len(log_dict[method][metric])

if args.start_idx is None:
    start_idx = 0
else:
    start_idx = args.start_idx

if args.end_idx is None:
    end_idx = length
else:
    end_idx = args.end_idx + 1

frame_idx = np.arange(start_idx, end_idx)

#evaluation a. quality-frame index
plt.rcParams['figure.figsize'] = (20, 10)
fig, ax = plt.subplots()
ax.plot(frame_idx, log_dict['bicubic']['quality'][start_idx:end_idx], label='bicubic', color='y', marker='o')
ax.plot(frame_idx, log_dict['lq_dnn']['quality'][start_idx:end_idx], label='s-dnn', color='g', marker='o')
ax.plot(frame_idx, log_dict['hq_dnn']['quality'][start_idx:end_idx], label='h-dnn', color='b', marker='o')
ax.plot(frame_idx, log_dict['hq_dnn_cache']['quality'][start_idx:end_idx], label='h-dnn cache', color='r', marker='o')
ax.set(xlabel='Frame Index', ylabel='PSNR (dB)')
ax.legend(loc='upper center', ncol=4, prop={'size':24})
ax.grid(True)
fig.savefig(os.path.join(result_dir, 'eval01_{}.png'.format(postfix)))

#b. quality-throughput
#TODO 1: handle key frame - key frame latency is discared in cache mode
#TODO 2: consider DNN latency by loading DNN measurement results
#TODO 3: latency = max(270p decode, count(inference) * time(inference)) vs. max(270p decode + cache, count(inference) * time(inference))

#lq_dnn_latency =
#hq_dnn_latency =
#lq_dnn_latency = max(np.average(log_dict['no_cache']['latency']), )
#hq_dnn_latency = max(np.average(log_dict['no_cache']['latency']), )

plt.rcParams['figure.figsize'] = (15, 15)
fig, ax = plt.subplots()
ax.plot([np.average(log_dict['bicubic']['latency'])], [np.average(log_dict['bicubic']['quality'])], label='bicubic', color='y', marker='o', markersize=12)
ax.plot([np.average(log_dict['lq_dnn']['latency'])], [np.average(log_dict['lq_dnn']['quality'])], label='s-dnn', color='g', marker='o', markersize=12)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot([np.average(log_dict['hq_dnn']['latency'])], [np.average(log_dict['hq_dnn']['quality'])], label='h-dnn', color='b', marker='o', markersize=12)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot([np.average(log_dict['hq_dnn_cache']['latency'])], [np.average(log_dict['hq_dnn_cache']['quality'])], label='h-dnn-cache', color='r', marker='o', markersize=12)
ax.grid(True)
ax.set(xlabel='Average Latency (msec)', ylabel='PSNR (dB)', xlim=(0,120))
ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=2, prop={'size':24})
fig.savefig(os.path.join(result_dir, 'eval02_{}.png'.format(postfix)))

#c. decode latency CDF (cache)
fig, ax = plt.subplots()
y = sorted(log_dict['hq_dnn_cache']['latency'])
data_size = len(log_dict['hq_dnn_cache']['latency'])
data_set= sorted(set(log_dict['hq_dnn_cache']['latency']))
bins = np.append(data_set, data_set[-1]+1)
counts, bin_edges = np.histogram(log_dict['hq_dnn_cache']['latency'], bins=bins, density=False)
counts = counts.astype(float)/data_size
cdf = np.cumsum(counts)
ax.plot(bin_edges[0:-1], cdf,linestyle='--', marker="o", color='r', label='h-dnn cache')
ax.grid(True)
ax.set(xlabel='Latency (msec)', ylabel='CDF', ylim=(0,1))
ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=1, prop={'size':24})
fig.savefig(os.path.join(result_dir, 'eval03_{}.png'.format(postfix)))

#d. interblock CDF (cache)
fig, ax = plt.subplots()
y = sorted(log_dict['hq_dnn_cache']['inter_block_skip'])
data_size = len(log_dict['hq_dnn_cache']['inter_block_skip'])
data_set= sorted(set(log_dict['hq_dnn_cache']['inter_block_skip']))
bins = np.append(data_set, data_set[-1]+1)
counts, bin_edges = np.histogram(log_dict['hq_dnn_cache']['inter_block_skip'], bins=bins, density=False)
counts = counts.astype(float)/data_size
cdf = np.cumsum(counts)
ax.grid(True)
ax.plot(bin_edges[0:-1], cdf,linestyle='--', marker="o", color='r', label='h-dnn cache')
ax.set(xlabel='Number of Non-skipped Inter-blocks', ylabel='CDF', ylim=(0,1))
ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=1, prop={'size':24})
fig.savefig(os.path.join(result_dir, 'eval04_{}.png'.format(postfix)))

#e. decode latency-interblock (cache)
fig, ax = plt.subplots()
x = log_dict['hq_dnn_cache']['inter_block_skip']
y = log_dict['hq_dnn_cache']['latency']
ax.scatter(x, y, marker="o", color='r', label='h-dnn cache')
ax.set(xlabel='Number of Non-skipped Inter-blocks', ylabel='Latency (msec)', xlim=(0,200), ylim=(0,100))
ax.grid(True)
ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=1, prop={'size':24})
fig.savefig(os.path.join(result_dir, 'eval05_{}.png'.format(postfix)))

#f. decode latency-interblock (%) (cache)
fig, ax = plt.subplots()
x = np.divide(log_dict['hq_dnn_cache']['inter_block_skip'], log_dict['hq_dnn_cache']['block']) * 100
y = log_dict['hq_dnn_cache']['latency']
ax.scatter(x, y, marker="o", color='r', label='h-dnn cache')
ax.set(xlabel='Non-skipped Inter-blocks (%)', ylabel='Latency (msec)', xlim=(0,50), ylim=(0, 100))
ax.grid(True)
ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=1, prop={'size':24})
fig.savefig(os.path.join(result_dir, 'eval06_{}.png'.format(postfix)))

#g. intra-inter block portion
plt.rcParams['figure.figsize'] = (20, 10)
fig, ax = plt.subplots()
ax.plot(frame_idx, log_dict['hq_dnn_cache']['inter_block'][start_idx:end_idx], label='Inter-predicted block', color='r', marker='o')
ax.plot(frame_idx, log_dict['hq_dnn_cache']['intra_block'][start_idx:end_idx], label='Intra-predicted block', color='b', marker='o')
ax.set(xlabel='Frame Index', ylabel='Count')
ax.legend(loc='upper center', ncol=2, prop={'size':24})
ax.grid(True)
fig.savefig(os.path.join(result_dir, 'eval07_{}.png'.format(postfix)))
