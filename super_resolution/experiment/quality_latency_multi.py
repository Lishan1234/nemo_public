import os, glob, random, sys, time, argparse
import re
import shutil
import subprocess
import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['font.size'] = 28
import numpy as np


def setup(args):
    #setup directories
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
    device_info = '{}_{}'.format(board, model)

    return device_info

def load_logs(args, quality_log_dir, latency_log_dir):
    #setup
    scales = [2, 3, 4]
    num_blocks = [1, 2, 4, 8]
    num_filters = [4, 8, 16, 32]
    metrics = ['quality', 'latency']
    _quality = [] #used for plotting a graph
    _latency = [] #used for plotting a graph
    log_dict = {}
    lq_dnn = {}
    for scale in scales:
        log_dict[scale] = {}
        log_dict[scale]['quality'] = []
        log_dict[scale]['latency'] = []
        lq_dnn[scale] = {}
        lq_dnn[scale]['num_block'] = 0
        lq_dnn[scale]['num_filter'] = 0
        lq_dnn[scale]['latency'] = 0

        count = 0
        for num_block in num_blocks:
            for num_filter in num_filters:
                model_name = 'EDSR_transpose_B{}_F{}_S{}'.format(num_block, num_filter, scale)
                quality_log_path = os.path.join(quality_log_dir, model_name, 'quality.log')
                latency_log_path = os.path.join(latency_log_dir, model_name, 'latest_results', 'benchmark_stats_{}.json'.format(model_name))
                assert os.path.isfile(quality_log_path)
                assert os.path.isfile(latency_log_path)

                with open(quality_log_path, 'r') as f:
                    quality_result = f.readlines()[-1].split('\t')
                latency_result = json.loads(open(latency_log_path).read())

                quality = float(quality_result[1])
                latency = latency_result['Execution_Data']['GPU_FP16_ub_float']['Forward Propagate']['Avg_Time'] / 1000.0

                log_dict[scale]['quality'].append(quality)
                log_dict[scale]['latency'].append(latency)
                _quality.append(quality)
                _latency.append(latency)

                if (lq_dnn[scale]['num_block'] == 0) or (latency < 30 and latency > lq_dnn[scale]['latency']):
                    lq_dnn[scale]['num_block'] = num_block
                    lq_dnn[scale]['num_filter'] = num_filter
                    lq_dnn[scale]['latency'] = latency
                    lq_dnn[scale]['index'] = count

                count += 1

        assert lq_dnn[scale]['latency'] != 0

        print('scale: {}, num_block: {}, num_filter: {}, latency: {:.2f}msec, quality gap: {:.2f}dB'.format(scale, lq_dnn[scale]['num_block'], lq_dnn[scale]['num_filter'], lq_dnn[scale]['latency'], log_dict[scale]['quality'][-1] - log_dict[scale]['quality'][lq_dnn[scale]['index']]))

    return log_dict, _quality, _latency

def plot_graph(result_dir, log_dict, _quality, _latency):
    #x_max = max(_latency)
    x_max = 240
    x_min = min(_latency)
    y_max = max(_quality)
    y_min = min(_quality)

    plt.rcParams['figure.figsize'] = (15, 10)
    fig, ax = plt.subplots()
    ax.scatter(log_dict[2]['latency'], log_dict[2]['quality'], label='x2', color='r', marker='o')
    ax.scatter(log_dict[3]['latency'], log_dict[3]['quality'], label='x3', color='g', marker='o')
    ax.scatter(log_dict[4]['latency'], log_dict[4]['quality'], label='x4', color='b', marker='o')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.set(xlabel='Average Latency (msec)', ylabel='Average PSNR (dB)', xlim=(x_min * 0.9, x_max * 1.1), ylim=(y_min * 0.9, y_max * 1.1))
    ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", ncol=3, prop={'size':24})
    fig.savefig(os.path.join(result_dir, 'quality_latency_multi.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video dataset")

    #data
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--target_resolution', type=int, default=1080)
    parser.add_argument('--video_len', type=int, default=60)
    parser.add_argument('--video_start', type=int, required=True)
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--device_id', type=str, default=None)

    args = parser.parse_args()

    device_info = setup(args)

    quality_log_dir = os.path.join(args.data_dir, args.dataset, '{}_{}_{}_{:.2f}'.format(args.target_resolution, args.video_len, args.video_start, args.fps), 'result')
    latency_log_dir = os.path.join(args.data_dir, 'runtime', 'benchmark', device_info)
    log_dict, _quality, _latency = load_logs(args, quality_log_dir, latency_log_dir)

    result_dir = os.path.join(args.data_dir, args.dataset, '{}_{}_{}_{:.2f}'.format(args.target_resolution, args.video_len, args.video_start, args.fps), 'result', device_info)
    os.makedirs(result_dir, exist_ok=True)
    plot_graph(result_dir, log_dict, _quality, _latency)
