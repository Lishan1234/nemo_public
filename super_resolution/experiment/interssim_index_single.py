import os, glob, random, sys, time, argparse
import re
import shutil
import subprocess
import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['font.size'] = 32
import numpy as np


def load_logs(args):
    log_dict = {}

    log_dict['starcraft'] = []
    log_path = os.path.join(args.data_dir, 'starcraft', 'log', 'inter_ssim_{:.2f}_{}p_lossless_{}sec_{}st.log'.format(args.fps, args.target_resolution, 60, 120))
    assert os.path.isfile(log_path)
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            result = line.split('\t')
            if len(result) == 4:
                log_dict['starcraft'].append(float(result[3]))

    log_dict['movie'] = []
    log_path = os.path.join(args.data_dir, 'movie', 'log', 'inter_ssim_{:.2f}_{}p_lossless_{}sec_{}st.log'.format(args.fps, args.target_resolution, 60, 125))
    assert os.path.isfile(log_path)
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = line.split('\t')
            if len(result) == 4:
                log_dict['movie'].append(float(result[3]))

    log_dict['basketball'] = []
    log_path = os.path.join(args.data_dir, 'basketball', 'log', 'inter_ssim_{:.2f}_{}p_lossless_{}sec_{}st.log'.format(args.fps, args.target_resolution, 60, 1090))
    assert os.path.isfile(log_path)
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = line.split('\t')
            if len(result) == 4:
                log_dict['basketball'].append(float(result[3]))

    return log_dict

def plot_graph(result_dir, log_dict):

    plt.rcParams['figure.figsize'] = (15, 10)
    fig, ax = plt.subplots()
    index = list(range(0, len(log_dict['starcraft'])))
    ax.plot(index, log_dict['starcraft'], label='starcraft', color='r', marker='o')
    index = list(range(0, len(log_dict['movie'])))
    ax.plot(index, log_dict['movie'], label='movie', color='g', marker='o')
    index = list(range(0, len(log_dict['basketball'])))
    ax.plot(index, log_dict['basketball'], label='basketball', color='b', marker='o')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.set(xlabel='Index', ylabel='Average SSIM', ylim=(0,1))
    ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", ncol=3, prop={'size':24})
    fig.savefig(os.path.join(result_dir, 'inter_ssim.png'))

    print('starcraft: {:.2f}'.format(np.average(log_dict['starcraft'])))
    print('movie: {:.2f}'.format(np.average(log_dict['movie'])))
    print('basketball: {:.2f}'.format(np.average(log_dict['basketball'])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video dataset")

    #data
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--target_resolution', type=int, default=1080)
    parser.add_argument('--fps', type=float, required=True)

    args = parser.parse_args()

    log_dict = load_logs(args)

    result_dir = os.path.join(args.data_dir)
    os.makedirs(result_dir, exist_ok=True)
    plot_graph(result_dir, log_dict)
