import os
import sys
import argparse
import collections

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tool.video import profile_video
from tool.libvpx import libvpx_save_metadata
from cache_profile.anchor_point_selector_uniform import APS_Uniform
from cache_profile.anchor_point_selector_random import APS_Random
from cache_profile.anchor_point_selector_nemo import APS_NEMO
from cache_profile.anchor_point_selector_uniform_eval import APS_Uniform_Eval
from cache_profile.anchor_point_selector_random_eval import APS_Random_Eval
from dnn.model.nemo_s import NEMO_S

class FDA():
    def __init__(self, vpxdec_file, dataset_dir, video_name, model_name, cache_profile_name, gop, num_visualized_frames, show_super_frame):
        self.vpxdec_file = vpxdec_file
        self.dataset_dir = dataset_dir
        self.video_name = video_name
        self.model_name = model_name
        self.cache_profile_name = cache_profile_name
        self.gop = gop
        self.num_visualized_frames = num_visualized_frames
        self.show_super_frame = show_super_frame

    def node_name(self, frame_index, node_index_dict):
        if not self.show_super_frame:
            node_name = '{}'.format(node_index_dict[frame_index])
        else:
            node_name = '{}.{}'.format(frame_index[0], frame_index[1])
        return  node_name

    #Caution: Need to run NEMO first to prepare 'metdata.txt' by 'online_cache_latency.py'
    def all(self):
        metadata_log_file = os.path.join(self.dataset_dir, 'log', self.video_name, self.model_name, self.cache_profile_name, 'metadata.txt')
        assert(os.path.exists(metadata_log_file))

        #node index
        node_index_dict = {}
        count = 0
        with open(metadata_log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                video_frame = int(line[0])
                super_frame = int(line[1])
                node_index_dict[(video_frame, super_frame)] = count
                count += 1

        #build a DAG
        G = nx.DiGraph()
        with open(metadata_log_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                result = line.strip().split('\t')
                video_frame = int(result[0])
                super_frame = int(result[1])
                frame_type = result[-1]
                is_anchor_point = int(result[2])
                node_name = self.node_name((video_frame, super_frame), node_index_dict)

                if len(result) == 12:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame, frame_type=frame_type, is_anchor_point=is_anchor_point)

                    #add edge
                    for i in range(3):
                        ref_video_frame = int(result[2*i+5])
                        ref_super_frame = int(result[2*i+6])
                        ref_node_name = self.node_name((ref_video_frame, ref_super_frame), node_index_dict)
                        G.add_edge(ref_node_name, node_name)
                else:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame, frame_type=frame_type, is_anchor_point=is_anchor_point)

        #log0: out-degree per anchor point (CDF)
        out_degree = []
        nodes = sorted(G.nodes, key=lambda x: float(x))
        for node in nodes:
            if G.nodes[node]['is_anchor_point'] == 1:
                out_degree.append(G.out_degree(node))
        out_degree.sort()
        count = {x:out_degree.count(x) for x in out_degree}
        value, count = list(count.keys()), list(count.values())
        cdf = []
        for i in range(len(count)):
            cdf.append(sum(count[0:i+1]) / sum(count))

        cdf_log_file = os.path.join(self.dataset_dir, 'log', self.video_name, self.model_name, self.cache_profile_name, 'out_degree_cdf.txt')
        is_first = True
        with open(cdf_log_file, 'w') as f:
            for x_val, y_val in zip(value, cdf):
                if is_first:
                    if x_val != 0:
                        f.write('{}\t{}\n'.format(0, 0))
                        f.write('{}\t{}\n'.format(x_val, 0))
                        f.write('{}\t{}\n'.format(x_val, y_val))
                    else:
                        f.write('{}\t{}\n'.format(x_val, 0))
                        f.write('{}\t{}\n'.format(x_val, y_val))
                    is_first = False
                else:
                    f.write('{}\t{}\n'.format(x_val, y_pre_val))
                    f.write('{}\t{}\n'.format(x_val, y_val))
                y_pre_val = y_val

        #log1: frame type per anchor point (CDF)
        key_frame = []
        alternative_reference_frame = []
        golden_frame = []
        normal_frame = []
        nodes = sorted(G.nodes, key=lambda x: float(x))
        for node in nodes:
            #print(node, G.out_degree(node), list(G.successors(node)))
            if G.nodes[node]['is_anchor_point'] == 1:
                if G.nodes[node]['frame_type'] == 'key_frame':
                    key_frame.append(G.out_degree(node))
                elif G.nodes[node]['frame_type'] == 'alternative_reference_frame':
                    alternative_reference_frame.append(G.out_degree(node))
                elif G.nodes[node]['frame_type'] == 'normal_frame':
                    if G.out_degree(node) > 1:
                        golden_frame.append(G.out_degree(node))
                    else:
                        normal_frame.append(G.out_degree(node))
                else:
                    print(G.nodes[node]['frame_type'])
                    raise NotImplementedError

        total_num = len(key_frame) + len(alternative_reference_frame) + len(golden_frame) + len(normal_frame)
        frame_type_log = os.path.join(self.dataset_dir, 'log', self.video_name, self.model_name, self.cache_profile_name, 'frame_type.txt')
        with open(frame_type_log, 'w') as f:
            f.write('Key\t{:.2f}\n'.format(len(key_frame)/total_num))
            f.write('Alf_ref\t{:.2f}\n'.format(len(alternative_reference_frame)/total_num))
            f.write('Golden\t{:.2f}\n'.format(len(golden_frame)/total_num))
            f.write('Others\t{:.2f}\n'.format(len(normal_frame)/total_num))

        #log2: (portion, degree of dependency) for each type
        key_frame = []
        alternative_reference_frame = []
        golden_frame = []
        normal_frame = []
        nodes = sorted(G.nodes, key=lambda x: float(x))
        for node in nodes:
            #print(node, G.out_degree(node), list(G.successors(node)))
            if G.nodes[node]['frame_type'] == 'key_frame':
                key_frame.append(G.out_degree(node))
            elif G.nodes[node]['frame_type'] == 'alternative_reference_frame':
                alternative_reference_frame.append(G.out_degree(node))
            elif G.nodes[node]['frame_type'] == 'normal_frame':
                if G.out_degree(node) > 1:
                    golden_frame.append(G.out_degree(node))
                else:
                    normal_frame.append(G.out_degree(node))
            else:
                print(G.nodes[node]['frame_type'])
                raise NotImplementedError

        frame_type_log_file = os.path.join(self.dataset_dir, 'log', self.video_name, 'frame_type.txt')
        with open(frame_type_log_file, 'w') as f:
            f.write("Key frame\t{:.2f}\t{:.2f}\n".format(len(key_frame)/len(nodes) * 100, np.average(key_frame)))
            f.write("Alternative reference frame\t{:.2f}\t{:.2f}\n".format(len(alternative_reference_frame)/len(nodes) * 100, np.average(alternative_reference_frame)))
            f.write("Golden frame\t{:.2f}\t{:.2f}\n".format(len(golden_frame)/len(nodes) * 100, np.average(golden_frame)))
            f.write("Normal frame\t{:.2f}\t{:.2f}\n".format(len(normal_frame)/len(nodes) * 100, np.average(normal_frame)))

    #deprecated
    def run(self, chunk_idx):
        #decode
        start_idx = self.gop * chunk_idx
        end_idx = self.gop * (chunk_idx + 1)
        postfix = 'chunk{:04d}'.format(chunk_idx)
        libvpx_save_metadata(self.vpxdec_file, self.dataset_dir, self.video_name, start_idx, end_idx, postfix)
        metadata_log_file = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'metadata.txt')

        #node index
        node_index_dict = {}
        count = 0
        with open(metadata_log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                video_frame = int(line[0])
                super_frame = int(line[1])
                node_index_dict[(video_frame, super_frame)] = count
                count += 1

        #build a DAG
        G = nx.DiGraph()
        with open(metadata_log_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                result = line.strip().split('\t')
                video_frame = int(result[0])
                super_frame = int(result[1])
                node_name = self.node_name((video_frame, super_frame), node_index_dict)

                if len(result) == 12:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)

                    #add edge
                    for i in range(3):
                        ref_video_frame = int(result[2*i+5])
                        ref_super_frame = int(result[2*i+6])
                        ref_node_name = self.node_name((ref_video_frame, ref_super_frame), node_index_dict)
                        G.add_edge(ref_node_name, node_name)
                else:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)

        #log0: out-degree per frame index
        if self.show_super_frame:
            nodes = sorted(G.nodes, key=lambda x: float(x))
        else:
            nodes = sorted(G.nodes, key=lambda x: int(x))
        nodes = G.nodes
        log_file = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'out_degree_per_frame.txt')
        with open(log_file, 'w') as f:
            for node in nodes:
                #print(node, G.out_degree(node), list(G.successors(node)))
                log = '{}\t{}\t{}\n'.format(G.nodes[node]['video_frame'], G.nodes[node]['super_frame'], G.out_degree(node))
                f.write(log)

        #log1: out-degree histogram
        degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        log_file = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'out_degree_histogram.txt')
        with open(log_file, 'w') as f:
            for deg, cnt in zip(deg, cnt):
                log = '{}\t{}\n'.format(deg, cnt)
                f.write(log)

        #build a subgraph
        del(G)
        G = nx.DiGraph()
        with open(metadata_log_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                result = line.strip().split('\t')
                video_frame = int(result[0])
                super_frame = int(result[1])
                frame_type = result[-1]
                node_name = self.node_name((video_frame, super_frame), node_index_dict)

                if len(result) == 12:
                    #add node
                    #G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame, frame_type=frame_type)

                    #add edge
                    for i in range(3):
                        ref_video_frame = int(result[2*i+5])
                        ref_super_frame = int(result[2*i+6])
                        ref_node_name = self.node_name((ref_video_frame, ref_super_frame), node_index_dict)
                        G.add_edge(ref_node_name, node_name)
                else:
                    #add node
                    #G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame, frame_type=frame_type)

                if self.num_visualized_frames is not None and idx + 1 >= self.num_visualized_frames:
                    break

        color_map = []
        size_map = []
        for node in G:
            if G.nodes[node]['frame_type'] == 'key_frame':
                color_map.append('blue')
                size_map.append(2800)
            elif G.nodes[node]['frame_type'] == 'alternative_reference_frame':
                color_map.append('green')
                size_map.append(1400)
            elif G.nodes[node]['frame_type'] == 'normal_frame':
                if G.out_degree(node) > 1:
                    color_map.append('yellow')
                    size_map.append(1400)
                else:
                    color_map.append('grey')
                    size_map.append(700)

        #visualize a DAG
        pos = nx.spring_layout(G,k=1/3.5)
        #nx.draw_networkx_nodes(G, pos, node_size=200)
        #nx.draw_networkx_edges(G, pos, width=5, arrowsize=5)
        #nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=size_map)
        #nx.draw_networkx_edges(G, pos, width=4, arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=20, font_color='black', font_family='sans-serif')
        #nx.draw_networkx_nodes(G, pos, node_size=100)
        nx.draw_networkx_edges(G, pos, width=3, arrowsize=20)
        graph_file = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'graph_spring.png')
        plt.savefig(graph_file)
        plt.clf()

        pos = nx.spectral_layout(G)
        """
        #hardcode for paper figure
        pos['0'] = [-0.3, 0.6]
        pos['1'] = [0.7,  0.6]
        pos['2'] = [-1.2, 0.0]
        pos['3'] = [-0.8, 0.0]
        pos['4'] = [-0.4, 0.0]
        pos['5'] = [0.0, 0.0]
        pos['6'] = [0.4, 0.0]
        pos['7'] = [0.8, 0.0]
        pos['8'] = [1.2, 0.0]
        pos['9'] = [1.6, 0.0]
        """
        nx.draw_networkx_nodes(G, pos, node_size=700)
        #nx.draw_networkx_edges(G, pos, width=4, arrowsize=15)
        #nx.draw_networkx_labels(G, pos, font_size=17, font_family='sans-serif')
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=100)
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=10)
        #nx.draw_networkx_labels(G, pos, font_size=17, font_family='sans-serif')
        graph_file = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'graph_spectral.png')
        plt.savefig(graph_file)
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame Dependency Analyzer')

    #options for libvpx
    parser.add_argument('--vpxdec_file', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--gop', type=int, required=True)
    parser.add_argument('--chunk_idx', type=int, default=None)
    parser.add_argument('--num_visualized_frames', type=int, default=None)
    parser.add_argument('--show_super_frame', type=bool, default=False)

    #dnn
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    #anchor point selector
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--mode', required=True)

    args = parser.parse_args()

    #scale, nhwc
    lr_video_file = os.path.join(args.dataset_dir, 'video', args.lr_video_name)
    hr_video_file = os.path.join(args.dataset_dir, 'video', args.hr_video_name)
    lr_video_info = profile_video(lr_video_file)
    hr_video_info = profile_video(hr_video_file)
    scale = int(hr_video_info['height'] / lr_video_info['height'])
    nhwc = [1, lr_video_info['height'], lr_video_info['width'], 3]

    #model
    nemo_s = NEMO_S(args.num_blocks, args.num_filters, scale)

    #cache profile
    aps = None
    if args.mode == 'uniform':
        aps = APS_Uniform
    elif args.mode == 'random':
        aps = APS_Random
    elif args.mode == 'nemo':
        aps = APS_NEMO
    elif args.mode == 'uniform_eval':
        aps = APS_Uniform_Eval
    elif args.mode == 'random_eval':
        aps = APS_Random_Eval
    else:
        raise NotImplementedError

    fda = FDA(os.path.abspath(args.vpxdec_file), os.path.abspath(args.dataset_dir), args.lr_video_name, nemo_s.name, '{}_{}.profile'.format(aps.NAME1, args.threshold),
            args.gop, args.num_visualized_frames, args.show_super_frame)

    if args.chunk_idx is None:
        fda.all()
    else:
        fda.run(args.chunk_idx)
