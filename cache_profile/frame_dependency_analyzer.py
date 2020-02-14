import os
import sys
import argparse
import collections

import networkx as nx
import matplotlib.pyplot as plt

from tool.video import profile_video
from tool.libvpx import libvpx_save_metadata

#Assumption: Fixed GOP interval
#TODO: set proper number of threads as CRA
#TODO: run()
#TODO: script/{}
#TODO: show_super_frame
class FDA():
    def __init__(self, vpxdec_file, dataset_dir, video_name, gop, num_visualized_frames, show_super_frame):
        self.vpxdec_file = vpxdec_file
        self.dataset_dir = dataset_dir
        self.video_name = video_name
        self.gop = gop
        self.num_visualized_frames = num_visualized_frames
        self.show_super_frame = show_super_frame

    def node_name(self, frame_index, node_index_dict):
        if not self.show_super_frame:
            node_name = '{}'.format(node_index_dict[frame_index])
        else:
            node_name = '{}.{}'.format(frame_index[0], frame_index[1])
        return  node_name

    def run(self, chunk_idx):
        #decode
        libvpx_save_metadata(self.vpxdec_file, self.dataset_dir, self.video_name, self.gop, chunk_idx)

        #load metadata
        postfix = 'chunk{:04d}'.format(chunk_idx)
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

                if len(result) == 11:
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
                node_name = self.node_name((video_frame, super_frame), node_index_dict)

                if len(result) == 11:
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

                if self.num_visualized_frames is not None and idx + 1 >= self.num_visualized_frames:
                    break

        #visualize a DAG
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_edges(G, pos, width=4, arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=17, font_family='sans-serif')
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
        nx.draw_networkx_edges(G, pos, width=4, arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=17, font_family='sans-serif')
        graph_file = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'graph_spectral.png')
        plt.savefig(graph_file)
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame Dependency Analyzer')

    #options for libvpx
    parser.add_argument('--vpxdec_file', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--video_name', type=str, required=True)
    parser.add_argument('--gop', type=int, required=True)
    parser.add_argument('--chunk_idx', type=int, default=None)
    parser.add_argument('--num_visualized_frames', type=int, default=None)
    parser.add_argument('--show_super_frame', type=bool, default=False)

    args = parser.parse_args()

    video_file = os.path.join(args.dataset_dir, 'video', args.video_name)
    video_info = profile_video(video_file)

    fda = FDA(os.path.abspath(args.vpxdec_file), os.path.abspath(args.dataset_dir), args.video_name, args.gop, \
                args.num_visualized_frames, args.show_super_frame)
    if args.chunk_idx is None:
        num_chunks = int(video_info['duration'] // (args.gop / video_info['frame_rate']))
        for i in range(num_chunks):
            fda.run(args.chunk_idx)
    else:
        fda.run(args.chunk_idx)
