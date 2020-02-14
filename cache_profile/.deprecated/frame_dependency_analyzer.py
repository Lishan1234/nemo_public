import os
import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt

#Assumption: Fixed GOP interval
#TODO: set proper number of threads as CRA
#TODO: run()
#TODO: script/{}
class FDA():
    def __init__(self, vpxdec_file, dataset_dir, video_name, num_threads, gop, num_visualized_frames):
        self.vpxdec_file = vpxdec_file
        self.dataset_dir = dataset_dir
        self.video_name = video_name
        self.num_threads = num_threads
        self.gop = gop
        self.num_visualized_frames = num_visualized_frames

    def run(self.chunk_idx):
        libvpx_save_metadata(self.vpxdec_file, self.dataset_dir, self.video_name, self.gop, chunk_idx)

    def decode(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        #execute
        cmd = '{} --codec=vp9 --noblit --threads={} --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-metadata'.format(self.vpxdec_file, self.num_threads, \
                start_idx, end_idx - start_idx, self.dataset_dir, self.video_name, postfix)
        print('FDA executes {}'.format(cmd))
        os.system(cmd)

    def log_graph(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        metadata_log_path = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'metadata.txt')

        G = nx.DiGraph()
        with open(metadata_log_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                result = line.strip().split('\t')
                video_frame = int(result[0])
                super_frame = int(result[1])
                node_name = '{}.{}'.format(video_frame, super_frame)

                if len(result) == 11:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)

                    #add edge
                    for i in range(3):
                        ref_video_frame = int(result[2*i+5])
                        ref_super_frame = int(result[2*i+6])
                        ref_node_name = '{}.{}'.format(ref_video_frame, ref_super_frame)
                        G.add_edge(ref_node_name, node_name)
                else:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)

        postfix = 'chunk{:04d}'.format(chunk_idx)
        nodes = sorted(G.nodes, key=lambda x: float(x))
        queue1_log_path = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'queue1.txt')
        queue2_log_path = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'queue2.txt')

        with open(queue1_log_path, 'w') as f1, open(queue2_log_path, 'w') as f2:
            for node in nodes:
                print(node, G.out_degree(node), list(G.successors(node)))
                log = '{}\t{}\t{}\n'.format(G.nodes[node]['video_frame'], G.nodes[node]['super_frame'], G.out_degree(node))
                if G.out_degree(node) >= 2:
                    f1.write(log)
                else:
                    f2.write(log)

    #TODO: add options for draw_spectral() (more clear nodes, edges)
    #Reference: https://networkx.github.io/documentation/stable/tutorial.html#drawing-graphs
    def visualize_graph(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        metadata_log_path = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'metadata.txt')

        G = nx.DiGraph()
        with open(metadata_log_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                result = line.strip().split('\t')
                video_frame = int(result[0])
                super_frame = int(result[1])
                node_name = '{}.{}'.format(video_frame, super_frame)

                if len(result) == 11:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)

                    #add edge
                    for i in range(3):
                        ref_video_frame = int(result[2*i+5])
                        ref_super_frame = int(result[2*i+6])
                        ref_node_name = '{}.{}'.format(ref_video_frame, ref_super_frame)
                        G.add_edge(ref_node_name, node_name)
                else:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)

                if self.num_visualized_frames is not None and idx + 1 >= self.num_visualized_frames:
                    break

        options = {
                'node_color': 'black',
                'node_size': 10,
                'width': 1,
                }
        graph_path = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'graph.png')
        nx.draw(G, **options)
        plt.savefig(graph_path)
        plt.clf()

        s_graph_path = os.path.join(self.dataset_dir, 'log', self.video_name, postfix, 'graph_spectral.png')
        nx.draw_spectral(G, **options)
        plt.savefig(s_graph_path)
        plt.clf()

    def run(self, chunk_idx=None):
        if chunk_idx is None:
            input_video_path = os.path.join(self.dataset_dir, 'video', self.video_name)
            input_video_info = profile_video(input_video_path)
            num_chunks = int(input_video_info['duration'] // (self.gop / input_video_info['frame_rate']))
            for i in range(num_chunks):
                self.decode(i)
                self.log_graph(i)
                self.visualize_graph(i)
        else:
            self.decode(chunk_idx)
            self.log_graph(chunk_idx)
            self.visualize_graph(chunk_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame Dependency Analyzer')

    #options for libvpx
    parser.add_argument('--vpxdec_file', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--video_name', type=str, required=True)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--gop', type=int, required=True)
    parser.add_argument('--chunk_idx', type=int, default=None)
    parser.add_argument('--num_visualized_frames', type=int, default=None)

    args = parser.parse_args()

    fda = FDA(os.path.abspath(args.vpxdec_file), os.path.abspath(args.dataset_dir),
                args.video_name, args.num_threads, args.gop, args.num_visualized_frames)
    fda.run(args.chunk_idx)
