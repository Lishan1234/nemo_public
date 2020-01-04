import os
import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt

#Assumption: Fixed GOP interval
#TODO: set proper number of threads as CRA
class FDA():
    def __init__(self, vpxdec_path, content_dir, input_video, num_threads, gop):
        self.vpxdec_path = vpxdec_path
        self.content_dir = content_dir
        self.input_video = input_video
        self.num_threads = num_threads
        self.gop = gop

    def run(self, chunk_idx):
        start_idx = chunk_idx * self.gop
        end_idx = (chunk_idx + 1) * self.gop
        postfix = 'chunk{:04d}'.format(chunk_idx)

        #execute
        cmd = '{} --codec=vp9 --noblit --threads={} --frame-buffers=50 --skip={} --limit={} \
                --content-dir={} --input-video={} --postfix={} --save-metadata'.format(self.vpxdec_path, self.num_threads, \
                start_idx, end_idx - start_idx, self.content_dir, self.input_video, postfix)
        print('FDA executes {}'.format(cmd))
        os.system(cmd)

    def analyze(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        metadata_log_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'metadata.txt')

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

        #2. log a DAG data
        nodes = sorted(G.nodes, key=lambda x: float(x))
        queue1_log_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'queue1.txt')
        queue2_log_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'queue2.txt')
        with open(queue1_log_path, 'w') as f1, open(queue2_log_path, 'w') as f2:
            for node in nodes:
                print(node, G.out_degree(node), list(G.successors(node)))
                log = '{}\t{}\t{}\n'.format(G.nodes[node]['video_frame'], G.nodes[node]['super_frame'], G.out_degree(node))
                if G.out_degree(node) >= 2:
                    f1.write(log)
                else:
                    f2.write(log)

        #3. visualize a DAG
        #TODO: add options for draw_spectral() (more clear nodes, edges)
        #Reference: https://networkx.github.io/documentation/stable/tutorial.html#drawing-graphs
        options = {
                'node_color': 'black',
                'node_size': 10,
                'width': 1,
                }
        graph_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'graph.png')
        nx.draw(G, **options)
        plt.savefig(graph_path)
        plt.clf()

        s_graph_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'graph_spectral.png')
        nx.draw_spectral(G, **options)
        plt.savefig(s_graph_path)
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame Dependency Analyzer')

    #options for libvpx
    parser.add_argument('--vpxdec_path', type=str, required=True)
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--input_video_name', type=str, required=True)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--gop', type=int, required=True)

    args = parser.parse_args()

    fda = FDA(os.path.abspath(args.vpxdec_path), os.path.abspath(args.content_dir),
                args.input_video_name, args.num_threads, args.gop)
    fda.run(0)
    fda.analyze(0)
