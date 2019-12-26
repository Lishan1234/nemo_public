import os
import sys
import argparse
import networkx as nx

#Assumption: Fixed GOP interval
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
        log_path = os.path.join(self.content_dir, 'log', self.input_video, postfix, 'metadata.txt')

        G = nx.DiGraph()
        with open(log_path, 'r') as f:
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
                        G.add_edge(node_name, ref_node_name)
                else:
                    #add node
                    G.add_node(node_name, video_frame=video_frame, super_frame=super_frame)

        #2. log a DAG data
        #iterate over node
        #if node has out_degree >= 2, log into queue1
        #if node has out_degree == 1, log into queue2

        #3. visualize a DAG

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
    #fda.run(0)
    fda.analyze(0)
