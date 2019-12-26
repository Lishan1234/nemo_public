import os
import sys
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache optimizer')

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
