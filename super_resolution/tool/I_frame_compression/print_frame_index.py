import subprocess
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='frame analyzer')
parser.add_argument('--video_path', type=str)
opt = parser.parse_args()

cmd = subprocess.Popen('ffprobe -show_frames {} -select_streams v | grep -E "pkt_size|pict_type"'.format(opt.video_path), shell=True, stdout=subprocess.PIPE)

#pkt_size_patt = re.compile('pkt_size')
pkt_size_patt = re.compile('pkt_size=\d+')
pict_type_patt = re.compile('pict_type=\S')

class Video:
    def __init__(self):
        self.frames = []

    def get_frame_info(self):
        I_frames = []
        P_frames = []
        B_frames = []

        for idx, frame in enumerate(self.frames):
            if frame.pict_type == 'I':
                I_frames.append(frame.pkt_size)
                print(idx)
            elif frame.pict_type == 'P':
                P_frames.append(frame.pkt_size)
            elif frame.pict_type == 'B':
                B_frames.append(frame.pkt_size)

        print(len(self.frames))

        return {'I': I_frames, 'P': P_frames, 'B': B_frames}

class Frame:
    def __init__(self, pkt_size, pict_type):
        self.pkt_size = pkt_size
        self.pict_type = pict_type

video = Video()
count = 0

for idx, line in enumerate(cmd.stdout):
    if idx % 2 == 0:
        m = pkt_size_patt.search(str(line))
        pkt_size = int(m.group().split('=')[-1]) / 1000 * 8 #kbps
        #print(pkt_size)
    else:
        m = pict_type_patt.search(str(line))
        pict_type = m.group().split('=')[-1]
        #print(pict_type)

        if pict_type == 'I':
            print((idx-1)/2)
            count += 1

print('total number of I-frames is {}'.format(count))
