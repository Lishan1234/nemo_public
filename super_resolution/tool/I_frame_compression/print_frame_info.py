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

        video.frames.append(Frame(pkt_size, pict_type))

result = video.get_frame_info()
I_frames, P_frames, B_frames = result['I'], result['P'], result['B']
print(len(I_frames), np.mean(I_frames), np.sum(I_frames), np.sum(I_frames)/(np.sum(I_frames)+np.sum(P_frames)+np.sum(B_frames))*100)
print(len(P_frames), np.mean(P_frames), np.sum(P_frames), np.sum(P_frames)/(np.sum(I_frames)+np.sum(P_frames)+np.sum(B_frames))*100, np.max(P_frames))
print(len(B_frames), np.mean(B_frames), np.sum(B_frames), np.sum(B_frames)/(np.sum(I_frames)+np.sum(P_frames)+np.sum(B_frames))*100)
