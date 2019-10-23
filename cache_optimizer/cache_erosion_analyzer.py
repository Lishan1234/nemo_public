import os
import sys
import struct

from option import args

class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

#Cache Erosion Analyzer (CRA)
class CRA():
    def __init__(self, args):
        self.frame_list = None
        self.content_dir = args.content_dir
        self.input_video_name = args.input_video_name
        self.dnn_video_name = args.dnn_video_name
        self.compare_video_name = args.compare_video_name
        self.num_cores = args.cra_num_cores

        self.result_dir = os.path.join(self.content_dir, "result", self.input_video_name)
        self.log_dir = os.path.join(self.result_dir, "log")
        self.profile_dir = os.path.join(self.result_dir, "profile")

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.profile_dir, exist_ok=True)

        self.base_cmd = "{} --codec=vp9 --summary --noblit --threads={} --frame-buffers=50  --content-dir={} --input-video={} --dnn-video={} --compare-video={}".format(args.vpxdec_path, args.cra_num_threads, self.content_dir, self.input_video_name, self.dnn_video_name, self.compare_video_name)
        if args.num_frames is not None:
            self.base_cmd = "{} --limit={}".format(self.base_cmd, args.num_frames)

    @classmethod
    def get_profile_name(cls, video_index, super_index):
        return "cra_v{}_s{}".format(video_index, super_index)

    def prepare_frame_index(self):
        metadata_log_path = os.path.join(self.log_dir, "metadata_thread01")
        video_index = 0
        super_index = 0

        #run a decoder to generate a log of frame index information
        cmd = "{} --decode-mode=0 --save-metadata".format(self.base_cmd)
        os.system(cmd)

        #load the log and read frame index information
        self.frame_list = []
        with open(metadata_log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                metadata = line.split('\t')
                video_index = int(metadata[0])
                super_index = int(metadata[1])
                frame = Frame(video_index, super_index)
                self.frame_list.append(frame)

    #generate cache profiles for CRA: 1 of |GOP| frames is selceted as an anchor point
    def save_cache_profile(self, video_index, super_index):
        cache_profile_path = os.path.join(self.profile_dir, CRA.get_profile_name(video_index, super_index))
        with open(cache_profile_path, "wb") as f:
            byte_value = 0
            for i, frame in enumerate(self.frame_list):
                if frame.video_index == video_index and frame.super_index == super_index:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frame_list) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def prepare_cache_profiles(self):
        for frame in self.frame_list:
            self.save_cache_profile(frame.video_index, frame.super_index)

    #TODO: use Process and Queue for parallel decoding on multi-cores
    #run the cache profiles & measure the quality
    def run_cache_profiles(self):
        for frame in self.frame_list:
            cache_profile_name = CRA.get_profile_name(frame.video_index, frame.super_index)
            cache_profile_path = os.path.join(self.profile_dir, cache_profile_name)
            cmd = "{} --decode-mode=2 --dnn-mode=2 --cache-policy=1 --cache-profile={} --save-quality --save-metadata".format(self.base_cmd, cache_profile_path)
            os.system(cmd)

if __name__ == '__main__':
    cra = CRA(args)
    cra.prepare_frame_index()
    cra.prepare_cache_profiles()
    cra.run_cache_profiles()