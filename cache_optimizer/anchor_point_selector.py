import os
import abc
from option import args

from utility import Frame

#Anchor Point Selector
class APS():
    __metaclass__ = abc.ABCMeta
    def __init__(self, args):
        self.content_dir = args.content_dir
        self.input_video_name = args.input_video_name
        self.dnn_video_name = args.dnn_video_name
        self.compare_video_name = args.compare_video_name
        self.num_cores = args.cra_num_cores

        self.cache_profile_list = []
        self.frame_list = []
        self.anchor_point_lists = []


        self.result_dir = os.path.join(self.content_dir, "result", self.input_video_name)
        self.log_dir = os.path.join(self.result_dir, "log")
        self.profile_dir = os.path.join(self.result_dir, "profile")

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.profile_dir, exist_ok=True)

        self.base_cmd = "{} --codec=vp9 --summary --noblit --threads={} --frame-buffers=50  --content-dir={} --input-video={} --dnn-video={} --compare-video={}".format(args.vpxdec_path, args.cra_num_threads, self.content_dir, self.input_video_name, self.dnn_video_name, self.compare_video_name)
        if args.num_frames is not None:
            self.base_cmd = "{} --limit={}".format(self.base_cmd, args.num_frames)

    def select_cache_profile(self):
        pass

    @abc.abstractmethod
    def select_anchor_points(self):
        pass

    def load_power_consumption(self):
        pass

    def load_frame_index(self):
        metadata_log_path = os.path.join(self.log_dir, "metadata_thread01")
        video_index = 0
        super_index = 0

        #run a decoder to generate a log of frame index information
        if not os.path.isfile(metadata_log_path):
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
                print(video_index, super_index)

    def load_cache_erosion(self):
        pass

    def load_reference_quality(self):
        pass

    def evaluate_anchor_point(self):
        #1. Total quality gain
        #2. Self quality gain
        pass

    def evaluate_cache_profiles(self):
        #1. Average quality
        #2. Worst quality degradation (vs. per-frame SR)
        #3. Worst 90%-tile quality degration (vs. per-frame SR)
        pass

class OptimizedAPS(APS):
    def __init__(self, args):
        super(OptimizedAPS, self).__init__(args)

    #Specifically, it is bilinear-cache.
    def load_bilinear_quality_profile(self):
        pass

    def select_anchor_points(self):
        pass

    def select_anchor_point(self):
        pass

class BaselineAPS(APS):
    def __init__(self):
        pass

    def select_anchor_points(self):
        pass

if __name__ == '__main__':
    aps = OptimizedAPS(args)
    aps.load_frame_index()

"""
    def load_dnn_quality(self):
        dnn_quality = []

        quality_log_path = os.path.join(self.content_dir, "sr_{}".format(self.input_video_name), "log", "quality.log".format(self.total_frames, frame_index))
        with open(quality_log_path, "r") as f:
            quality_log = f.readlines()
            for line in quality_log:
                dnn_quality.append(line.split('\t')[0])

        return dnn_quality

    def load_cache_quality(self):
        cache_qualty = []

        for i in range(self.total_frames):
            quality = []
            quality_log_path = os.path.join(self.log_dir, "quality_g{}_i{}.log".format(self.total_frames, frame_index))
            with open(quality_log_path, "r") as f:
                quality_log = f.readlines()
                for line in quality_log:
                    quality.append(line.split('\t')[0])

            cache_qualty.append(quality)

        return cache_quality
"""
