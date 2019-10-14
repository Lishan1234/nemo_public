import os
import sys
import struct

from option import args

class FrameIndex():
    def __init__(self, current_frame_index, super_frame_index):
        self.current_frame_index = current_frame_index;
        self.super_frame_index= super_frame_index;

def get_profile_name(current_frame_index, super_frame_index):
    return "c{}_s{}.profile".format(current_frame_index, super_frame_index)

class CacheErosionAnalyzer():
    def __init__(self):
        self.frame_index_list = None
        self.content_dir = args.content_dir
        self.input_video_name = args.input_video_name
        self.dnn_video_name = args.dnn_video_name
        self.compare_video_name = args.compare_video_name
        self.num_cores = args.cra_num_cores

        self.result_dir = os.path.join(self.content_dir, "result", "cra_{}".format(self.input_video_name))
        self.log_dir = os.path.join(self.result_dir, "log")
        self.profile_dir = os.path.join(self.result_dir, "profile")

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.profile_dir, exist_ok=True)

        self.base_cmd = "{} --codec=vp9 --summary --noblit --threads={} --frame-buffers=50  --content-dir={} --input-video={} --dnn-video={} --compare-video={} --prefix=cra_{}".format(args.vpxdec_path, args.cra_num_threads, self.content_dir, self.input_video_name, self.dnn_video_name, self.compare_video_name, self.input_video_name)
        if args.cra_total_frames is not None:
            self.base_cmd = "{} --limit={}".format(self.base_cmd, args.cra_total_frames)

    def prepare_frame_index(self):
        #run a decoder to generate a log of frame index information
        current_frame_index = 0
        super_frame_index = 0
        cmd = "{} --decode-mode=0 --save-metadata".format(self.base_cmd)
        os.system(cmd)

        #load the log and read frame index information
        self.frame_index_list = []
        metadata_log_path = os.path.join(self.log_dir, "metadata_thread01.log")
        with open(metadata_log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                metadata = line.split('\t')
                current_frame_index = int(metadata[0])
                super_frame_index = int(metadata[1])
                frame_index = FrameIndex(current_frame_index, super_frame_index)
                self.frame_index_list.append(frame_index)

    #generate cache profiles for CRA: 1 of |GOP| frames is selceted as an anchor point
    def save_cache_profile(self, current_frame_index, super_frame_index):
        cache_profile_path = os.path.join(self.profile_dir, get_profile_name(current_frame_index, super_frame_index))
        with open(cache_profile_path, "wb") as f:
            byte_value = 0
            for i, frame_index in enumerate(self.frame_index_list):
                if frame_index.current_frame_index == current_frame_index and frame_index.super_frame_index == super_frame_index:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frame_index_list) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def prepare_cache_profiles(self):
        for frame_index in self.frame_index_list:
            self.save_cache_profile(frame_index.current_frame_index, frame_index.super_frame_index)

    #TODO: use Process and Queue for parallel decoding on multi-cores
    #run the cache profiles & measure the quality
    def run_cache_profiles(self):
        for frame_index in self.frame_index_list:
            cache_profile_name = get_profile_name(frame_index.current_frame_index, frame_index.super_frame_index)
            cache_profile_path = os.path.join(self.profile_dir, cache_profile_name)
            cmd = "{} --decode-mode=2 --dnn-mode=2 --cache-policy=1 --cache-profile={} --save-quality --save-metadata".format(self.base_cmd, cache_profile_path)
            os.system(cmd)

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

    #calculate cache erosion metrics and save these in a log file
    def analyze_cache_erosion(self):
        #TODO: 1. load quality results, 2. calculate cache erosion metrics, 3. save these metrics
        dnn_quality = self.load_dnn_quality()
        cache_quality = self.load_cache_quality()
        cache_erosion_metrics = {}

        #TODO: add input arguments called cra_window_size (multiple)
        #TODO: calculate cache erosion metrics for each cra_widow_size
        #TODO: save a cache erosion log

        cache_erosion_log_path = os.path.join(self.log_dir, "cache_erosion.log")
        #with open(cache_erosion_log_path, "wb") as f:
        #    for i in range(self.total_frames):

if __name__ == '__main__':
    #TODO: check whether cache profile is corretely generated by loading it in libvpx
    cra = CacheErosionAnalyzer()
    cra.prepare_frame_index()
    cra.prepare_cache_profiles()
    cra.run_cache_profiles()
