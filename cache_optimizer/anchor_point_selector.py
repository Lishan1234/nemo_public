import os
import abc
import copy
import numpy as np
import logging
import sys
import math
import struct

from option import args
from cache_erosion_analyzer import CRA

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

"""
0.0: use estimate_quality()
0.1: use calculate_quality()
"""

VERSION = "0.0"

class Quality():
    def __init__(self, mse, max_value=255.0):
        self.psnr = psnr
        self.mse = mse
        self.max_value = max_value

class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

class CacheProfile():
    def __init__(self):
        self.anchor_points = []
        self.sr_cache_quality = {}
        self.sr_cache_quality['estimated'] = None
        self.sr_cache_quality['measured'] = None

    def add_anchor_point(self, anchor_point):
        self.anchor_points.append(anchor_point)
        self.sr_cache_quality['estimated'] = estimate_quality(self, anchor_point)

    def count_anchor_points(self):
        return len(self.anchor_points)

    def get_profile_name(self, prefix, start_idx=None, end_idx=None):
        profile_name = ""
        if prefix:
            profile_name += "{}_ap{}".format(prefix, self.count_anchor_points())
        else:
            profile_name += "ap{}".format(self.count_anchor_points())
        if start_idx is not None:
            profile_name += "_s{}".format(start_idx)
        if end_idx is not None:
            profile_name += "_e{}".format(end_idx)
        return profile_name

    def is_anchor_point(self, video_index, super_index):
        for anchor_point in self.anchor_points:
            if anchor_point.video_index == video_index and anchor_point.super_index == super_index:
                return True

        return False

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

    #TODO: Save a cache profile
    def save(self, path):
        pass

class Video():
    def __init__(self, sr_quality, bilinear_quality, bilinear_cache_quality=None):
        self.sr_quality = sr_quality
        self.bilinear_quality = bilinear_quality
        self.bilinear_cache_quality = bilinear_cache_quality

class AnchorPoint():
    def __init__(self, video_index, super_index, sr_cache_quality):
        self.video_index = video_index
        self.super_index= super_index
        self.sr_cache_quality = sr_cache_quality

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

def estimate_quality(cache_profile, anchor_point):
    if cache_profile.sr_cache_quality['estimated'] is not None:
        assert len(cache_profile.sr_cache_quality['estimated']) == len(anchor_point.sr_cache_quality)
        return np.maximum(cache_profile.sr_cache_quality['estimated'], anchor_point.sr_cache_quality)
    else:
        return anchor_point.sr_cache_quality

#TODO: Calculate exact merged quality between two DAGs
def calculate_quality(cache_profile, anchor_point, video):
    pass

#TODO: Measure cache profile's quality
def measure_quality(cache_profile):
    pass

#Anchor Point Selector
class APS():
    __metaclass__ = abc.ABCMeta
    def __init__(self, args):
        self.content_dir = args.content_dir
        self.input_video_name = args.input_video_name
        self.dnn_video_name = args.dnn_video_name
        self.compare_video_name = args.compare_video_name
        self.num_cores = args.cra_num_cores
        self.result_dir = os.path.join(self.content_dir, "result", self.input_video_name)
        self.log_dir = os.path.join(self.result_dir, "log")
        self.profile_dir = os.path.join(self.result_dir, "profile")
        self.base_cmd = "{} --codec=vp9 --summary --noblit --threads={} --frame-buffers=50  --content-dir={} --input-video={} --dnn-video={} --compare-video={}".format(args.vpxdec_path, args.cra_num_threads, self.content_dir, self.input_video_name, self.dnn_video_name, self.compare_video_name)
        if args.num_frames is not None:
            self.base_cmd = "{} --limit={}".format(self.base_cmd, args.num_frames)

        self.cache_profiles = []
        self.anchor_points = []
        self.frames = []
        self.sr_quality = []
        self.bilinear_quality = []
        self.bilinear_cache_quality = []
        self.start_idx = args.aps_start_idx
        self.end_idx = args.aps_end_idx

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.profile_dir, exist_ok=True)

    @abc.abstractmethod
    def select_anchor_points(self):
        pass

    #TODO: load a power profile
    def prepare_power_consumption(self):
        pass

    def is_valid_anchor_point(self, video_index):
        bt = True if (self.start_idx is None) else (video_index >= self.start_idx)
        lt = True if (self.end_idx is None) else (video_index <= self.end_idx)
        return bt and lt

    def prepare_frame(self):
        metadata_log_path = os.path.join(self.log_dir, "metadata_thread01")
        video_index = 0
        super_index = 0

        #run a decoder to generate a log of frame index information
        cmd = "{} --decode-mode=0 --save-metadata".format(self.base_cmd)
        os.system(cmd)

        #load the log and read frame index information
        with open(metadata_log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                metadata = line.split('\t')
                video_index = int(metadata[0])
                super_index = int(metadata[1])
                frame = Frame(video_index, super_index)
                self.frames.append(frame)
                logging.debug("video_index: {}, super_index: {}".format(video_index, super_index))

    #TODO: load MSE
    def prepare_anchor_point(self):
        #load sr-cache quality
        for frame in self.frames:
            if self.is_valid_anchor_point(frame.video_index):
                log_name = "quality_cache_{}".format(CRA.get_profile_name(frame.video_index, frame.super_index))
                log_path= os.path.join(self.log_dir, log_name)

                sr_cache_quality = []
                with open(log_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        sr_cache_quality.append(float(line.split('\t')[1]))

                anchor_point = AnchorPoint(frame.video_index, frame.super_index, sr_cache_quality)
                frame = Frame(frame.video_index, frame.super_index)
                self.anchor_points.append(anchor_point)

                logging.debug("{}-Anchor point: video_index {}, super_index {}, average psnr {}".format(self.__class__.__name__, anchor_point.video_index, anchor_point.super_index, np.round(np.average(anchor_point.sr_cache_quality),2)))

    #TODO: load sr, bilinear, bilinear-cache quality
    def prepare_video_quality(self):
        #bilinear_quliaty
        cmd = "{} --decode-mode=3 --save-quality".format(self.base_cmd)
        os.system(cmd)
        log_path = os.path.join(self.log_dir, "quality_bilinear")
        with open(log_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.bilinear_quality.append(float(line.split('\t')[1]))
        logging.debug("bilinear_quality: {}".format(self.bilinear_quality))

        #sr_quality
        cmd = "{} --decode-mode=1 --dnn-mode=2 --save-quality".format(self.base_cmd)
        os.system(cmd)
        log_path = os.path.join(self.log_dir, "quality_sr")
        with open(log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.sr_quality.append(float(line.split('\t')[1]))
        logging.debug("sr_quality: {}".format(self.sr_quality))

        #bilinear_quliaty
        cmd = "{} --decode-mode=2 --save-quality".format(self.base_cmd)
        os.system(cmd)
        log_path = os.path.join(self.log_dir, "quality_bilinear")
        with open(log_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.bilinear_cache_quality.append(float(line.split('\t')[1]))
        logging.debug("bilinear_cache_quality: {}".format(self.bilinear_cache_quality))

    def save_cache_profiles(self):
        for cache_profile in self.cache_profiles:
            profile_name = cache_profile.get_profile_name(self.__class__.__name__, self.start_idx, self.end_idx)
            profile_path = os.path.join(self.profile_dir, profile_name)
            with open(profile_path, "wb") as f:
                byte_value = 0
                for i, frame in enumerate(self.frames):
                    if cache_profile.is_anchor_point(frame.video_index, frame.super_index):
                        byte_value += 1 << (i % 8)

                    if i % 8 == 7:
                        f.write(struct.pack("=B", byte_value))
                        byte_value = 0

                if len(self.frames) % 8 != 0:
                    f.write(struct.pack("=B", byte_value))

    def run_cache_profiles(self):
        for cache_profile in self.cache_profiles:
            profile_name = cache_profile.get_profile_name(self.__class__.__name__, self.start_idx, self.end_idx)
            profile_path = os.path.join(self.profile_dir, profile_name)
            cmd = "{} --decode-mode=2 --dnn-mode=2 --cache-policy=1 --cache-profile={} --save-quality --save-metadata".format(self.base_cmd, profile_path)
            os.system(cmd)

    def prepare_cache_quality(self):
        for cache_profile in self.cache_profiles:
            profile_name = cache_profile.get_profile_name(self.__class__.__name__, self.start_idx, self.end_idx)
            log_name = "quality_cache_{}".format(profile_name)
            log_path= os.path.join(self.log_dir, log_name)

            cache_profile.sr_cache_quality['measured'] = []
            with open(log_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    cache_profile.sr_cache_quality['measured'].append(float(line.split('\t')[1]))

    def get_log_name(self, prefix):
        log_name = "{}".format(self.__class__.__name__)
        log_name += "_{}".format(prefix)
        if self.start_idx is not None: log_name += "_s{}".format(self.start_idx)
        if self.end_idx is not None: log_name += "_e{}".format(self.end_idx)

        return log_name

    def evaluate_anchor_points(self):
        log_name = self.get_log_name("anchor_points")
        log_path = os.path.join(self.log_dir, log_name)

        with open(log_path, "w") as f:
            for anchor_point in self.anchor_points:
                psnr_gain = np.subtract(anchor_point.sr_cache_quality, self.bilinear_cache_quality)
                total_psnr_gain = np.sum(psnr_gain)
                f.write("{}\t{}\t{}\n".format(anchor_point.video_index, anchor_point.super_index, np.round(total_psnr_gain, 2)))

    def evaluate_cache_profiles(self):
        log_name = self.get_log_name("cache_profiles")
        log_path = os.path.join(self.log_dir,log_name)

        with open(log_path, "w") as f:
            for cache_profile in self.cache_profiles:
                start_idx = 0 if self.start_idx is None else self.start_idx
                end_idx = self.frames[-1].video_idx if self.end_idx is None else self.end_idx
                estimated_quality = cache_profile.sr_cache_quality['estimated'][start_idx:end_idx + 1]
                measured_quality = cache_profile.sr_cache_quality['measured'][start_idx:end_idx + 1]
                average_error = np.average(np.absolute(np.subtract(estimated_quality, measured_quality)))

                average_quality = np.average(measured_quality)
                error_quality = np.subtract(self.sr_quality[start_idx:end_idx + 1], measured_quality)
                error_quality_metrics = np.percentile(error_quality, [50, 90, 100], interpolation='nearest')
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(cache_profile.count_anchor_points(), np.round(np.average(measured_quality), 2), np.round(np.average(estimated_quality), 2), np.round(average_error, 2), np.round(error_quality_metrics[0], 2), np.round(error_quality_metrics[1], 2), np.round(error_quality_metrics[2], 2)))

    def get_highest_quality(self):
        start_idx = 0 if self.start_idx is None else self.start_idx
        end_idx = self.frames[-1].video_idx if self.end_idx is None else self.end_idx
        cache_profile = self.cache_profiles[-1]
        return np.average(cache_profile.sr_cache_quality['measured'][start_idx:end_idx+1])

class OptimizedAPS(APS):
    def __init__(self, args):
        super(OptimizedAPS, self).__init__(args)
        self.threshold = args.aps_threshold
        self.current_cache_profile = CacheProfile()

    def select_anchor_points(self):
        total_gain = 0

        while len(self.anchor_points) > 0:
            anchor_point, psnr_gain = self.select_anchor_point()

            if psnr_gain < self.threshold:
                print("Stop search at {} anchor points".format(self.current_cache_profile.count_anchor_points()))
                break

            self.current_cache_profile.add_anchor_point(anchor_point)
            self.cache_profiles.append(copy.deepcopy(self.current_cache_profile))

            logging.debug("{}-Cache profile: {} anchor points, average psnr {}, psnr gain: {}dB".format(self.__class__.__name__, self.current_cache_profile.count_anchor_points(), np.round(np.average(self.current_cache_profile.sr_cache_quality['estimated']), 2), np.round(psnr_gain, 2)))

    #TODO: MSE-based
    def select_anchor_point(self):
        max_psnr_gain = 0
        index = 0

        if self.current_cache_profile.sr_cache_quality['estimated'] is None:
            prev_psnr_sum = 0
        else:
            prev_psnr_sum = np.sum(self.current_cache_profile.sr_cache_quality['estimated'])

        for i, anchor_point in enumerate(self.anchor_points):
            curr_psnr_sum = np.sum(estimate_quality(self.current_cache_profile, anchor_point))
            psnr_gain = curr_psnr_sum - prev_psnr_sum
            if psnr_gain > max_psnr_gain:
                max_psnr_gain = psnr_gain
                index = i

        return self.anchor_points.pop(index), max_psnr_gain

#Option: a) Number of anchor points, b) Average PSNR
class BaselineAPS(APS):
    def __init__(self, args, threshold):
        super(BaselineAPS, self).__init__(args)
        self.threshold = threshold
        self.current_cache_profile = None

    def select_anchor_points(self):
        total_gain = 0
        num_anchor_points = 1
        start_idx = 0 if self.start_idx is None else self.start_idx
        end_idx = self.frames[-1].video_idx if self.end_idx is None else self.end_idx

        while num_anchor_points < len(self.anchor_points):
            self.current_cache_profile = CacheProfile()
            for i in range(num_anchor_points):
                index = i * math.floor(len(self.anchor_points)/num_anchor_points)
                self.current_cache_profile.add_anchor_point(self.anchor_points[index])

            if np.average(self.current_cache_profile.sr_cache_quality['estimated'][start_idx:end_idx + 1]) >= self.threshold:
                print("Stop search at {} anchor points".format(self.current_cache_profile.count_anchor_points()))
                break

            self.cache_profiles.append(copy.deepcopy(self.current_cache_profile))
            num_anchor_points += 1

            logging.debug("{}-Cache profile: {} anchor points, {}dB".format(self.__class__.__name__, self.current_cache_profile.count_anchor_points(), np.round(np.average(self.current_cache_profile.sr_cache_quality['estimated'][start_idx:end_idx + 1]),2)))

if __name__ == '__main__':
    aps = OptimizedAPS(args)
    aps.prepare_frame()
    aps.prepare_video_quality()
    aps.prepare_anchor_point()
    aps.select_anchor_points()
    aps.evaluate_anchor_points()
    aps.save_cache_profiles()
    aps.run_cache_profiles()
    aps.prepare_cache_quality()
    aps.evaluate_cache_profiles()
    threshold = aps.get_highest_quality()
    print("OptimizedAPS threshold is {}".format(threshold))

    aps = BaselineAPS(args, threshold)
    aps.prepare_frame()
    aps.prepare_video_quality()
    aps.prepare_anchor_point()
    aps.select_anchor_points()
    aps.save_cache_profiles()
    aps.run_cache_profiles()
    aps.prepare_cache_quality()
    aps.evaluate_cache_profiles()
