class CacheProfile():
    def __init__(self, frame_list, quality_list):
        self.frame_list = frame_list
        self.quality_list = quality_list

    def __len__(self):
        return len(self.frame_list)

    def count_anchor_points(self):
        count = 0
        for frame in self.frame_list:
            if frame.apply_dnn is True:
                count += 1
        return count

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

    #Save a cache profile
    def save(self, path):
        pass

class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index
        self.apply_dnn = False

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

class AnchorPoint():
    def __init__(self, video_index, super_index, sr_quality, bilinear_quality):
        self.video_index = video_index
        self.super_index= super_index
        self.sr_quality = sr_quality
        self.bilinear_quality = biliner_quality

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

#Add a single anchor point to a cache profile
def add_anchor_point(cache_profile, anchor_point):
    pass

#TODO: max-based
def estimate_quality(cache_profile, anchor_poist):
    pass

#TODO: DAG-basd
def calculate_quality():
    pass
