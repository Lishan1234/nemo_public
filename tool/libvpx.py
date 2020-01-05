import math

class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index

    @property
    def name(self):
        return '{}.{}'.format(self.video_index, self.super_index)

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.video_index == other.video_index and self.super_index == other.super_index:
                return True
            else:
                return False
        else:
            return False

class CacheProfile():
    def __init__(self, frames, cache_profile, save_dir, name):
        assert (frames is None or cache_profile is None)

        if frames is not None:
            self.frames = frames
            self.anchor_points = []
            self.estimated_quality = None
            self.measured_quality = None

        if cache_profile is not None:
            self.frames = cache_profile.frames
            self.anchor_points = cache_profile.anchor_points
            self.estimated_quality = cache_profile.estimated_quality
            self.measured_quality = cache_profile.measured_quality

        self.save_dir = save_dir
        self.name = name

    @classmethod
    def fromframes(cls, frames, save_dir, name):
        return cls(frames, None, save_dir, name)

    @property
    def path(self):
        return os.path.join(self.save_dir, self.name)

    def fromcacheprofile(cls, cache_profile, save_dir, name):
        return cls(None, cache_profile, save_dir, name)

    def add_anchor_point(self, frame, quality=None):
        self.anchor_points.append(frame)
        self.quality = quality

    def count_anchor_points(self):
        return len(self.anchor_points)

    def set_estimated_quality(self, quality):
        self.estimated_quality = quality

    def set_measured_quality(self, quality):
        self.measured_quality = quality

    def save(self):
        path = os.path.join(self.save_dir, self.name)
        with open(path, "wb") as f:
            byte_value = 0
            for i, frame in enumerate(cache_profile.frames):
                if frame in cache_profile.anchor_points:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(cache_profile.frames) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

#ref: https://developers.google.com/media/vp9/settings/vod
def get_num_threads(resolution):
    tile_size = 256
    if resolution >= tile_size:
        num_tiles = resolution // tile_size
        log_num_tiles = math.floor(math.log(num_tiles, 2))
        num_threads = (2**log_num_tiles) * 2
    else:
        num_threads = 2
    return num_threads

if __name__ == '__main__':
    frame_list = [Frame(0,1)]
    frame1 = Frame(0,1)
    print(frame1 == frame_list[0])
