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
    def __init__(self, frames, cache_profile):
        assert (frames is None or cache_profile is None)

        if frames is not None:
            self.frames = frames
            self.anchor_points = []
            self.quality = None

        if cache_profile is not None:
            self.frames = cache_profile.frames
            self.anchor_points = cache_profile.anchor_points
            self.quality = cache_profile.quality

    @classmethod
    def fromframes(cls, frames):
        return cls(frames, None)

    def fromcacheprofile(cls, cache_profile):
        return cls(None, cache_profile)

    def add_anchor_point(self, frame, quality=None):
        self.anchor_points.append(frame)
        self.quality = quality

    def count_anchor_points(self):
        return len(self.anchor_points)

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

if __name__ == '__main__':
    frame_list = [Frame(0,1)]
    frame1 = Frame(0,1)
    print(frame1 == frame_list[0])
