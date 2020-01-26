import shlex
import subprocess
import json
import os

def profile_video(video_path):
    cmd = "ffprobe -v quiet -print_format json -show_streams -show_entries format"
    args = shlex.split(cmd)
    args.append(video_path)

    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    #height, width
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']
    duration = float(ffprobeOutput['format']['duration'])

    #fps
    fps_line = ffprobeOutput['streams'][0]['avg_frame_rate']
    frame_rate = float(fps_line.split('/')[0]) / float(fps_line.split('/')[1])

    result = {}
    result['height'] = height
    result['width'] = width
    result['frame_rate'] = frame_rate
    result['duration'] = duration

    return result

class FFmpegOption():
    def __init__(self, filter_type, filter_fps, upsample):
        if filter_type not in ['key', 'uniform', 'none']:
            raise ValueError('filter type is not valid: {}'.format(filter_type))
        if filter_type is 'uniform' and filter_fps is None:
            raise ValueError('filter fps is not set: {}'.format(filter_fps))
        #if upsample not in ['bilinear']:
        #    raise ValueError('upsample is not valid: {}'.format(upsample))

        self.filter_type = filter_type
        self.filter_fps = filter_fps
        self.upsample = upsample

    def summary(self, video_name):
        if self.filter_type == 'key':
            return '{}.key'.format(video_name)
        elif self.filter_type == 'uniform':
            return '{}.uniform_{:.2f}'.format(video_name, self.filter_fps)
        elif self.filter_type == 'none':
            return video_name

    def filter(self):
        if self.filter_type == 'key':
            return '-vf "select=eq(pict_type\,I)" -vsync vfr'
        elif self.filter_type == 'uniform':
            return '-vf fps={}'.format(self.filter_fps)
        elif self.filter_type == 'none':
            return ''

    def filter_rescale(self, width, height):
        if self.filter_type == 'key':
            return '-vf "select=eq(pict_type\,I)",scale={}:{} -vsync vfr -sws_flags {}'.format(width, height, self.upsample)
        elif self.filter_type == 'uniform':
            return '-vf fps={},scale={}:{} -sws_flags {}'.format(self.filter_fps, width, height, self.upsample)
        elif self.filter_type == 'none':
            return '-vf scale={}:{} -sws_flags {}'.format(width, height, self.upsample)

class VideoMetadata():
    def __init__(self, video_format, start_time, duration):
        self.video_format = video_format
        self.start_time = start_time
        self.duration = duration

    #TODO: add bitrate and vidoe_format
    def summary(self, resolution, is_encoded):
        name = '{}p'.format(resolution)
        if self.start_time is not None:
            name += '_s{}'.format(self.start_time)
        if self.duration is not None:
            name += '_d{}'.format(self.duration)
        if is_encoded: name += '_encoded'
        name += '.{}'.format(self.video_format)
        return name
