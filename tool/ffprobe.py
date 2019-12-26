import shlex
import subprocess
import json

def profile_video(video_path):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(video_path)

    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    #height, width
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']

    #fps
    fps_line = ffprobeOutput['streams'][0]['avg_frame_rate']
    frame_rate = float(fps_line.split('/')[0]) / float(fps_line.split('/')[1])

    result = {}
    result['height'] = height
    result['width'] = width
    result['frame_rate'] = frame_rate

    return result
