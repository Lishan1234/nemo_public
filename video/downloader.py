import os
import logging
import shutil
import sys

from option import args

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
"""
dataset-[content name]-[video]-(video data, manifest file)
       -[content-video name metadata]
e.g., content name: game-lol, game-dota2
e.g., video name: 2160p.webm
"""
#Downloader

#Input: (video link, content name) - save in a file & load, parse to read

#Check: youtube-dl

#1. Download a 4K video (if not exist) with name "2160p.webm"

#2. Write a content-video name pair in a file
#   (Update for each download to handle deletion)

#Q. Should we use 24fps or 30fps? 많은 종류가 있는 쪽으로 / Youtube 전문 통계로 확인 가능

def get_download_cmd(url):
    return cmd

class Downloader():
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.content_name = args.content_name
        self.content_dir = os.path.join(self.dataset_dir, self.content_name)
        self.video_dir = os.path.join(self.content_dir, "video")
        self.url = args.url

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        youtube_dl_path = shutil.which("youtube-dl")
        if youtube_dl_path is None:
            logging.error("youtube-dl does not exist")
            sys.exit()

    def download(self):
        cmd = '{} -f "bestvideo[height=2160][fps=30]" -o "{}/2160p.%(ext)s" {}'.format(youtube_dl_path, self.video_dir, self.url)
        logging.info("cmd: {}".format(cmd))
        os.system(cmd)

if __name__ == '__main__':
    downloader = Downloader(args)
    downloader.download()
