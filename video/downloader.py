import os
import logging
import shutil
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--content_name', type=str, required=True)
parser.add_argument('--youtubedl_path', type=str, required=True)
parser.add_argument('--url', type=str, required=True)

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
