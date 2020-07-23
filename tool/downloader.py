import os
import logging
import shutil
import sys
import argparse

def get_download_cmd(url):
    return cmd

#1,2,3: 2020.06.18
def get_video_url(content, index):
    url = None
    if content == 'product_review': #keywork: product_review
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=2rsXULPbAR0"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=YhysZu9jOt0"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=qY2BdjhPmSE"
    elif content == 'how_to': #keyword: how-to
        if index == 0:
            pass
        elif index == 1:
            #url = "https://www.youtube.com/watch?v=mJD30Y4PwV0"
            url = "https://www.youtube.com/watch?v=pFeQnVvS-yI"
        elif index == 2:
            #url = "https://www.youtube.com/watch?v=Tq-o_296vz4"
            url = "https://www.youtube.com/watch?v=UlKkQ9qnmzY"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=pFeQnVvS-yI"
    elif content == 'vlogs':
        if index == 0:
            pass
        elif index == 1:
            #url = "https://www.youtube.com/watch?v=AoGXdUePwro"
            url = "https://www.youtube.com/watch?v=07XFuFlRvj8"
        elif index == 2:
            #url = "https://www.youtube.com/watch?v=rQKyds1j5C0"
            url = "https://www.youtube.com/watch?v=ibGJXBCTgr4"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=07XFuFlRvj8"
    elif content == 'skit':
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=V0f3IXzc530"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=iBr6k71-w1s"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=PPquOgXjGvo"
    elif content == 'game_play': #keyword: gameplay
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=_56DGiboFF8"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=btmN-bWwv0A"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=nVhXp6FX7Y4"
    elif content == 'haul':
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=_3RVTSpno7Q"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=8DY1dfAZCzQ"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=0HWuSCm3V5M"
    elif content == 'challenge':
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=ZCg9xHNPR3k"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=6T1g_CtvszI"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=FNU_MPQnMY8"
    elif content == 'education': #keyword: education
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=rPe4yziWiOg"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=yC-QHB6EGFM"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=Ny9ZYjAYCB0"
    elif content == 'favorite':
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=9ALj1JxO7e0"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=FXZhEGoS6x8"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=ap4giHCbjYI"
    elif content == 'unboxing':
        if index == 0:
            pass
        elif index == 1:
            url = "https://www.youtube.com/watch?v=l0DoQYGZt8M"
        elif index == 2:
            url = "https://www.youtube.com/watch?v=ouOUyiTCPWE"
        elif index == 3:
            url = "https://www.youtube.com/watch?v=87INnNz4Jzw"
    return url

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--youtubedl_path', type=str, default='/usr/local/bin/youtube-dl')
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--index', type=int, required=True)

    args = parser.parse_args()

    url = get_video_url(args.content, args.index)
    print(args.content, args.index)
    assert(url is not None)
    video_path= os.path.join(args.video_dir, '{}{}.webm'.format(args.content, args.index))
    cmd = "{} -f 'bestvideo[height=2160][fps=30][ext=webm]' -o {} {}".format(args.youtubedl_path, video_path, url)
    os.system(cmd)
