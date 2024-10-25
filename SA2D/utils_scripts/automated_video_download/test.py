from datetime import datetime, timezone
import time
import os
import subprocess
import signal

def download_video(video_url, dest_path):
    command = ['yt-dlp', '-o', dest_path, '--no-live-from-start', video_url, '--hls-use-mpegts']

    pro = subprocess.Popen(command) 
    time.sleep(8)
    os.kill(pro.pid, signal.CTRL_C_EVENT)

def extract_id(yt_link):
    video_id = yt_link.split('=')[-1]
    return video_id

def current_time():
    # Get current time in UTC
    current_time_utc = datetime.now(timezone.utc)

    # format time
    formatted_utc_time = current_time_utc.strftime("%Y-%m-%d_%H:%M:%S_UTC")

    return formatted_utc_time

def make_dest_path(yt_link):
    video_id = extract_id(yt_link)
    video_name = video_id + "_" + current_time() + ".%(ext)s"
    path = ".//videos//" + video_id + "//" + video_name

    return path



yt_links = [
    'https://www.youtube.com/watch?v=3LXQWU67Ufk',
]

video_count = 0
index = 0
while video_count < 500:
    if (index == len(yt_links)):
        index = 0
    
    try:
        video_url = yt_links[index]
        print(make_dest_path(video_url))
        download_video(video_url, make_dest_path(video_url))
#     except:
#         print("FAILED!!!!!!!!!!!!", video_url)
    except Exception as e:
        # 处理未知异常的代码
        print(f"An unknown error occurred: {e}")

    video_count += 1
    index += 1
    print("Downloaded", video_count, "video(s)")

    for i in range(20):
        try:
            time.sleep(1)
        except:
            print("sleep", i)