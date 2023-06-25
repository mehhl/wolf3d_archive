import vlc
import time
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class MyHandler(PatternMatchingEventHandler):
    def __init__(self, list_player):
        PatternMatchingEventHandler.__init__(self, patterns=["*.mp4", "*.avi"], ignore_directories=True)  # Add more or change file extensions as per your needs.
        self.list_player = list_player
        self.current_file = None

    def on_created(self, event):
        print(f"{event.src_path} has been added!")
        if self.current_file is None or Path(event.src_path).stat().st_ctime > self.current_file.stat().st_ctime:
            self.play_media(Path(event.src_path))

    def play_media(self, file):
        self.current_file = file
        media = vlc.Media(str(file))
        media_list = vlc.MediaList([media])
        self.list_player.set_media_list(media_list)
        self.list_player.play()
        print(f"Now playing {file}.")

fp = Path(os.getcwd()) / 'final_run_log' / 'wandb'
fp = fp / 'offline-run-20230624_183505-yw5h1dgx'
fp = fp / 'files' / 'media' / 'videos'

# list all vids
videos = [name for name in fp.iterdir() if name.is_file()]

# sort by creation time
videos.sort(key=lambda x: x.stat().st_ctime, reverse=True)

# Create player and play the most recent video
instance = vlc.Instance()
list_player = instance.media_list_player_new()
media = vlc.Media(str(videos[0]))
media_list = vlc.MediaList([media])
list_player.set_media_list(media_list)
list_player.set_playback_mode(vlc.PlaybackMode.loop)  # Enable looping
list_player.play()

# Create the event handler
event_handler = MyHandler(list_player)
observer = Observer()
observer.schedule(event_handler, str(fp), recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
