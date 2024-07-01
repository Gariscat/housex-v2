"""import tkinter as tk
from pydub import AudioSegment
from pydub.playback import play
import threading
import time

class AudioPlayer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.audio = AudioSegment.from_file(filepath)
        self.playing = False
        self.paused = False
        self.current_time = 0

        self.root = tk.Tk()
        self.setup_ui()
        self.root.mainloop()

    def setup_ui(self):
        self.root.title("Audio Player")

        # Title Label
        self.title_label = tk.Label(self.root, text=self.filepath.split("/")[-1])
        self.title_label.pack(pady=10)

        # Comment Section
        self.comment_label = tk.Label(self.root, text="Comments:")
        self.comment_label.pack(pady=5)
        self.comment_text = tk.Text(self.root, height=5, width=50)
        self.comment_text.pack(pady=5)

        # Progress Bar
        self.progress = tk.Scale(self.root, from_=0, to=len(self.audio), orient=tk.HORIZONTAL, length=400, command=self.set_position)
        self.progress.pack(pady=20)

        # Control Buttons
        self.play_button = tk.Button(self.root, text="Play", command=self.play_audio)
        self.play_button.pack(side=tk.LEFT, padx=10)
        self.pause_button = tk.Button(self.root, text="Pause", command=self.pause_audio)
        self.pause_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=10)

    def play_audio(self):
        if not self.playing:
            self.playing = True
            self.thread = threading.Thread(target=self.playback)
            self.thread.start()
        else:
            self.paused = False

    def playback(self):
        while self.playing and self.current_time < len(self.audio):
            if not self.paused:
                segment = self.audio[self.current_time:self.current_time + 1000]
                play(segment)
                self.current_time += 1000
                self.progress.set(self.current_time)
                time.sleep(1)
            else:
                time.sleep(0.1)

    def pause_audio(self):
        self.paused = True

    def stop_audio(self):
        self.playing = False
        self.paused = False
        self.current_time = 0
        self.progress.set(self.current_time)

    def set_position(self, value):
        self.current_time = int(value)
        if self.paused:
            self.play_audio()
        else:
            self.paused = True

if __name__ == "__main__":
    ap = "/Users/ca7ax/Downloads/audio-100-test/Where You Wanna Be - R3HAB _ Elena Temnikova.ogg"
    AudioPlayer(ap)"""

import tkinter as tk
from tkinter import filedialog
import pygame
import threading

class AudioPlayer:
    def __init__(self, filepath):
        self.filepath = filepath
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        self.playing = False
        self.paused = False
        self.user_dragging = False

        self.root = tk.Tk()
        self.setup_ui()
        self.root.mainloop()

    def setup_ui(self):
        self.root.title("Audio Player")

        # Title Label
        self.title_label = tk.Label(self.root, text=self.filepath.split("/")[-1])
        self.title_label.pack(pady=10)

        # Comment Section
        self.comment_label = tk.Label(self.root, text="Comments:")
        self.comment_label.pack(pady=5)
        self.comment_text = tk.Text(self.root, height=5, width=50)
        self.comment_text.pack(pady=5)

        # Progress Bar
        self.progress = tk.Scale(self.root, from_=0, to=pygame.mixer.Sound(self.filepath).get_length(), orient=tk.HORIZONTAL, length=400, command=self.set_position)
        self.progress.pack(pady=20)

        # Control Buttons
        self.play_button = tk.Button(self.root, text="Play", command=self.play_audio)
        self.play_button.pack(side=tk.LEFT, padx=10)
        self.pause_button = tk.Button(self.root, text="Pause", command=self.pause_audio)
        self.pause_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.progress.bind("<ButtonPress-1>", self.start_drag)
        self.progress.bind("<ButtonRelease-1>", self.stop_drag)

        self.update_progress()

    def start_drag(self, event):
        self.user_dragging = True

    def stop_drag(self, event):
        self.user_dragging = False

    def play_audio(self):
        if not self.playing:
            pygame.mixer.music.play()
            self.playing = True
        elif self.paused:
            pygame.mixer.music.unpause()
        self.paused = False

    def pause_audio(self):
        pygame.mixer.music.pause()
        self.paused = True

    def stop_audio(self):
        pygame.mixer.music.stop()
        self.playing = False
        self.paused = False
        self.progress.set(0)

    def set_position(self, value):
        if self.user_dragging:
            pygame.mixer.music.play(start=float(value))
            self.paused = False

    def update_progress(self):
        if self.playing and not self.paused and not self.user_dragging:
            current_pos = pygame.mixer.music.get_pos() / 1000.0
            self.progress.set(current_pos)
        self.root.after(1000, self.update_progress)


if __name__ == '__main__':
    ap = "/Users/ca7ax/Downloads/audio-100-test/Where You Wanna Be - R3HAB _ Elena Temnikova.ogg"
    AudioPlayer(ap)
